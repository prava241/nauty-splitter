/**
 * nauty_parallel.c - Implementation of Cilk parallel interface for nauty
 * 
 * This file provides parallel versions of the key nauty algorithms using
 * Cilk splitters for thread-local state and reducers for statistics.
 */

#include "nauty_parallel.h"

#ifdef ENABLE_PARALLELISM

/* Splitter copy functions */
static void splitter_int_copy(void *dst, void *src) {
    *(int *)dst = *(int *)src;
}

static void splitter_double_copy(void *dst, void *src) {
    *(double *)dst = *(double *)src;
}

static void splitter_set_copy(void *dst, void *src) {
    memcpy(dst, src, sizeof(set));
}

static void splitter_short_copy(void *dst, void *src) {
    *(short *)dst = *(short *)src;
}

/* Forward declarations */
static int firstpathnode_parallel(graph *g, int level, int numcells, 
                                int m_val, int n_val, graph *canong, optionblk *options);
static int othernode_parallel(graph *g, int level, int numcells, 
                            int m_val, int n_val, graph *canong, optionblk *options);

/* Reducers for statistics tracking */
CILK_C_DECLARE_REDUCER(unsigned long) reducer_numnodes = 
    CILK_C_INIT_REDUCER_OPADD(unsigned long, 0, +, unsigned long, 0);

CILK_C_DECLARE_REDUCER(unsigned long) reducer_numbadleaves = 
    CILK_C_INIT_REDUCER_OPADD(unsigned long, 0, +, unsigned long, 0);

CILK_C_DECLARE_REDUCER(unsigned long) reducer_tctotal = 
    CILK_C_INIT_REDUCER_OPADD(unsigned long, 0, +, unsigned long, 0);

CILK_C_DECLARE_REDUCER(unsigned long) reducer_canupdates = 
    CILK_C_INIT_REDUCER_OPADD(unsigned long, 0, +, unsigned long, 0);

CILK_C_DECLARE_REDUCER(unsigned long) reducer_invapplics = 
    CILK_C_INIT_REDUCER_OPADD(unsigned long, 0, +, unsigned long, 0);

CILK_C_DECLARE_REDUCER(unsigned long) reducer_invsuccesses = 
    CILK_C_INIT_REDUCER_OPADD(unsigned long, 0, +, unsigned long, 0);

CILK_C_DECLARE_REDUCER(int) reducer_invarsuclevel = 
    CILK_C_INIT_REDUCER_MIN(int, NAUTY_INFINITY);

CILK_C_DECLARE_REDUCER(int) reducer_maxlevel = 
    CILK_C_INIT_REDUCER_MAX(int, 1);

CILK_C_DECLARE_REDUCER(unsigned long) reducer_numgenerators = 
    CILK_C_INIT_REDUCER_OPADD(unsigned long, 0, +, unsigned long, 0);

CILK_C_DECLARE_REDUCER(int) reducer_errstatus = 
    CILK_C_INIT_REDUCER_OPADD(int, 0, +, int, 0);

typedef struct {
    double grpsize1;
    int grpsize2;
} groupsize_t;

static void groupsize_identity(void* reducer, void* value) {
    ((groupsize_t*)value)->grpsize1 = 1.0;
    ((groupsize_t*)value)->grpsize2 = 0;
}

static void groupsize_reduce(void* reducer, void* left, void* right) {
    multiply_groups(
        &((groupsize_t*)left)->grpsize1, 
        &((groupsize_t*)left)->grpsize2,
        ((groupsize_t*)right)->grpsize1,
        ((groupsize_t*)right)->grpsize2
    );
}

CILK_C_DECLARE_REDUCER(groupsize_t) reducer_groupsize = 
    CILK_C_INIT_REDUCER(groupsize_t, groupsize_reduce, groupsize_identity, 0);

static splitter_int_t *lab_splitter;     /* Vertex labeling */
static splitter_int_t *ptn_splitter;     /* Partition */
static splitter_int_t *orbits_splitter;  /* Orbit information */
static splitter_set_t *active_splitter;  /* Active cells */
static splitter_set_t *fixedpts_splitter; /* Fixed points */
static splitter_int_t *workperm_splitter; /* Work permutation */
static splitter_short_t *firstcode_splitter; /* First path refinement codes */
static splitter_short_t *canoncode_splitter; /* Canonical refinement codes */
static splitter_int_t *firsttc_splitter; /* First path target cells */

/* Global options */
static boolean getcanon, digraph, domarkers, doschreier;
static optionblk *global_options;
static graph *global_g;
static graph *global_canong;
static int global_m, global_n;

static int alloc_n = 0, alloc_m = 0;

void nauty_parallel_init(int n_val, int m_val) {
    if (n_val <= alloc_n && m_val <= alloc_m) {
        return;
    }
    
    nauty_parallel_cleanup();
    
    global_n = n_val;
    global_m = m_val;
    alloc_n = n_val;
    alloc_m = m_val;
    
    /* Register reducers */
    CILK_C_REGISTER_REDUCER(reducer_numnodes);
    CILK_C_REGISTER_REDUCER(reducer_numbadleaves);
    CILK_C_REGISTER_REDUCER(reducer_tctotal);
    CILK_C_REGISTER_REDUCER(reducer_canupdates);
    CILK_C_REGISTER_REDUCER(reducer_invapplics);
    CILK_C_REGISTER_REDUCER(reducer_invsuccesses);
    CILK_C_REGISTER_REDUCER(reducer_invarsuclevel);
    CILK_C_REGISTER_REDUCER(reducer_maxlevel);
    CILK_C_REGISTER_REDUCER(reducer_groupsize);
    CILK_C_REGISTER_REDUCER(reducer_errstatus);
    CILK_C_REGISTER_REDUCER(reducer_numgenerators);
    
    /* Allocate and initialize splitters */
    lab_splitter = (splitter_int_t*)malloc(n_val * sizeof(splitter_int_t));
    ptn_splitter = (splitter_int_t*)malloc(n_val * sizeof(splitter_int_t));
    orbits_splitter = (splitter_int_t*)malloc(n_val * sizeof(splitter_int_t));
    workperm_splitter = (splitter_int_t*)malloc(n_val * sizeof(splitter_int_t));
    
    firstcode_splitter = (splitter_short_t*)malloc((n_val+2) * sizeof(splitter_short_t));
    canoncode_splitter = (splitter_short_t*)malloc((n_val+2) * sizeof(splitter_short_t));
    firsttc_splitter = (splitter_int_t*)malloc((n_val+2) * sizeof(splitter_int_t));
    
    fixedpts_splitter = (splitter_set_t*)malloc(m_val * sizeof(splitter_set_t));
    active_splitter = (splitter_set_t*)malloc(m_val * sizeof(splitter_set_t));
    
    if (!lab_splitter || !ptn_splitter || !orbits_splitter || !workperm_splitter ||
        !firstcode_splitter || !canoncode_splitter || !firsttc_splitter ||
        !fixedpts_splitter || !active_splitter) {
        fprintf(stderr, "nauty_parallel: memory allocation failed\n");
        exit(1);
    }
    
    /* Initialize splitters with default values */
    for (int i = 0; i < n_val; i++) {
        lab_splitter[i] = CILK_C_INIT_SPLITTER(int, splitter_int_copy, i);
        ptn_splitter[i] = CILK_C_INIT_SPLITTER(int, splitter_int_copy, 0);
        orbits_splitter[i] = CILK_C_INIT_SPLITTER(int, splitter_int_copy, i);
        workperm_splitter[i] = CILK_C_INIT_SPLITTER(int, splitter_int_copy, 0);
    
        CILK_C_REGISTER_SPLITTER(lab_splitter[i]);
        CILK_C_REGISTER_SPLITTER(ptn_splitter[i]);
        CILK_C_REGISTER_SPLITTER(orbits_splitter[i]);
        CILK_C_REGISTER_SPLITTER(workperm_splitter[i]);
    }
    
    /* Initialize control splitters */
    for (int i = 0; i < n_val+2; i++) {
        firstcode_splitter[i] = CILK_C_INIT_SPLITTER(short, splitter_short_copy, 0);
        canoncode_splitter[i] = CILK_C_INIT_SPLITTER(short, splitter_short_copy, 0);
        firsttc_splitter[i] = CILK_C_INIT_SPLITTER(int, splitter_int_copy, -1);
        
        CILK_C_REGISTER_SPLITTER(firstcode_splitter[i]);
        CILK_C_REGISTER_SPLITTER(canoncode_splitter[i]);
        CILK_C_REGISTER_SPLITTER(firsttc_splitter[i]);
    }
    
    /* Initialize set splitters */
    for (int i = 0; i < m_val; i++) {
        set empty_set = 0; /* Initialize with empty set */
        fixedpts_splitter[i] = CILK_C_INIT_SPLITTER(set, splitter_set_copy, empty_set);
        active_splitter[i] = CILK_C_INIT_SPLITTER(set, splitter_set_copy, empty_set);
        
        CILK_C_REGISTER_SPLITTER(fixedpts_splitter[i]);
        CILK_C_REGISTER_SPLITTER(active_splitter[i]);
    }
    
    /* Reset reducers */
    REDUCER_VIEW(reducer_numnodes) = 0;
    REDUCER_VIEW(reducer_numbadleaves) = 0;
    REDUCER_VIEW(reducer_tctotal) = 0;
    REDUCER_VIEW(reducer_canupdates) = 0;
    REDUCER_VIEW(reducer_invapplics) = 0;
    REDUCER_VIEW(reducer_invsuccesses) = 0;
    REDUCER_VIEW(reducer_invarsuclevel) = NAUTY_INFINITY;
    REDUCER_VIEW(reducer_maxlevel) = 1;
    REDUCER_VIEW(reducer_errstatus) = 0;
    REDUCER_VIEW(reducer_numgenerators) = 0;
    
    /* Reset group size reducer */
    groupsize_t grp = {1.0, 0};
    REDUCER_VIEW(reducer_groupsize) = grp;
}

/**
 * Free all parallel construct resources
 */
void nauty_parallel_cleanup(void) {
    if (alloc_n == 0 && alloc_m == 0) {
        return; /* Nothing to clean up */
    }
    
    /* Unregister reducers */
    CILK_C_UNREGISTER_REDUCER(reducer_numnodes);
    CILK_C_UNREGISTER_REDUCER(reducer_numbadleaves);
    CILK_C_UNREGISTER_REDUCER(reducer_tctotal);
    CILK_C_UNREGISTER_REDUCER(reducer_canupdates);
    CILK_C_UNREGISTER_REDUCER(reducer_invapplics);
    CILK_C_UNREGISTER_REDUCER(reducer_invsuccesses);
    CILK_C_UNREGISTER_REDUCER(reducer_invarsuclevel);
    CILK_C_UNREGISTER_REDUCER(reducer_maxlevel);
    CILK_C_UNREGISTER_REDUCER(reducer_groupsize);
    CILK_C_UNREGISTER_REDUCER(reducer_errstatus);
    CILK_C_UNREGISTER_REDUCER(reducer_numgenerators);
    
    /* Free splitters */
    free(lab_splitter);
    free(ptn_splitter);
    free(orbits_splitter);
    free(workperm_splitter);
    free(firstcode_splitter);
    free(canoncode_splitter);
    free(firsttc_splitter);
    free(fixedpts_splitter);
    free(active_splitter);
    
    alloc_n = alloc_m = 0;
}

/**
 * Modified breakout function that uses splitters directly
 * This is called when a new vertex is selected to individualize
 */
static void breakout_splitter(int level, int tc, int tv) {
    int i, j, lab_tc;
    
    /* Read the current partition */
    lab_tc = SPLITTER_READ(lab_splitter[tc]);
    
    /* Individualize: swap the vertices in positions tc and tc+1, etc. */
    for (i = tc; i < global_n && SPLITTER_READ(ptn_splitter[i]) > 0; i++) {
        j = SPLITTER_READ(lab_splitter[i]);
        if (j == tv) break;
    }
    
    /* Since we're using splitters, we need to write each change individually */
    SPLITTER_WRITE(lab_splitter[i], lab_tc);
    SPLITTER_WRITE(lab_splitter[tc], tv);
    SPLITTER_WRITE(ptn_splitter[tc], 0);
    
    /* Update active cells for refinement */
    for (i = 0; i < global_m; i++) {
        set active_i = SPLITTER_READ(active_splitter[i]);
        if (tc < WORDSIZE * i || tc >= WORDSIZE * (i + 1)) {
            SPLITTER_WRITE(active_splitter[i], active_i);
        } else {
            /* Add tc to active set */
            ADDELEMENT(&active_i, tc);
            SPLITTER_WRITE(active_splitter[i], active_i);
            break;
        }
    }
}

/**
 * ask kenny if this function is necessary
 */
static void recover_splitter(int level) {
    int i, j;
    
    /* Find the individualized vertex at this level */
    for (i = 0; i < global_n; i++) {
        if (SPLITTER_READ(ptn_splitter[i]) == 0) {
            if (--level < 0) return;
        }
    }
    
    /* Restore the partition value */
    SPLITTER_WRITE(ptn_splitter[i-1], NAUTY_INFINITY);
}

static int process_branch(int level, int numcells, int tc, int tv) {
    breakout_splitter(level, tc, tv);
    
    int rtnlevel = othernode_parallel(global_g, level+1, numcells+1, 
                                    global_m, global_n, global_canong, global_options);
    
    /* After returning, we would normally recover the partition,
    but with splitters this is unnecessary as each task gets its own view */
    
    return rtnlevel;
}

/**
 * Main entry point for parallel-enabled nauty
 */
void nauty_parallel(graph *g_arg, int *lab, int *ptn, set *active_arg,
    int *orbits_arg, optionblk *options, statsblk *stats_arg,
    set *ws_arg, int worksize, int m_arg, int n_arg, graph *canong_arg) {
    
    nauty_parallel_init(n_arg, m_arg);

    getcanon = options->getcanon;
    digraph = options->digraph;
    domarkers = options->writemarkers;
    doschreier = options->schreier;
    global_options = options;
    global_g = g_arg;
    global_canong = canong_arg;
    
    /* Copy input arrays to splitters */
    for (int i = 0; i < n_arg; i++) {
        SPLITTER_WRITE(lab_splitter[i], lab[i]);
        SPLITTER_WRITE(ptn_splitter[i], ptn[i]);
        SPLITTER_WRITE(orbits_splitter[i], i); /* Initialize orbits */
    }
    
    if (active_arg != NULL) {
        for (int i = 0; i < m_arg; i++) {
            SPLITTER_WRITE(active_splitter[i], active_arg[i]);
        }
    } else {
        set active_local = 0;
        for (int i = 0; i < n_arg; i++) {
            int wi = i / WORDSIZE;
            if (wi < m_arg) {
                active_local = SPLITTER_READ(active_splitter[wi]);
                ADDELEMENT(&active_local, i);
                SPLITTER_WRITE(active_splitter[wi], active_local);
            }
            while (i < n_arg-1 && ptn[i] > 0) i++;
        }
    }
    
    int numcells = 0;
    if (options->defaultptn) {
        numcells = 1;
    } else {
        for (int i = 0; i < n_arg; i++) {
            if (ptn[i] == 0) numcells++;
        }
    }
    
    REDUCER_VIEW(reducer_numnodes) = 0;
    REDUCER_VIEW(reducer_numbadleaves) = 0;
    REDUCER_VIEW(reducer_tctotal) = 0;
    REDUCER_VIEW(reducer_canupdates) = 0;
    REDUCER_VIEW(reducer_invapplics) = 0;
    REDUCER_VIEW(reducer_invsuccesses) = 0;
    REDUCER_VIEW(reducer_invarsuclevel) = NAUTY_INFINITY;
    REDUCER_VIEW(reducer_maxlevel) = 1;
    REDUCER_VIEW(reducer_errstatus) = 0;
    REDUCER_VIEW(reducer_numgenerators) = 0;
    
    groupsize_t grp = {1.0, 0};
    REDUCER_VIEW(reducer_groupsize) = grp;
    
    if (doschreier) {
    }
    
    int rtnlevel = cilk_spawn firstpathnode_parallel(g_arg, 1, numcells, 
                                                m_arg, n_arg, canong_arg, options);
    
    cilk_sync;
    
    if (rtnlevel == NAUTY_ABORTED) {
        REDUCER_VIEW(reducer_errstatus) = NAUABORTED;
    } else if (rtnlevel == NAUTY_KILLED) {
        REDUCER_VIEW(reducer_errstatus) = NAUKILLED;
    }
    
    for (int i = 0; i < n_arg; i++) {
        lab[i] = SPLITTER_READ(lab_splitter[i]); 
        orbits_arg[i] = SPLITTER_READ(orbits_splitter[i]);
    }
    
    int numorbits = 0;
    for (int i = 0; i < n_arg; i++) {
        if (orbits_arg[i] == i) numorbits++;
    }
    
    stats_arg->grpsize1 = REDUCER_VIEW(reducer_groupsize).grpsize1;
    stats_arg->grpsize2 = REDUCER_VIEW(reducer_groupsize).grpsize2;
    stats_arg->numorbits = numorbits;
    stats_arg->numgenerators = REDUCER_VIEW(reducer_numgenerators);
    stats_arg->numnodes = REDUCER_VIEW(reducer_numnodes);
    stats_arg->numbadleaves = REDUCER_VIEW(reducer_numbadleaves);
    stats_arg->tctotal = REDUCER_VIEW(reducer_tctotal);
    stats_arg->canupdates = REDUCER_VIEW(reducer_canupdates);
    stats_arg->invapplics = REDUCER_VIEW(reducer_invapplics);
    stats_arg->invsuccesses = REDUCER_VIEW(reducer_invsuccesses);
    stats_arg->invarsuclevel = 
        (REDUCER_VIEW(reducer_invarsuclevel) == NAUTY_INFINITY) ?
            0 : REDUCER_VIEW(reducer_invarsuclevel);
    stats_arg->maxlevel = REDUCER_VIEW(reducer_maxlevel);
    stats_arg->errstatus = REDUCER_VIEW(reducer_errstatus);
    
    if (doschreier) {
    }
}

static void get_perm_from_splitters(int *perm, int n_val) {
    for (int i = 0; i < n_val; i++) {
        perm[i] = SPLITTER_READ(lab_splitter[i]);
    }
}

static void get_ptn_from_splitters(int *ptn, int n_val) {
    for (int i = 0; i < n_val; i++) {
        ptn[i] = SPLITTER_READ(ptn_splitter[i]);
    }
}

static void get_active_from_splitters(set *active, int m_val) {
    for (int i = 0; i < m_val; i++) {
        active[i] = SPLITTER_READ(active_splitter[i]);
    }
}

/**
 * Parallel implementation of firstpathnode
 */
static int firstpathnode_parallel(
    graph *g, int level, int numcells,
    int m_val, int n_val, graph *canong, optionblk *options) {
    
    REDUCER_VIEW(reducer_numnodes)++;
    if (level > REDUCER_VIEW(reducer_maxlevel)) {
        REDUCER_VIEW(reducer_maxlevel) = level;
    }
    
#ifdef NAUTY_IN_MAGMA
    if (main_seen_interrupt) return NAUTY_KILLED;
#else
    if (nauty_kill_request) return NAUTY_KILLED;
#endif
    
    /* Create temporary arrays for doref */
    int lab_temp[MAXN], ptn_temp[MAXN], workperm_temp[MAXN];
    set active_temp[MAXM];
    
    get_perm_from_splitters(lab_temp, n_val);
    get_ptn_from_splitters(ptn_temp, n_val);
    get_active_from_splitters(active_temp, m_val);
    for (int i = 0; i < n_val; i++) {
        workperm_temp[i] = SPLITTER_READ(workperm_splitter[i]);
    }
    
    /* Refine partition */
    int qinvar = 0, refcode = 0;
    doref(g, lab_temp, ptn_temp, level, &numcells, &qinvar, workperm_temp,
        active_temp, &refcode, options->dispatch->refine, options->invarproc,
        options->mininvarlevel, options->maxinvarlevel, options->invararg,
        options->digraph, m_val, n_val);
    
    /* Copy refined partition back to splitters */
    for (int i = 0; i < n_val; i++) {
        SPLITTER_WRITE(lab_splitter[i], lab_temp[i]);
        SPLITTER_WRITE(ptn_splitter[i], ptn_temp[i]);
        SPLITTER_WRITE(workperm_splitter[i], workperm_temp[i]);
    }
    for (int i = 0; i < m_val; i++) {
        SPLITTER_WRITE(active_splitter[i], active_temp[i]);
    }
    
    SPLITTER_WRITE(firstcode_splitter[level], (short)refcode);
    
    if (qinvar > 0) {
        REDUCER_VIEW(reducer_invapplics)++;
        
        if (qinvar == 2) {
            REDUCER_VIEW(reducer_invsuccesses)++;
            
            if (level < REDUCER_VIEW(reducer_invarsuclevel)) {
                REDUCER_VIEW(reducer_invarsuclevel) = level;
            }
        }
    }
    
    int tc = -1, tcellsize = 0;
    set tcell[MAXM];
    EMPTYSET(tcell, m_val);
    
    if (numcells != n_val) {
        maketargetcell(g, lab_temp, ptn_temp, level, tcell, &tcellsize,
                    &tc, options->tc_level, options->digraph, -1, 
                    options->dispatch->targetcell, m_val, n_val);
        
        REDUCER_VIEW(reducer_tctotal) += tcellsize;
    }
    
    SPLITTER_WRITE(firsttc_splitter[level], tc);
    
    if (options->usernodeproc) {
        (*options->usernodeproc)(g, lab_temp, ptn_temp, level, numcells, tc, 
                            (int)refcode, m_val, n_val);
    }
    
    /* Process terminal node (discrete partition) */
    if (numcells == n_val) {
        firstterminal(lab_temp, level);
        
        if (options->userlevelproc) {
            int orbits_temp[MAXN];
            for (int i = 0; i < n_val; i++) {
                orbits_temp[i] = SPLITTER_READ(orbits_splitter[i]);
            }
            
            (*options->userlevelproc)(lab_temp, ptn_temp, level, orbits_temp, 
                                    NULL, 0, 1, 1, n_val, 0, n_val);
        }
        
        if (getcanon && options->usercanonproc) {
            (*options->dispatch->updatecan)(g, canong, lab_temp, 0, m_val, n_val);
            REDUCER_VIEW(reducer_canupdates)++;
            
            if ((*options->usercanonproc)(g, lab_temp, canong, 
                                        REDUCER_VIEW(reducer_canupdates),
                                        (int)SPLITTER_READ(canoncode_splitter[level]), 
                                        m_val, n_val)) {
                return NAUTY_ABORTED;
            }
        }
        
        return level-1;
    }
    
    /* Check for cheap automorphism */
    if (!(*options->dispatch->cheapautom)(ptn_temp, level, options->digraph, n_val)) {
        /* Handle non-cheap automorphism case if needed */
    }
    
    int childcount = 0;
    int tv1 = -1; /* First target vertex */
    
    for (int tv = nextelement(tcell, m_val, -1); tv >= 0; tv = nextelement(tcell, m_val, tv)) {
        if (tv1 == -1) tv1 = tv;
        int tv_orbit = SPLITTER_READ(orbits_splitter[tv]);
        
        /* Only process if this vertex is not equivalent to a previously processed one */
        if (tv_orbit == tv) {
            if (childcount == 0) {
                /* For first child, process in this branch */
                /* We don't need to manually copy lab/ptn because 
                the breakout_splitter will modify our thread's view of the splitters */
                
                /* Save the fixed point */
                set fixedpts = SPLITTER_READ(fixedpts_splitter[0]);
                ADDELEMENT(&fixedpts, tv);
                SPLITTER_WRITE(fixedpts_splitter[0], fixedpts);
                
                breakout_splitter(level+1, tc, tv);
                
                int rtnlevel = firstpathnode_parallel(g, level+1, numcells+1, 
                                                m_val, n_val, canong, options);
                
                fixedpts = SPLITTER_READ(fixedpts_splitter[0]);
                DELELEMENT(&fixedpts, tv);
                SPLITTER_WRITE(fixedpts_splitter[0], fixedpts);
                
                if (rtnlevel < level) {
                    return rtnlevel;
                }
            } else {
                /* For subsequent children, spawn new tasks */
                cilk_spawn process_branch(level, numcells, tc, tv);
            }
            
            childcount++;
        }
        
        if (tv_orbit == tv1) {
            groupsize_t local_group = REDUCER_VIEW(reducer_groupsize);
            multiply_groups(&local_group.grpsize1, &local_group.grpsize2, childcount, 0);
            REDUCER_VIEW(reducer_groupsize) = local_group;
        }
    }
    
    cilk_sync;
    
    if (options->userlevelproc) {
        int orbits_temp[MAXN];
        for (int i = 0; i < n_val; i++) {
            orbits_temp[i] = SPLITTER_READ(orbits_splitter[i]);
        }
        
        (*options->userlevelproc)(lab_temp, ptn_temp, level, orbits_temp, 
                                NULL, tv1, childcount, tcellsize, 
                                numcells, childcount, n_val);
    }
    
    return level-1;
}

/**
 * Parallel implementation of othernode
 */
static int othernode_parallel(
    graph *g, int level, int numcells,
    int m_val, int n_val, graph *canong, optionblk *options) {
    
    REDUCER_VIEW(reducer_numnodes)++;
    if (level > REDUCER_VIEW(reducer_maxlevel)) {
        REDUCER_VIEW(reducer_maxlevel) = level;
    }
    
    /* Check for interruptions */
#ifdef NAUTY_IN_MAGMA
    if (main_seen_interrupt) return NAUTY_KILLED;
#else
    if (nauty_kill_request) return NAUTY_KILLED;
#endif
    
    /* Create temporary arrays for doref */
    int lab_temp[MAXN], ptn_temp[MAXN], workperm_temp[MAXN];
    set active_temp[MAXM];
    
    get_perm_from_splitters(lab_temp, n_val);
    get_ptn_from_splitters(ptn_temp, n_val);
    get_active_from_splitters(active_temp, m_val);
    for (int i = 0; i < n_val; i++) {
        workperm_temp[i] = SPLITTER_READ(workperm_splitter[i]);
    }
    
    /* Refine partition */
    int qinvar = 0, refcode = 0;
    doref(g, lab_temp, ptn_temp, level, &numcells, &qinvar, workperm_temp,
        active_temp, &refcode, options->dispatch->refine, options->invarproc,
        options->mininvarlevel, options->maxinvarlevel, options->invararg,
        options->digraph, m_val, n_val);
    
    /* Copy refined partition back to splitters */
    for (int i = 0; i < n_val; i++) {
        SPLITTER_WRITE(lab_splitter[i], lab_temp[i]);
        SPLITTER_WRITE(ptn_splitter[i], ptn_temp[i]);
        SPLITTER_WRITE(workperm_splitter[i], workperm_temp[i]);
    }
    for (int i = 0; i < m_val; i++) {
        SPLITTER_WRITE(active_splitter[i], active_temp[i]);
    }
    
    if (qinvar > 0) {
        REDUCER_VIEW(reducer_invapplics)++;
        
        if (qinvar == 2) {
            REDUCER_VIEW(reducer_invsuccesses)++;
            
            if (level < REDUCER_VIEW(reducer_invarsuclevel)) {
                REDUCER_VIEW(reducer_invarsuclevel) = level;
            }
        }
    }
    
    /* Compare with canonical code */
    short code = (short)refcode;
    boolean better_labeling = FALSE;
    
    if (getcanon) {
        short canoncode_val = SPLITTER_READ(canoncode_splitter[level]);
        
        if (code < canoncode_val) {
            SPLITTER_WRITE(canoncode_splitter[level], code);
            better_labeling = TRUE;
        }
    }
    
    int tc = -1, tcellsize = 0;
    set tcell[MAXM];
    EMPTYSET(tcell, m_val);
    boolean needtarget = (numcells < n_val);
    
    if (needtarget) {
        maketargetcell(g, lab_temp, ptn_temp, level, tcell, &tcellsize,
                    &tc, options->tc_level, options->digraph, -1, 
                    options->dispatch->targetcell, m_val, n_val);
        
        REDUCER_VIEW(reducer_tctotal) += tcellsize;
    }
    
    if (options->usernodeproc) {
        (*options->usernodeproc)(g, lab_temp, ptn_temp, level, numcells, tc, 
                            (int)code, m_val, n_val);
    }
    
    if (better_labeling && options->usercanonproc) {
        (*options->dispatch->updatecan)(g, canong, lab_temp, 0, m_val, n_val);
        REDUCER_VIEW(reducer_canupdates)++;
        
        if ((*options->usercanonproc)(g, lab_temp, canong, 
                                    REDUCER_VIEW(reducer_canupdates),
                                    (int)code, m_val, n_val)) {
            return NAUTY_ABORTED;
        }
    }
    
    if (numcells == n_val) {
        if (options->userlevelproc) {
            int orbits_temp[MAXN];
            for (int i = 0; i < n_val; i++) {
                orbits_temp[i] = SPLITTER_READ(orbits_splitter[i]);
            }
            
            (*options->userlevelproc)(lab_temp, ptn_temp, level, orbits_temp, 
                                    NULL, 0, 1, 1, n_val, 0, n_val);
        }
        
        return level-1;
    }
    
    if (needtarget) {
        for (int tv = nextelement(tcell, m_val, -1); tv >= 0; tv = nextelement(tcell, m_val, tv)) {
            cilk_spawn process_branch(level, numcells, tc, tv);
        }
        
        cilk_sync;
    }
    
    return level-1;
}

#endif /* ENABLE_PARALLELISM */