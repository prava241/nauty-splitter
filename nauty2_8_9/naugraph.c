/*****************************************************************************
*                                                                            *
*  Graph-specific auxiliary source file for version 2.8 of nauty.            *
*                                                                            *
*   Copyright (1984-2019) Brendan McKay.  All rights reserved.               *
*   Subject to waivers and disclaimers in nauty.h.                           *
*                                                                            *
*   CHANGE HISTORY                                                           *
*       16-Nov-00 : initial creation out of nautil.c                         *
*       22-Apr-01 : added aproto line for Magma                              *
*                   EXTDEFS is no longer required                            *
*                   removed dynamic allocation from refine1()                *
*       21-Nov-01 : use NAUTYREQUIRED in naugraph_check()                    *
*       23-Nov-06 : add targetcell(); make bestcell() local                  *
*       10-Dec-06 : remove BIGNAUTY                                          *
*       10-Nov-09 : remove shortish and permutation types                    *
*       23-May-10 : add densenauty()                                         *
*       15-Jan-12 : add TLS_ATTR attributes                                  *
*       23-Jan-13 : add some parens to make icc happy                        *
*       15-Oct-19 : fix default size of dnwork[] to match densenauty()       *
*        6-Apr-21 : increase work space in densenauty()                      *
*                                                                            *
*****************************************************************************/

#define ONE_WORD_SETS
#include "nauty.h"

    /* macros for hash-codes: */
#define MASH(l,i) ((((l) ^ 065435) + (i)) & 077777)
    /* : expression whose long value depends only on long l and int/long i.
     Anything goes, preferably non-commutative. */

#define CLEANUP(l) ((int)((l) % 077777))
    /* : expression whose value depends on long l and is less than 077777
     when converted to int then short.  Anything goes. */

#if  MAXM==1
#define M 1
#else
#define M m
#endif

/* aproto: header new_nauty_protos.h */

dispatchvec dispatch_graph =
  {isautom,testcanlab,updatecan,refine,refine1,cheapautom,targetcell,
   naugraph_freedyn,naugraph_check,NULL,NULL};

#if !MAXN
DYNALLSTAT(set,workset,workset_sz);
DYNALLSTAT(int,workperm,workperm_sz);
DYNALLSTAT(int,bucket,bucket_sz);
DYNALLSTAT(set,dnwork,dnwork_sz);
#else
static TLS_ATTR set workset[MAXM];   /* used for scratch work */
static TLS_ATTR int workperm[MAXN];
static TLS_ATTR int bucket[MAXN+2];
static TLS_ATTR set dnwork[2*500*MAXM];
#endif

/*****************************************************************************
*                                                                            *
*  isautom(g,perm,digraph,m,n) = TRUE iff perm is an automorphism of g       *
*  (i.e., g^perm = g).  Symmetry is assumed unless digraph = TRUE.           *
*                                                                            *
*****************************************************************************/

boolean
isautom(graph *g, int *perm, boolean digraph, int m, int n)
{
    set *pg;
    int pos;
    set *pgp;
    int posp,i;

    for (pg = g, i = 0; i < n; pg += M, ++i)
    {
        pgp = GRAPHROW(g,perm[i],M);
        pos = (digraph ? -1 : i);

        while ((pos = nextelement(pg,M,pos)) >= 0)
        {
            posp = perm[pos];
            if (!ISELEMENT(pgp,posp)) return FALSE;
        }
    }
    return TRUE;
}

/*****************************************************************************
*                                                                            *
*  testcanlab(g,canong,lab,samerows,m,n) compares g^lab to canong,           *
*  using an ordering which is immaterial since it's only used here.  The     *
*  value returned is -1,0,1 if g^lab <,=,> canong.  *samerows is set to      *
*  the number of rows (0..n) of canong which are the same as those of g^lab. *
*                                                                            *
*  GLOBALS ACCESSED: workset<rw>,permset(),workperm<rw>                      *
*                                                                            *
*****************************************************************************/

int
testcanlab(graph *g, graph *canong, splitter_int_t *lab, int *samerows, int m, int n)
{
    int i,j;
    set *ph;

#if !MAXN
    DYNALLOC1(int,workperm,workperm_sz,n,"testcanlab");
    DYNALLOC1(set,workset,workset_sz,m,"testcanlab");
#endif

    for (i = 0; i < n; ++i) workperm[SPLITTER_READ(lab[i])] = i;

    for (i = 0, ph = canong; i < n; ++i, ph += M)
    {
        permset(GRAPHROW(g,SPLITTER_READ(lab[i]),M),workset,M,workperm);
        for (j = 0; j < M; ++j)
            if (workset[j] < ph[j])
            {
                *samerows = i;
                return -1;
            }
            else if (workset[j] > ph[j])
            {
                *samerows = i;
                return 1;
            }
    }

    *samerows = n;
    return 0;
}

/*****************************************************************************
*                                                                            *
*  updatecan(g,canong,lab,samerows,m,n) sets canong = g^lab, assuming        *
*  the first samerows of canong are ok already.                              *
*                                                                            *
*  GLOBALS ACCESSED: permset(),workperm<rw>                                  *
*                                                                            *
*****************************************************************************/

void
updatecan(graph *g, graph *canong, int *lab, int samerows, int m, int n)
{
    int i;
    set *ph;

#if !MAXN
    DYNALLOC1(int,workperm,workperm_sz,n,"updatecan");
#endif

    for (i = 0; i < n; ++i) workperm[lab[i]] = i;

    for (i = samerows, ph = GRAPHROW(canong,samerows,M);
                                               i < n; ++i, ph += M)
        permset(GRAPHROW(g,lab[i],M),ph,M,workperm);
}

/*****************************************************************************
*                                                                            *
*  refine(g,lab,ptn,level,numcells,count,active,code,m,n) performs a         *
*  refinement operation on the partition at the specified level of the       *
*  partition nest (lab,ptn).  *numcells is assumed to contain the number of  *
*  cells on input, and is updated.  The initial set of active cells (alpha   *
*  in the paper) is specified in the set active.  Precisely, x is in active  *
*  iff the cell starting at index x in lab is active.                        *
*  The resulting partition is equitable if active is correct (see the paper  *
*  and the Guide).                                                           *
*  *code is set to a value which depends on the fine detail of the           *
*  algorithm, but which is independent of the labelling of the graph.        *
*  count is used for work space.                                             *
*                                                                            *
*  GLOBALS ACCESSED:  workset<w>,bit<r>,nextelement(),bucket<w>,workperm<w>  *
*                                                                            *
*****************************************************************************/

void
refine(graph *g, splitter_int_t *lab, splitter_int_t *ptn, int level, splitter_int_t *numcells,
       int *count, set *active, int *code, int m, int n)
{

#if MAXM==1
    refine1(g,lab,ptn,level,numcells,count,active,code,m,n);
}
#else

    int i,c1,c2,labc1;
    setword x;
    set *set1,*set2;
    int split1,split2,cell1,cell2;
    int cnt,bmin,bmax;
    long longcode;
    set *gptr;
    int maxcell,maxpos,hint;

#if !MAXN
    DYNALLOC1(int,workperm,workperm_sz,n,"refine");
    DYNALLOC1(set,workset,workset_sz,m,"refine");
    DYNALLOC1(int,bucket,bucket_sz,n+2,"refine");
#endif

    longcode = SPLITTER_READ(*numcells);
    split1 = -1;
    hint = 0;
    while (SPLITTER_READ(*numcells) < n && ((split1 = hint, ISELEMENT(active,split1))
                         || (split1 = nextelement(active,M,split1)) >= 0
                         || (split1 = nextelement(active,M,-1)) >= 0))
    {
        DELELEMENT(active,split1);
        for (split2 = split1; SPLITTER_READ(ptn[split2]) > level; ++split2) {}
        longcode = MASH(longcode,split1+split2);
        if (split1 == split2)       /* trivial splitting cell */
        {
            gptr = GRAPHROW(g,SPLITTER_READ(lab[split1]),M);
            for (cell1 = 0; cell1 < n; cell1 = cell2 + 1)
            {
                for (cell2 = cell1; SPLITTER_READ(ptn[cell2]) > level; ++cell2) {}
                if (cell1 == cell2) continue;
                c1 = cell1;
                c2 = cell2;
                while (c1 <= c2)
                {
                    labc1 = SPLITTER_READ(lab[c1]);
                    if (ISELEMENT(gptr,labc1))
                        ++c1;
                    else
                    {
                        SPLITTER_WRITE(lab[c1]) = SPLITTER_READ(lab[c2]);
                        SPLITTER_WRITE(lab[c2]) = labc1;
                        --c2;
                    }
                }
                if (c2 >= cell1 && c1 <= cell2)
                {
                    SPLITTER_WRITE(ptn[c2]) = level;
                    longcode = MASH(longcode,c2);
                    SPLITTER_WRITE(*numcells) = SPLITTER_READ(*numcells) + 1;
                    if (ISELEMENT(active,cell1) || c2-cell1 >= cell2-c1)
                    {
                        ADDELEMENT(active,c1);
                        if (c1 == cell2) hint = c1;
                    }
                    else
                    {
                        ADDELEMENT(active,cell1);
                        if (c2 == cell1) hint = cell1;
                    }
                }
            }
        }

        else        /* nontrivial splitting cell */
        {
            EMPTYSET(workset,m);
            for (i = split1; i <= split2; ++i)
                ADDELEMENT(workset,SPLITTER_READ(lab[i]));
            longcode = MASH(longcode,split2-split1+1);

            for (cell1 = 0; cell1 < n; cell1 = cell2 + 1)
            {
                for (cell2 = cell1; SPLITTER_READ(ptn[cell2]) > level; ++cell2) {}
                if (cell1 == cell2) continue;
                i = cell1;
                set1 = workset;
                set2 = GRAPHROW(g,SPLITTER_READ(lab[i]),m);
                cnt = 0;
                for (c1 = m; --c1 >= 0;)
                    if ((x = ((*set1++) & (*set2++))) != 0)
                        cnt += POPCOUNT(x);

                count[i] = bmin = bmax = cnt;
                bucket[cnt] = 1;
                while (++i <= cell2)
                {
                    set1 = workset;
                    set2 = GRAPHROW(g,SPLITTER_READ(lab[i]),m);
                    cnt = 0;
                    for (c1 = m; --c1 >= 0;)
                        if ((x = ((*set1++) & (*set2++))) != 0)
                            cnt += POPCOUNT(x);

                    while (bmin > cnt) bucket[--bmin] = 0;
                    while (bmax < cnt) bucket[++bmax] = 0;
                    ++bucket[cnt];
                    count[i] = cnt;
                }
                if (bmin == bmax)
                {
                    longcode = MASH(longcode,bmin+cell1);
                    continue;
                }
                c1 = cell1;
                maxcell = -1;
                for (i = bmin; i <= bmax; ++i)
                    if (bucket[i])
                    {
                        c2 = c1 + bucket[i];
                        bucket[i] = c1;
                        longcode = MASH(longcode,i+c1);
                        if (c2-c1 > maxcell)
                        {
                            maxcell = c2-c1;
                            maxpos = c1;
                        }
                        if (c1 != cell1)
                        {
                            ADDELEMENT(active,c1);
                            if (c2-c1 == 1) hint = c1;
                            SPLITTER_WRITE(*numcells) = SPLITTER_READ(*numcells) + 1;
                        }
                        if (c2 <= cell2) SPLITTER_WRITE(ptn[c2-1]) = level;
                        c1 = c2;
                    }
                for (i = cell1; i <= cell2; ++i)
                    workperm[bucket[count[i]]++] = SPLITTER_READ(lab[i]);
                for (i = cell1; i <= cell2; ++i) SPLITTER_READ(lab[i]) = workperm[i];
                if (!ISELEMENT(active,cell1))
                {
                    ADDELEMENT(active,cell1);
                    DELELEMENT(active,maxpos);
                }
            }
        }
    }

    longcode = MASH(longcode,SPLITTER_READ(*numcells));
    *code = CLEANUP(longcode);
}
#endif /* else case of MAXM==1 */

/*****************************************************************************
*                                                                            *
*  refine1(g,lab,ptn,level,numcells,count,active,code,m,n) is the same as    *
*  refine(g,lab,ptn,level,numcells,count,active,code,m,n), except that       *
*  m==1 is assumed for greater efficiency.  The results are identical in all *
*  respects.  See refine (above) for the specs.                              *
*                                                                            *
*****************************************************************************/

void
refine1(graph *g, splitter_int_t *lab, splitter_int_t *ptn, int level, splitter_int_t *numcells,
       int *count, set *active, int *code, int m, int n)
{
    int i,c1,c2,labc1;
    setword x;
    int split1,split2,cell1,cell2;
    int cnt,bmin,bmax;
    long longcode;
    set *gptr,workset0;
    int maxcell,maxpos,hint;

#if !MAXN 
    DYNALLOC1(int,workperm,workperm_sz,n,"refine1"); 
    DYNALLOC1(int,bucket,bucket_sz,n+2,"refine1"); 
#endif

    longcode = SPLITTER_READ(*numcells);
    split1 = -1;

    hint = 0;
    while (SPLITTER_READ(*numcells) < n && ((split1 = hint, ISELEMENT1(active,split1))
                         || (split1 = nextelement(active,1,split1)) >= 0
                         || (split1 = nextelement(active,1,-1)) >= 0))
    {
        DELELEMENT1(active,split1);
        for (split2 = split1; SPLITTER_READ(ptn[split2]) > level; ++split2) {}
        longcode = MASH(longcode,split1+split2);
        if (split1 == split2)       /* trivial splitting cell */
        {
            gptr = GRAPHROW(g,SPLITTER_READ(lab[split1]),1);
            for (cell1 = 0; cell1 < n; cell1 = cell2 + 1)
            {
                for (cell2 = cell1; SPLITTER_READ(ptn[cell2]) > level; ++cell2) {}
                if (cell1 == cell2) continue;
                c1 = cell1;
                c2 = cell2;
                while (c1 <= c2)
                {
                    labc1 = SPLITTER_READ(lab[c1]);
                    if (ISELEMENT1(gptr,labc1))
                        ++c1;
                    else
                    {
                        SPLITTER_WRITE(lab[c1]) = SPLITTER_READ(lab[c2]);
                        SPLITTER_WRITE(lab[c2]) = labc1;
                        --c2;
                    }
                }
                if (c2 >= cell1 && c1 <= cell2)
                {
                    SPLITTER_WRITE(ptn[c2]) = level;
                    longcode = MASH(longcode,c2);
                    SPLITTER_WRITE(*numcells) = SPLITTER_READ(*numcells) + 1;
                    if (ISELEMENT1(active,cell1) || c2-cell1 >= cell2-c1)
                    {
                        ADDELEMENT1(active,c1);
                        if (c1 == cell2) hint = c1;
                    }
                    else
                    {
                        ADDELEMENT1(active,cell1);
                        if (c2 == cell1) hint = cell1;
                    }
                }
            }
        }

        else        /* nontrivial splitting cell */
        {
            workset0 = 0;
            for (i = split1; i <= split2; ++i)
                ADDELEMENT1(&workset0,SPLITTER_READ(lab[i]));
            longcode = MASH(longcode,split2-split1+1);

            for (cell1 = 0; cell1 < n; cell1 = cell2 + 1)
            {
                for (cell2 = cell1; SPLITTER_READ(ptn[cell2]) > level; ++cell2) {}
                if (cell1 == cell2) continue;
                i = cell1;
                if ((x = workset0 & g[SPLITTER_READ(lab[i])]) != 0)
                    cnt = POPCOUNT(x);
                else
                    cnt = 0;
                count[i] = bmin = bmax = cnt;
                bucket[cnt] = 1;
                while (++i <= cell2)
                {
                    if ((x = workset0 & g[SPLITTER_READ(lab[i])]) != 0)
                        cnt = POPCOUNT(x);
                    else
                        cnt = 0;
                    while (bmin > cnt) bucket[--bmin] = 0;
                    while (bmax < cnt) bucket[++bmax] = 0;
                    ++bucket[cnt];
                    count[i] = cnt;
                }
                if (bmin == bmax)
                {
                    longcode = MASH(longcode,bmin+cell1);
                    continue;
                }
                c1 = cell1;
                maxcell = -1;
                for (i = bmin; i <= bmax; ++i)
                    if (bucket[i])
                    {
                        c2 = c1 + bucket[i];
                        bucket[i] = c1;
                        longcode = MASH(longcode,i+c1);
                        if (c2-c1 > maxcell)
                        {
                            maxcell = c2-c1;
                            maxpos = c1;
                        }
                        if (c1 != cell1)
                        {
                            ADDELEMENT1(active,c1);
                            if (c2-c1 == 1) hint = c1;
                            SPLITTER_WRITE(*numcells) = SPLITTER_READ(*numcells) + 1;
                        }
                        if (c2 <= cell2) SPLITTER_WRITE(ptn[c2-1]) = level;
                        c1 = c2;
                    }
                for (i = cell1; i <= cell2; ++i)
                    workperm[bucket[count[i]]++] = SPLITTER_READ(lab[i]);
                for (i = cell1; i <= cell2; ++i) SPLITTER_READ(lab[i]) = workperm[i];
                if (!ISELEMENT1(active,cell1))
                {
                    ADDELEMENT1(active,cell1);
                    DELELEMENT1(active,maxpos);
                }
            }
        }
    }

    longcode = MASH(longcode,SPLITTER_READ(*numcells));
    *code = CLEANUP(longcode);
}

void
refine1_splitter(graph *g, splitter_int_t *lab, splitter_int_t *ptn, int level, int *numcells,
       int *count, splitter_set_t *active, int *code, int m, int n)
{
    int i,c1,c2,labc1;
    setword x;
    int split1,split2,cell1,cell2;
    int cnt,bmin,bmax;
    long longcode;
    set *gptr,workset0;
    int maxcell,maxpos,hint;

#if !MAXN 
    DYNALLOC1(int,workperm,workperm_sz,n,"refine1"); 
    DYNALLOC1(int,bucket,bucket_sz,n+2,"refine1"); 
#endif

    longcode = *numcells;
    split1 = -1;

    hint = 0;
    while (*numcells < n && ((split1 = hint, ISELEMENT1(active,split1))
                         || (split1 = nextelement(active,1,split1)) >= 0
                         || (split1 = nextelement(active,1,-1)) >= 0))
                         //nextelement should have a splitter version
    {
        DELELEMENT1(active,split1);
        for (split2 = split1; SPLITTER_READ(ptn[split2]) > level; ++split2) {}
        longcode = MASH(longcode,split1+split2);
        if (split1 == split2)       /* trivial splitting cell */
        {
            gptr = GRAPHROW(g,SPLITTER_READ(lab[split1]),1);
            for (cell1 = 0; cell1 < n; cell1 = cell2 + 1)
            {
                for (cell2 = cell1; SPLITTER_READ(ptn[cell2]) > level; ++cell2) {}
                if (cell1 == cell2) continue;
                c1 = cell1;
                c2 = cell2;
                while (c1 <= c2)
                {
                    labc1 = SPLITTER_READ(lab[c1]);
                    if (ISELEMENT1(gptr,labc1))
                        ++c1;
                    else
                    {
                        SPLITTER_WRITE(lab[c1]) = SPLITTER_READ(lab[c2]);
                        SPLITTER_WRITE(lab[c2]) = labc1;
                        --c2;
                    }
                }
                if (c2 >= cell1 && c1 <= cell2)
                {
                    SPLITTER_WRITE(ptn[c2]) = level;
                    longcode = MASH(longcode,c2);
                    ++*numcells;
                    if (ISELEMENT1(active,cell1) || c2-cell1 >= cell2-c1)
                    {
                        ADDELEMENT1(active,c1);
                        if (c1 == cell2) hint = c1;
                    }
                    else
                    {
                        ADDELEMENT1(active,cell1);
                        if (c2 == cell1) hint = cell1;
                    }
                }
            }
        }

        else        /* nontrivial splitting cell */
        {
            workset0 = 0;
            for (i = split1; i <= split2; ++i)
                ADDELEMENT1(&workset0,lab[i]);  // need splitter add element
            longcode = MASH(longcode,split2-split1+1);

            for (cell1 = 0; cell1 < n; cell1 = cell2 + 1)
            {
                for (cell2 = cell1; SPLITTER_READ(ptn[cell2]) > level; ++cell2) {}
                if (cell1 == cell2) continue;
                i = cell1;
                if ((x = workset0 & g[SPLITTER_READ(lab[i])]) != 0)
                    cnt = POPCOUNT(x);
                else
                    cnt = 0;
                count[i] = bmin = bmax = cnt;
                bucket[cnt] = 1;
                while (++i <= cell2)
                {
                    if ((x = workset0 & g[SPLITTER_READ(lab[i])]) != 0)
                        cnt = POPCOUNT(x);
                    else
                        cnt = 0;
                    while (bmin > cnt) bucket[--bmin] = 0;
                    while (bmax < cnt) bucket[++bmax] = 0;
                    ++bucket[cnt];
                    count[i] = cnt;
                }
                if (bmin == bmax)
                {
                    longcode = MASH(longcode,bmin+cell1);
                    continue;
                }
                c1 = cell1;
                maxcell = -1;
                for (i = bmin; i <= bmax; ++i)
                    if (bucket[i])
                    {
                        c2 = c1 + bucket[i];
                        bucket[i] = c1;
                        longcode = MASH(longcode,i+c1);
                        if (c2-c1 > maxcell)
                        {
                            maxcell = c2-c1;
                            maxpos = c1;
                        }
                        if (c1 != cell1)
                        {
                            ADDELEMENT1(active,c1);
                            if (c2-c1 == 1) hint = c1;
                            ++*numcells;
                        }
                        if (c2 <= cell2) SPLITTER_WRITE(ptn[c2-1]) = level;
                        c1 = c2;
                    }
                for (i = cell1; i <= cell2; ++i)
                    workperm[bucket[count[i]]++] = SPLITTER_READ(lab[i]);
                for (i = cell1; i <= cell2; ++i) SPLITTER_WRITE(lab[i]) = workperm[i];
                if (!ISELEMENT1(active,cell1))
                {
                    ADDELEMENT1(active,cell1);
                    DELELEMENT1(active,maxpos);
                }
            }
        }
    }

    longcode = MASH(longcode,*numcells);
    *code = CLEANUP(longcode);
}

/*****************************************************************************
*                                                                            *
*  cheapautom(ptn,level,digraph,n) returns TRUE if the partition at the      *
*  specified level in the partition nest (lab,ptn) {lab is not needed here}  *
*  satisfies a simple sufficient condition for its cells to be the orbits of *
*  some subgroup of the automorphism group.  Otherwise it returns FALSE.     *
*  It always returns FALSE if digraph!=FALSE.                                *
*                                                                            *
*  nauty assumes that this function will always return TRUE for any          *
*  partition finer than one for which it returns TRUE.                       *
*                                                                            *
*****************************************************************************/

boolean
cheapautom(splitter_int_t *ptn, int level, boolean digraph, int n)
{
    int i,k,nnt;

    if (digraph) return FALSE;

    k = n;
    nnt = 0;
    for (i = 0; i < n; ++i)
    {
        --k;
        if (SPLITTER_READ(ptn[i]) > level)
        {
            ++nnt;
            while (SPLITTER_READ(ptn[++i]) > level) {}
        }
    }

    return (k <= nnt + 1 || k <= 4);
}

/*****************************************************************************
*                                                                            *
*  bestcell(g,lab,ptn,level,tc_level,m,n) returns the index in lab of the    *
*  start of the "best non-singleton cell" for fixing.  If there is no        *
*  non-singleton cell it returns n.                                          *
*  This implementation finds the first cell which is non-trivially joined    *
*  to the greatest number of other cells.                                    *
*                                                                            *
*  GLOBALS ACCESSED: bit<r>,workperm<rw>,workset<rw>,bucket<rw>              *
*                                                                            *
*****************************************************************************/

static int
bestcell(const graph *g, const int *lab, const int *ptn, int level,
         int tc_level, int m, int n)
{
    int i;
    set *gp;
    setword setword1,setword2;
    int v1,v2,nnt;

#if !MAXN 
    DYNALLOC1(int,workperm,workperm_sz,n,"bestcell"); 
    DYNALLOC1(set,workset,workset_sz,m,"bestcell"); 
    DYNALLOC1(int,bucket,bucket_sz,n+2,"bestcell"); 
#endif

   /* find non-singleton cells: put starts in workperm[0..nnt-1] */

    i = nnt = 0;

    while (i < n)
    {
        if (ptn[i] > level)
        {
            workperm[nnt++] = i;
            while (ptn[i] > level) ++i;
        }
        ++i;
    }

    if (nnt == 0) return n;

    /* set bucket[i] to # non-trivial neighbours of n.s. cell i */

    for (i = nnt; --i >= 0;) bucket[i] = 0;

    for (v2 = 1; v2 < nnt; ++v2)
    {
        EMPTYSET(workset,m);
        i = workperm[v2] - 1;
        do
        {
            ++i;
            ADDELEMENT(workset,lab[i]);
        }
        while (ptn[i] > level);
        for (v1 = 0; v1 < v2; ++v1)
        {
            gp = GRAPHROW(g,lab[workperm[v1]],m);
#if  MAXM==1
            setword1 = *workset & *gp;
            setword2 = *workset & ~*gp;
#else
            setword1 = setword2 = 0;
            for (i = m; --i >= 0;)
            {
                setword1 |= workset[i] & gp[i];
                setword2 |= workset[i] & ~gp[i];
            }
#endif
            if (setword1 != 0 && setword2 != 0)
            {
                ++bucket[v1];
                ++bucket[v2];
            }
        }
    }

    /* find first greatest bucket value */

    v1 = 0;
    v2 = bucket[0];
    for (i = 1; i < nnt; ++i)
        if (bucket[i] > v2)
        {
            v1 = i;
            v2 = bucket[i];
        }

    return (int)workperm[v1];
}

static int
bestcell_splitter(const graph *g, splitter_int_t *lab, splitter_int_t *ptn, int level,
         int tc_level, int m, int n)
{
    int i;
    set *gp;
    setword setword1,setword2;
    int v1,v2,nnt;

#if !MAXN 
    DYNALLOC1(int,workperm,workperm_sz,n,"bestcell"); 
    DYNALLOC1(set,workset,workset_sz,m,"bestcell"); 
    DYNALLOC1(int,bucket,bucket_sz,n+2,"bestcell"); 
#endif

   /* find non-singleton cells: put starts in workperm[0..nnt-1] */

    i = nnt = 0;

    while (i < n)
    {
        if (SPLITTER_READ(ptn[i]) > level)
        {
            workperm[nnt++] = i;
            while (SPLITTER_READ(ptn[i]) > level) ++i;
        }
        ++i;
    }

    if (nnt == 0) return n;

    /* set bucket[i] to # non-trivial neighbours of n.s. cell i */

    for (i = nnt; --i >= 0;) bucket[i] = 0;

    for (v2 = 1; v2 < nnt; ++v2)
    {
        EMPTYSET(workset,m);
        i = workperm[v2] - 1;
        do
        {
            ++i;
            ADDELEMENT(workset,SPLITTER_READ(lab[i]));
        }
        while (SPLITTER_READ(ptn[i])> level);
        for (v1 = 0; v1 < v2; ++v1)
        {
            gp = GRAPHROW(g,lab[workperm[v1]],m);
#if  MAXM==1
            setword1 = *workset & *gp;
            setword2 = *workset & ~*gp;
#else
            setword1 = setword2 = 0;
            for (i = m; --i >= 0;)
            {
                setword1 |= workset[i] & gp[i];
                setword2 |= workset[i] & ~gp[i];
            }
#endif
            if (setword1 != 0 && setword2 != 0)
            {
                ++bucket[v1];
                ++bucket[v2];
            }
        }
    }

    /* find first greatest bucket value */

    v1 = 0;
    v2 = bucket[0];
    for (i = 1; i < nnt; ++i)
        if (bucket[i] > v2)
        {
            v1 = i;
            v2 = bucket[i];
        }

    return (int)workperm[v1];
}

/*****************************************************************************
*                                                                            *
*  targetcell(g,lab,ptn,level,tc_level,digraph,hint,m,n) returns the index   *
*  in lab of the next cell to split.                                         *
*  hint is a suggestion for the answer, which is obeyed if it is valid.      *
*  Otherwise we use bestcell() up to tc_level and the first non-trivial      *
*  cell after that.                                                          *
*                                                                            *
*****************************************************************************/

int
targetcell(graph *g, int *lab, int *ptn, int level,
       int tc_level, boolean digraph, int hint, int m, int n)
{
    int i;

    if (hint >= 0 && ptn[hint] > level &&
                     (hint == 0 || ptn[hint-1] <= level))
        return hint;
    else if (level <= tc_level)
        return bestcell(g,lab,ptn,level,tc_level,m,n);
    else
    {
        for (i = 0; i < n && ptn[i] <= level; ++i) {}
        return (i == n ? 0 : i);
    }
}

int
targetcell_splitter(graph *g, splitter_int_t *lab, splitter_int_t *ptn, int level,
       int tc_level, boolean digraph, int hint, int m, int n)
{
    int i;

    if (hint >= 0 && SPLITTER_READ(ptn[hint]) > level &&
                     (hint == 0 || SPLITTER_READ(ptn[hint]) <= level))
        return hint;
    else if (level <= tc_level)
        return bestcell_splitter(g,lab,ptn,level,tc_level,m,n);
    else
    {
        for (i = 0; i < n && SPLITTER_READ(ptn[i]) <= level; ++i) {}
        return (i == n ? 0 : i);
    }
}

/*****************************************************************************
*                                                                            *
*  densenauty(g,lab,ptn,orbits,&options,&stats,m,n,h)                        *
*  is a slightly simplified interface to nauty().  It allocates enough       *
*  workspace for 500 automorphisms and checks that the densegraph dispatch   *
*  vector is in use.                                                         *
*                                                                            *
*****************************************************************************/

void
densenauty(graph *g, int *lab, int *ptn, int *orbits,
           optionblk *options, statsblk *stats, int m, int n, graph *h)
{
    if (options->dispatch != &dispatch_graph)
    {
        fprintf(ERRFILE,"Error: densenauty() needs standard options block\n");
        exit(1);
    }

#if !MAXN
    /* Don't increase 2*500*m in the next line unless you also increase
       the default declaration of dnwork[] earlier. */
    DYNALLOC1(set,dnwork,dnwork_sz,2*500*m,"densenauty malloc");
#endif

    nauty(g,lab,ptn,NULL,orbits,options,stats,dnwork,2*500*m,m,n,h);
}

/*****************************************************************************
*                                                                            *
*  naugraph_check() checks that this file is compiled compatibly with the    *
*  given parameters.   If not, call exit(1).                                 *
*                                                                            *
*****************************************************************************/

void
naugraph_check(int wordsize, int m, int n, int version)
{
    if (wordsize != WORDSIZE)
    {
        fprintf(ERRFILE,"Error: WORDSIZE mismatch in naugraph.c\n");
        exit(1);
    }

#if MAXN
    if (m > MAXM)
    {
        fprintf(ERRFILE,"Error: MAXM inadequate in naugraph.c\n");
        exit(1);
    }

    if (n > MAXN)
    {
        fprintf(ERRFILE,"Error: MAXN inadequate in naugraph.c\n");
        exit(1);
    }
#endif

    if (version < NAUTYREQUIRED)
    {
        fprintf(ERRFILE,"Error: naugraph.c version mismatch\n");
        exit(1);
    }
}

/*****************************************************************************
*                                                                            *
*  naugraph_freedyn() - free the dynamic memory in this module               *
*                                                                            *
*****************************************************************************/

void
naugraph_freedyn(void)
{
#if !MAXN
    DYNFREE(workset,workset_sz);
    DYNFREE(workperm,workperm_sz);
    DYNFREE(bucket,bucket_sz);
    DYNFREE(dnwork,dnwork_sz);
#endif
}
