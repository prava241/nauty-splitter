/**
 * nauty_parallel.h - Cilk parallel interface for nauty
 *
 * This header provides parallelism for the nauty graph
 * canonicalization algorithm using Cilk splitters.
 */

 #ifndef NAUTY_PARALLEL_H
 #define NAUTY_PARALLEL_H
 
 #include "nauty.h"
 
 // Only compile parallel code if ENABLE_PARALLELISM is defined
 #ifdef ENABLE_PARALLELISM
 
 #include <cilk/cilk.h>
 #include <cilk/cilk_api.h>
 #include <cilk/common.h>
 #include <cilk/reducer.h>
 #include <cilk/reducer_opadd.h>
 #include <cilk/reducer_opand.h>
 #include <cilk/reducer_opor.h>
 #include <cilk/reducer_min_max.h>
 
 // Splitter type definitions for nauty data structures
 typedef CILK_C_DECLARE_SPLITTER(int) splitter_int_t;
 typedef CILK_C_DECLARE_SPLITTER(double) splitter_double_t;
 typedef CILK_C_DECLARE_SPLITTER(set) splitter_set_t;
 typedef CILK_C_DECLARE_SPLITTER(short) splitter_short_t;
 
 // Parallel-enabled versions of key functions
 extern void nauty_parallel(graph *g_arg, int *lab, int *ptn, set *active_arg,
     int *orbits_arg, optionblk *options, statsblk *stats_arg,
     set *ws_arg, int worksize, int m_arg, int n_arg, graph *canong_arg);
 
 // Initialization and cleanup
 extern void nauty_parallel_init(int n, int m);
 extern void nauty_parallel_cleanup(void);
 
 #define NAUTY_PARALLEL_AVAILABLE 1
 
 #else

 #define NAUTY_PARALLEL_AVAILABLE 0
 
 static inline void
 nauty_parallel(graph *g_arg, int *lab, int *ptn, set *active_arg,
     int *orbits_arg, optionblk *options, statsblk *stats_arg,
     set *ws_arg, int worksize, int m_arg, int n_arg, graph *canong_arg)
 {
     nauty(g_arg, lab, ptn, active_arg, orbits_arg, options, stats_arg,
           ws_arg, worksize, m_arg, n_arg, canong_arg);
 }
 
 #define nauty_parallel_init(n, m) do {} while(0)
 #define nauty_parallel_cleanup() do {} while(0)
 
 #endif /* ENABLE_PARALLELISM */
 
 #endif /* NAUTY_PARALLEL_H */