/* TODO:
 *        add complements for ordinary graphs
 *        rigid + unique point = rigid
 */

/* geng.c  version 3.7; B D McKay, April 2024. */

#define USAGE \
"geng [-cCmtfkbd#D#] [-kTSPF] [-uygsnh] [-lvq] \n\
              [-x#X#] n [mine[:maxe]] [res/mod] [file]"

#define HELPTEXT \
" Generate all graphs of a specified class.\n\
\n\
      n    : the number of vertices\n\
 mine:maxe : a range for the number of edges\n\
              #:0 means '# or more' except in the case 0:0\n\
   res/mod : only generate subset res out of subsets 0..mod-1\n\
\n\
     -c    : only write connected graphs\n\
     -C    : only write biconnected graphs\n\
     -t    : only generate triangle-free graphs\n\
     -f    : only generate 4-cycle-free graphs\n\
     -p    : only generate 5-cycle-free graphs\n\
     -k    : only generate K4-free graphs\n\
     -T    : only generate chordal graphs\n\
     -S    : only generate split graphs\n\
     -P    : only generate perfect graphs\n\
     -F    : only generate claw-free graphs\n\
     -b    : only generate bipartite graphs\n\
                (-t, -f and -b can be used in any combination)\n\
		The combination -bT gives bipartite graphs with no\n\
                  induced cycles longer than 4-cycles.\n\
     -m    : save memory at the expense of time (only makes a\n\
                difference in the absence of -b, -t, -f and n <= 28).\n\
     -d#   : a lower bound for the minimum degree\n\
     -D#   : an upper bound for the maximum degree\n\
     -v    : display counts by number of edges\n\
     -l    : canonically label output graphs\n\
\n\
     -u    : do not output any graphs, just generate and count them\n\
     -g    : use graph6 output (default)\n\
     -s    : use sparse6 output\n\
     -h    : for graph6 or sparse6 format, write a header too\n\
\n\
     -q    : suppress auxiliary output (except from -v)\n\
\n\
  See program text for much more information.\n"


/*  Parameters:

             n    = the number of vertices (1..MAXN)
                        Note that MAXN is limited to min(WORDSIZE,64)
             mine = the minimum number of edges (no bounds if missing)
             maxe = the maximum number of edges (same as mine if missing)
                    0 means "infinity" except in the case "0-0"
             mod, res = a way to restrict the output to a subset.
                        All the graphs in G(n,mine..maxe) are divided into
                        disjoint classes C(0,mod),C(1,mod),...,C(mod-1,mod),
                        of very approximately equal size.
                        Only the class C(res,mod) is written.

                        If the -x or -X switch is used, they must have the 
                        same value for different values of res; otherwise 
                        the partitioning may not be valid.  In this case
                        (-x,-X with constant value), the usual relationships 
                        between modulo classes are obeyed; for example 
                        C(3,4) = C(3,8) union C(7,8).  This is not true
                        if 3/8 and 7/8 are done with -x or -X values
                        different from those used for 3/4.

             file = a name for the output file (stdout if missing or "-")

             All switches can be concatenated or separate.  However, the
             value of -d must be attached to the "d", and similarly for "x".

             -c    : only write connected graphs
             -C    : only write biconnected graphs
             -t    : only generate triangle-free graphs
             -f    : only generate 4-cycle-free graphs
             -b    : only generate bipartite graphs
                        (-t, -f and -b can be used in any combination)
             -m    : save memory at expense of time (only makes a
                        difference in the absence of -b, -t, -f and n <= 30).
             -D<int> : specify an upper bound for the maximum degree.
                     The value of the upper bound must be adjacent to
                     the "D".  Example: -D6
             -d<int> : specify a lower bound for the minimum degree.
                     The value of the upper bound must be adjacent to
                     the "d".  Example: -d6
             -v    : display counts by number of edges
             -l    : canonically label output graphs

             -u    : do not output any graphs, just generate and count them
             -g    : use graph6 output (default)
             -s    : use sparse6 output
             -n    : use nauty format instead of graph6 format for output
             -y    : use the obsolete y-format for output
             -h    : for graph6 or sparse6 format, write a header too

             -q    : suppress auxiliary output (except from -v)

             -x<int> : specify a parameter that determines how evenly
                     the res/mod facility splits the graphs into subsets.
                     High values mean more even splitting at slight cost
                     to the total time.  The default is 20*mod, and the
                     the legal minimum is 3*mod.  More information is given 
                     under "res/mod" above.
             -X<lev> : move the initial splitting level higher by <lev>,
                     in order to force more even splitting at the cost
                     of speed.  Default is -X0.  More information is given
                     under "res/mod" above.

Output formats.

  The output format is determined by the mutually exclusive switches
  -u, -n, -y, -g and -s.  The default is -g.

  -u suppresses output of graphs completely.

  -s and -g specify sparse6 and graph6 format, defined elsewhere.
  In this case a header is also written if -h is present.

  If -y is present, graphs will be written in y-format.
  y-format is obsolete and only provided for backwards compatibility.

    Each graph occupies one line with a terminating newline.
    Except for the newline, each byte has the format  01xxxxxx, where
    each "x" represents one bit of data.
    First byte:  xxxxxx is the number of vertices n
    Other ceiling(n(n-1)/12) bytes:  These contain the upper triangle of
    the adjacency matrix in column major order.  That is, the entries
    appear in the order (0,1),(0,2),(1,2),(0,3),(1,3),(2,3),(0,4),... .
    The bits are used in left to right order within each byte.
    Any unused bits on the end are set to zero.

  If -n is present, any output graphs are written in nauty format.

    For a graph of n vertices, the output consists of one int giving
    the number of vertices, and n setwords containing the adjacency
    matrix.  Note that this is system dependent (i.e. don't use it).
    It will not work properly if the output is to stdout and your
    system distinguishes binary and text files.

OUTPROC feature.

   By defining the C preprocessor variable OUTPROC at compile time
   (for Unix the syntax is -DOUTPROC=procname on the cc command),
   geng can be made to call a procedure of your manufacture with
   each output graph instead of writing anything. Your procedure
   needs to have type void and the argument list (FILE *f, graph
   *g, int n). f is a stream open for writing, g is the graph in
   nauty format, and n is the number of vertices. Your procedure
   can be in a separate file so long as it is linked with geng. The
   global variables sparse6, graph6, quiet, nooutput, nautyformat,
   and canonise (all type boolean) can be used to test
   for the presence of the flags -s, -g, -q, -u, -n, and -l,
   respectively. If -l is present, the group size and similar
   details can be found in the global variable nauty_stats.

PRUNE feature.

   By defining the C preprocessor variable PRUNE at compile time, geng
   can be made to call
        int PRUNE(graph *g,int n,int maxn) 
   for each intermediate (and final) graph, and reject it if 
   the value returned is nonzero.  The arguments are:

     g      = the graph in nauty format (m=1)
     n      = the number of vertices in g
     maxn   = the number of vertices for output 
              (the value you gave on the command line to geng)

   geng constructs the graph starting with vertex 0, then adding
   vertices 1,2,3,... in that order.  Each graph in the sequence is
   an induced subgraph of all later graphs in the sequence.

   A call is made for all orders from 1 to maxn.  In testing for
   a uniform property (such as a forbidden subgraph or forbidden
   induced subgraph) it might save time to notice that a call to
   PRUNE for n implies that the call for n-1 already passed. 

   For very fast tests, it might be worthwhile using PREPRUNE as
   well or instead. It has the same meaning but is applied earlier
   and more often.

   If -c or -C is given, the connectivity test is done before
   PRUNE but not necessarily before PREPRUNE.

   Some parameters are available in global variables:
   geng_mindeg, geng_maxdeg, geng_mine, geng_maxe, geng_connec;
  
SUMMARY

   If the C preprocessor variable SUMMARY is defined at compile time, the
   procedure SUMMARY(nauty_counter nout, double cpu) is called just before
   the program exits.  The purpose is to allow reporting of statistics
   collected by PRUNE or OUTPROC.  The values nout and cpu are the output
   count and cpu time reported on the >Z line.
   Output should be written to stderr.

INSTRUMENT feature.

   If the C preprocessor variable INSTRUMENT is defined at compile time,
   extra code is inserted to collect statistics during execution, and
   more information is written to stderr at termination.

CALLING FROM A PROGRAM

   It is possible to call geng from another program instead of using it
   as a stand-alone program.  The main requirement is to change the name
   of the main program to be other than "main".  This is done by defining
   the preprocessor variable GENG_MAIN.  You might also like to define
   OUTPROC to be the name of a procedure to receive the graphs. To call
   the program you need to define an argument list argv[] consistent with
   the usual one; don't forget that argv[0] is the command name and not
   the first argument.  The value of argc is the number of strings in
   argv[]; that is, one more than the number of arguments.  See the
   sample program callgeng.c.

   You can also call geng from multiple threads at once, see the sample
   program callgeng2.c.

**************************************************************************

Counts and sample performance statistics.

    Here we give some graph counts and approximate execution times
    on a Linux computer with Intel Core i7-4790 nominally 3.6GHz,
    compiled with gcc 6.2.0.
    Times are with the -u option (generate but don't write); add
    0.2-0.3 microseconds per graph for output to a file.


      General Graphs                     C3-free Graphs (-t)

      1              1                    1              1
      2              2                    2              2
      3              4                    3              3
      4             11                    4              7
      5             34                    5             14
      6            156                    6             38
      7           1044                    7            107
      8          12346                    8            410
      9         274668   0.08 sec         9           1897
     10       12005168   2.7 sec         10          12172  
     11     1018997864   207 sec         11         105071   0.09 sec
     12   165091172592   9 hr            12        1262180   0.8 sec
     13 50502031367952   108 days        13       20797002   11 sec
    These can be done in about half      14      467871369   220 sec
    the time by setting the edge limit   15    14232552452   1.7 hr
    half way then adding complements.    16   581460254001   65 hr
                                         17 31720840164950   145 days


     C4-free Graphs  (-f)              (C3,C4)-free Graphs (-tf)

      1             1                      1             1
      2             2                      2             2 
      3             4                      3             3 
      4             8                      4             6
      5            18                      5            11
      6            44                      6            23
      7           117                      7            48 
      8           351                      8           114 
      9          1230                      9           293
     10          5069                     10           869
     11         25181   0.04 sec          11          2963 
     12        152045   0.17 sec          12         12066   0.03 sec
     13       1116403   1.0 sec           13         58933   0.10 sec
     14       9899865   7.5 sec           14        347498   0.5 sec
     15     104980369   71 sec            15       2455693   2.7 sec
     16    1318017549   14 min            16      20592932   19 sec
     17   19427531763   3.4 hr            17     202724920   170 sec
     18  333964672216   56 hr             18    2322206466   32 min
     19 6660282066936   45 days           19   30743624324   7 hr
                                          20  468026657815   4 days
                                          21 8161170076257   106 days


     Bipartite Graphs (-b)            C4-free Bipartite Graphs (-bf)

      1              1                      1             1
      2              2                      2             2
      3              3                      3             3
      4              7                      4             6
      5             13                      5            10
      6             35                      6            21
      7             88                      7            39
      8            303                      8            86
      9           1119                      9           182
     10           5479                     10           440
     11          32303   0.03 sec          11          1074
     12         251135   2.3 sec           12          2941
     13        2527712   1.7 sec           13          8424   0.04 sec
     14       33985853   19 sec            14         26720   0.11 sec
     15      611846940   4.9 min           15         90883   0.33 sec
     16    14864650924   1.8 hr            16        340253   1.1 sec
     17   488222721992   2.4 days          17       1384567   3.7 sec
     18 21712049275198   105 days          18       6186907   14 sec
                                           19      30219769   59 sec
                                           20     161763233   280 sec
                                           21     946742190   24 min
                                           22    6054606722   2.5 hr
                                           23   42229136988   17 hr
                                           24  320741332093   125 hr
                                           25 2648348712904   58 days

If you know any more of these counts, please tell me.

**************************************************************************

Hints:

To make all the graphs of order n, without restriction on type,
it is fastest to make them up to binomial(n,2)/2 edges and append
the complement of those with strictly less than binomial(n,2)/2 edges.

If it is necessary to split the computation into pieces, it is more
efficient to use the res/mod feature than to split by numbers of edges.

**************************************************************************

    Author:   B. D. McKay, Sep 1991 and many later dates.
              Copyright  B. McKay (1991-2018).  All rights reserved.
              This software is subject to the conditions and waivers
              detailed in the file nauty.h.

    Changes:  Nov 18, 1991 : added -d switch
                             fixed operation for n=16
              Nov 26, 1991 : added OUTPROC feature
              Nov 29, 1991 : -c implies mine >= n-1
              Jan  8, 1992 : make writeny() not static
              Jan 10, 1992 : added -n switch
              Feb  9, 1992 : fixed case of n=1
              Feb 16, 1992 : changed mine,maxe,maxdeg testing
              Feb 19, 1992 : added -b, -t and -u options
                             documented OUTPROC and added external
                                 declaration for it.
              Feb 20, 1992 : added -v option
              Feb 22, 1992 : added INSTRUMENT compile-time option
              Feb 23, 1992 : added xbnds() for more effective pruning
              Feb 24, 1992 : added -l option
              Feb 25, 1992 : changed writenauty() to use fwrite()
              Mar 11, 1992 : completely revised many parts, incl
                             new refinement procedure for fast rejection,
                             distance invariant for regular graphs
              May 19, 1992 : modified userautomproc slightly.  xorb[]
                             is no longer idempotent but it doesn't matter.
                             Speed-up of 2-5% achieved.
              June 5, 1993 : removed ";" after "CPUDEFS" to avoid illegal
                             empty declaration.
              Nov 24, 1994 : tested for 0 <= res < mod

              Apr 13, 1996 : Major overhaul.  Renamed "geng".
                             Changed argument syntax.
                             Removed 16-vertex limit.
                             Added -s, -m, -x.  Allowed combinations.
                             Replaced code for non-general graphs.
                             Very many small changes.
              Jul 12, 1996 : Changed semantics of -x and res/mod.
                             Changed >A line and added fflush()/
                             All switches can be concatenated or not.
              Aug 16, 1996 : Added -X switch and PRUNE() feature.
                             Fixed case of argument 0-0.
              Sep 22, 1996 : Improved 1-2% by tweaking refinex().
              Jan 21, 1997 : Renamed to geng.  
                             Changed -s to -f, and added -sghq.
              Sep  7, 1997 : Fixed WORDSIZE=16 problems.
              Sep 22, 1997 : Use "wb" open for nautyformat.
              Jan 26, 1998 : Added SUMMARY feature.
              Mar  4, 1998 : Added -C.
              Mar 12, 1998 : Moved stats to nauty_stats.
              Jan  1, 2000 : Changed -d to -D and added -d.
              Feb 24, 2000 : Raised limit to 32 vertices.
              Mar  3, 2000 : Made some counts into unsigned long.
                             (Includes first arg to SUMMARY.)
              Mar 12, 2000 : Used bigint for counts that may exceed 2^32.
                             Now all counts from very long runs are ok.
              Oct 12, 2000 : Changed maxef[32] to 92 after confirmation
                             from Yang Yuansheng.  The old value of 93 was
                             valid but 92 is slightly more efficient.
              Nov 16, 2000 : Used fuction prototypes.
              Jul 31, 2001 : Added PREPRUNE
               May 7, 2004 : Complete all function prototypes
              Nov 24, 2004 : Force -m for very large sizes
                             Add -bf automatically if generating trees
               Apr 1, 2007 : Write >A in one fputs() to try to reduce
                             mixing of outputs in multi-process pipes.
              Sep 19, 2007 : Force -m for n > 28 regardless of word size.
              Nov 29, 2008 : Slightly improved connectivity testing.
              Mar 3,  2015 : Improve maxdeg tweaking.
              Jan 18, 2016 : Replace bigint by nauty_counter.
               Mar 8, 2018 : Can now compile for MAXN up to WORDSIZE.
                             Use setword instead of unsigned for xword.
                             Revised splitting level.
                             Updated sample execution times.
              Mar 10, 2018 : Fix overflow at impossibly large n, maxdeg.
              Jan 14, 2019 : Define geng_mindeg, geng_maxdeg, geng_mine, geng_maxe.
               Jun 1, 2021 : Define geng_connec.
               Jun 4, 2021 : Improve performance for -c and -C with small edge count.
              Jun 21, 2021 : K1 is not 2-connected.
              May 15, 2022 : findmax() now deposits -1 at the end of the extended
                              sequence in case geng is being called as a function.
              Oct 10, 2022 : Obsolete y-format removed

**************************************************************************/

  /* 1 = geng, 2 = genbg, 3 = gentourng, 4 = gentreeg, 5 = genktreeg */
#define NAUTY_PGM  1

#ifndef MAXN
#define MAXN WORDSIZE         /* not more than WORDSIZE */
#endif

#if MAXN > WORDSIZE || MAXN > 64
 #error "Can't have MAXN greater than min(64,WORDSIZE)"
#endif

#define ONE_WORD_SETS
#include "gtools.h"   /* which includes nauty.h and stdio.h */

/* No need for TLS if not calling from a program. */
#ifndef GENG_MAIN
#undef TLS_ATTR
#define TLS_ATTR
#endif

typedef setword xword;

static TLS_ATTR void (*outproc)(FILE*,graph*,int);

static TLS_ATTR FILE *outfile;           /* file for output graphs */
static TLS_ATTR int connec;              /* 1 for -c, 2 for -C, 0 for neither */
static TLS_ATTR boolean bipartite;       /* presence of -b */
static TLS_ATTR boolean trianglefree;    /* presence of -t */
static TLS_ATTR boolean squarefree;      /* presence of -f */
static TLS_ATTR boolean pentagonfree;    /* presence of -p */
static TLS_ATTR boolean k4free;          /* presence of -k */
static TLS_ATTR boolean splitgraph;      /* presence of -S */
static TLS_ATTR boolean chordal;         /* presence of -T */
static TLS_ATTR boolean perfect;         /* presence of -P */
static TLS_ATTR boolean clawfree;        /* presence of -F */
static TLS_ATTR boolean savemem;         /* presence of -m */
static TLS_ATTR boolean verbose;         /* presence of -v */
boolean TLS_ATTR nautyformat;            /* presence of -n */
boolean TLS_ATTR graph6;                 /* presence of -g */
boolean TLS_ATTR sparse6;                /* presence of -s */
boolean TLS_ATTR nooutput;               /* presence of -u */
boolean TLS_ATTR canonise;               /* presence of -l */
boolean TLS_ATTR quiet;                  /* presence of -q */
boolean TLS_ATTR header;                 /* presence of -h */
statsblk TLS_ATTR nauty_stats;
static TLS_ATTR int mindeg,maxdeg,maxn,mine,maxe,mod,res;
#define PRUNEMULT 50   /* bigger -> more even split at greater cost */
static TLS_ATTR int min_splitlevel,odometer,splitlevel,multiplicity;
static TLS_ATTR graph gcan[MAXN];

#define XBIT(i) ((xword)1 << (i))
#define XPOPCOUNT(x) POPCOUNT(x)
#define XNEXTBIT(x) (WORDSIZE-1-FIRSTBITNZ(x))   /* Assumes non-zero */

typedef struct
{
    int ne,dmax;          /* values used for xlb,xub calculation */
    int xlb,xub;          /* saved bounds on extension degree */
    xword lo,hi;          /* work purposes for orbit calculation */
    xword xstart[MAXN+1]; /* index into xset[] for each cardinality */
    xword *xset;          /* array of all x-sets in card order */
    xword *xcard;         /* cardinalities of all x-sets */
    xword *xinv;          /* map from x-set to index in xset */
    xword *xorb;          /* min orbit representative */
    xword *xx;            /* (-b, -t, -s, -m) candidate x-sets */
                      /*   note: can be the same as xcard */
    xword xlim;           /* number of x-sets in xx[] */
} leveldata;

static TLS_ATTR leveldata data[MAXN];      /* data[n] is data for n -> n+1 */
static TLS_ATTR nauty_counter ecount[1+MAXN*(MAXN-1)/2];  /* counts by number of edges */
static TLS_ATTR nauty_counter nodes[MAXN];     /* nodes at each level */

#ifdef INSTRUMENT
static TLS_ATTR nauty_counter rigidnodes[MAXN],fertilenodes[MAXN];
static TLS_ATTR nauty_counter a1calls,a1nauty,a1succs;
static TLS_ATTR nauty_counter a2calls,a2nauty,a2uniq,a2succs;
#endif

/* The numbers below are actual maximum edge counts.
   geng works correctly with any upper bounds.
   To extend known upper bounds upwards:
       (n-1, E) -> (n, E + floor(2*E/(n-2))),
   which is done by the procedure findmaxe().
*/

static TLS_ATTR int maxeb[66] =     /* max edges for -b */
 {0,0,1,2,4, -1};
static TLS_ATTR int maxet[66] =     /* max edges for -t */
 {0,0,1,2,4, -1};
static TLS_ATTR int maxef[66] =     /* max edges for -f */
 {0,0,1,3,4, 6,7,9,11,13,
  16,18,21,24,27, 30,33,36,39,42,
  46,50,52,56,59, 63,67,71,76,80,
  85,90,92,96,102, 106,110,113,117,122,
  127, -1};
static TLS_ATTR int maxep[66] =    /* max edges for -p */
 {0,0,1,3,6, 7,9,12,16,20,
  25,30,36,42,49, 56,64,72,81,90,
  100,110,121,132,144, 156,169,182,196,210,
  225,240,256, -1};
static TLS_ATTR int maxeft[66] =    /* max edges for -ft */
 {0,0,1,2,3, 5,6,8,10,12,
  15,16,18,21,23, 26,28,31,34,38,
  41,44,47,50,54, 57,61,65,68,72,
  76,80,85,87,90, 95,99,104,109,114,
  120,124,129,134,139, 145,150,156,162,168,
  175,176,178, -1};
static TLS_ATTR int maxept[66] =    /* max edges for -pt */
 {0,0,1,2,4, 6,9,12,16,20,
  25,30,36,42,49, 56,64,72,81,90,
  100,110,121,132,144, 156,169,182,196,210,
  225,240,256,-1};
static TLS_ATTR int maxepf[66] =    /* max edges for -pf */
 {0,0,1,3,4, 6,7,9,10,12,
  14,16,18,20,23, 25,28,30,33,35,
  38,42,43,45,48, 50,53,55,58,62,
  65,67,70,73,77, 79,82,86,89,93,
  96,100,105,107, -1};
static TLS_ATTR int maxepft[66] =    /* max edges for -pft */
 {0,0,1,2,3, 4,6,7,9,10,
  12,14,16,18,21, 22,24,26,29,31,
  34,36,39,42,45, 48,52,53,56,58,
  61,64,67,70,74, 77,81,84,88,92,
  96,100,105,106,108, 110,115,118,122,126,
  130,134,138,142,147, 151,156,160,165,170,
  175,180,186,187,189, -1};
static TLS_ATTR int maxebf[66] =    /* max edges for -bf */
  {0,0,1,2,3, 4,6,7,9,10,
  12,14,16,18,21, 22,24,26,29,31,
  34,36,39,42,45, 48,52,53,56,58,
  61,64,67,70,74, 77,81,84,88,92,
  96,100,105,106,108, 110,115,118,122,126,
  130,134,138,142,147, 151,156,160,165,170,
  175,180,186,187, -1};

#ifdef PLUGIN
#include PLUGIN
#endif

#ifdef OUTPROC
extern void OUTPROC(FILE*,graph*,int);
#endif
#ifdef PRUNE
extern int PRUNE(graph*,int,int);
#endif
#ifdef PREPRUNE
extern int PREPRUNE(graph*,int,int);
#endif
#ifdef SUMMARY
extern void SUMMARY(nauty_counter,double);
#endif

#if defined(PRUNE) || defined(PREPRUNE)
int TLS_ATTR geng_mindeg, geng_maxdeg, geng_mine, geng_maxe, geng_connec;
#endif

/************************************************************************/

#define EXTEND(table,n) ((n) <= 1 ? 0 : (n) == 2 ? 1 : \
     table[(n)-1] + (2*table[(n)-1]/((n)-2)))

static int
findmaxe(int *table, int n)
/* Extend table to MAXN vertices if necessary, and return table[n]. */
{
    int i;

    for (i = 0; i <= MAXN && table[i] >= 0; ++i) {}
    for ( ; i <= MAXN; ++i) table[i] = EXTEND(table,i);

    return table[n];
}

/************************************************************************/

void
writeg6x(FILE *f, graph *g, int n)
/* write graph g (n vertices) to file f in graph6 format */
{
    writeg6(f,g,1,n);
}

/************************************************************************/

void
writes6x(FILE *f, graph *g, int n)
/* write graph g (n vertices) to file f in sparse6 format */
{
    writes6(f,g,1,n);
}

/***********************************************************************/

static void
nullwrite(FILE *f, graph *g, int n)
/* don't write graph g (n vertices) to file f */
{
}

/***********************************************************************/

void
writenauty(FILE *f, graph *g, int n)
/* write graph g (n vertices) to file f in nauty format.
   Each graph is preceded by the number of vertices. */
{
    int nn;

    nn = n;

    if (fwrite((char *)&nn,sizeof(int),(size_t)1,f) != 1 ||
          fwrite((char*)g,sizeof(graph),(size_t)n,f) != n)
    {
        fprintf(stderr,">E writenauty : error on writing file\n");
        exit(2);
    }
}

/*********************************************************************/

static boolean
isconnected(graph *g, int n)
/* test if g is connected */
{
    setword seen,expanded,toexpand,allbits;
    int i;

    allbits = ALLMASK(n);

    expanded = bit[n-1];
    seen = expanded | g[n-1];

    while (seen != allbits && (toexpand = (seen & ~expanded))) /* not == */
    {
        i = FIRSTBITNZ(toexpand);
        expanded |= bit[i];
        seen |= g[i];
    }

    return  seen == allbits;
}

static boolean
connpreprune(graph *g, int n, int maxn)
/* This function speeds up the generation of connected graphs
   with not many edges. */
{
    setword notvisited,queue;
    int ne,nc,i;

    if (n == maxn || maxe - maxn >= 5) return 0;

    ne = 0;
    for (i = 0; i < n; ++i) ne += POPCOUNT(g[i]);
    ne /= 2;

    nc = 0;
    notvisited = ALLMASK(n);

    while (notvisited)
    {
        ++nc;
        queue = SWHIBIT(notvisited);
        notvisited &= ~queue;
        while (queue)
        {
            TAKEBIT(i,queue);
            notvisited &= ~bit[i];
            queue |= g[i] & notvisited;
        }
    }

    if (ne - n + nc > maxe - maxn + 1) return TRUE;

    return FALSE;
}

/**********************************************************************/
 
static boolean
isbiconnected(graph *g, int n)
/* test if g is biconnected */
{
    int sp,v,w;
    setword sw;
    setword visited;
    int numvis,num[MAXN],lp[MAXN],stack[MAXN];
 
    if (n <= 2) return FALSE;
 
    visited = bit[0];
    stack[0] = 0;
    num[0] = 0;
    lp[0] = 0;
    numvis = 1;
    sp = 0;
    v = 0;
 
    for (;;)
    {
        if ((sw = g[v] & ~visited))           /* not "==" */
        {
            w = v;
            v = FIRSTBITNZ(sw);       /* visit next child */
            stack[++sp] = v;
            visited |= bit[v];
            lp[v] = num[v] = numvis++;
            sw = g[v] & visited & ~bit[w];
            while (sw)
            {
                w = FIRSTBITNZ(sw);
                sw &= ~bit[w];
                if (num[w] < lp[v])  lp[v] = num[w];
            }
        }
        else
        {
            w = v;                  /* back up to parent */
            if (sp <= 1)          return numvis == n;
            v = stack[--sp];
            if (lp[w] >= num[v])  return FALSE;
            if (lp[w] < lp[v])    lp[v] = lp[w];
        }
    }
}

/**********************************************************************/

static void
gcomplement(graph *g, graph *gc, int n)
/* Take the complement of g and put it in gc */
{
    int i;
    setword all;

    all = ~(setword)BITMASK(n-1);
    for (i = 0; i < n; ++i)
        gc[i] = g[i] ^ all ^ bit[i];
}

/**********************************************************************/

static boolean
distinvar(graph *g, int *invar, int n)
/* make distance invariant
   return FALSE if n-1 not maximal else return TRUE */
{
    int w;
    setword workset,frontier;
    setword sofar;
    int inv,d,v;

    for (v = n-1; v >= 0; --v)
    {
        inv = 0;
        sofar = frontier = bit[v];
        for (d = 1; frontier != 0; ++d)
        {
            workset = 0;
            inv += POPCOUNT(frontier) ^ (0x57 + d);
            while (frontier)
            {
                w = FIRSTBITNZ(frontier);
                frontier ^= bit[w];
                workset |= g[w];
            }
            frontier = workset & ~sofar;
            sofar |= frontier;
        }
        invar[v] = inv;
        if (v < n-1 && inv > invar[n-1]) return FALSE;
    }
    return TRUE;
}

/**************************************************************************/

static void
maketgraph(graph *g, xword *h, int n)
/* make x-format graph from nauty format graph */
{
    setword gi;
    int i,j;
    xword hi;

    for (i = 0; i < n; ++i)
    {
        hi = 0;
        gi = g[i];
        while (gi)
        {
            j = FIRSTBITNZ(gi);
            gi ^= bit[j];
            hi |= XBIT(j);
        }
        h[i] = hi;
    }
}

/**************************************************************************/

static void
make0graph(graph *g, xword *h, int n)
/* make x-format graph without edges */
{
    int i;

    for (i = 0; i < n; ++i) h[i] = 0;
}

/**************************************************************************/

static void
makebgraph(graph *g, xword *h, int n)
/* make x-format graph of different colour graph */
{
    setword seen1,seen2,expanded,w;
    setword restv;
    xword xseen1,xseen2;
    int i;

    restv = 0;
    for (i = 0; i < n; ++i) restv |= bit[i];

    seen1 = seen2 = 0;
    expanded = 0;

    while (TRUE)
    {
        if ((w = ((seen1 | seen2) & ~expanded)) == 0)
        {
            xseen1 = 0;
            w = seen1;
            while (w)
            {
                i = FIRSTBITNZ(w);
                w ^= bit[i];
                xseen1 |= XBIT(i);
            }
            xseen2 = 0;
            w = seen2;
            while (w)
            {
                i = FIRSTBITNZ(w);
                w ^= bit[i];
                xseen2 |= XBIT(i);
            }

            w = seen1;
            while (w)
            {
                i = FIRSTBITNZ(w);
                w ^= bit[i];
                h[i] = xseen2;
            }
            w = seen2;
            while (w)
            {
                i = FIRSTBITNZ(w);
                w ^= bit[i];
                h[i] = xseen1;
            }

            restv &= ~(seen1 | seen2);
            if (restv == 0) return;
            i = FIRSTBITNZ(restv);
            seen1 = bit[i];
            seen2 = 0;
        }
        else
            i = FIRSTBITNZ(w);

        expanded |= bit[i];
        if (bit[i] & seen1) seen2 |= g[i];
        else                seen1 |= g[i];
    }
}

/**************************************************************************/
 
static void
makeb6graph(graph *g, xword *h, int n)
/* make x-format bipartite girth 6 graph */
{
    setword w,x;
    xword hi;
    int i,j;

    makebgraph(g,h,n);

    for (i = 0; i < n; ++i)
    {
        w = g[i];
        x = 0;
        while (w)
        {
            j = FIRSTBITNZ(w);
            w ^= bit[j];
            x |= g[j];
        }
        x &= ~bit[i];
        hi = h[i];
        while (x)
        {
            j = FIRSTBITNZ(x);
            x ^= bit[j];
            hi |= XBIT(j);
        }
        h[i] = hi;
    }
}

/**************************************************************************/
 
static void
makefgraph(graph *g, xword *h, int n)
/* make x-format square graph */
{
    setword w,x;
    xword hi;
    int i,j;

    for (i = 0; i < n; ++i)
    {
        w = g[i];
        x = 0;
        while (w)
        {
            j = FIRSTBITNZ(w);
            w ^= bit[j];
            x |= g[j];
        }
        x &= ~bit[i];
        hi = 0;
        while (x)
        {
            j = FIRSTBITNZ(x);
            x ^= bit[j];
            hi |= XBIT(j);
        }
        h[i] = hi;
    }
}

/**************************************************************************/ 
 
static void 
makeftgraph(graph *g, xword *h, int n)
/* make x-format girth-5 graph */
{
    setword w,x; 
    xword hi;
    int i,j;
 
    for (i = 0; i < n; ++i)
    { 
        w = g[i]; 
        x = g[i];
        while (w) 
        {
            j = FIRSTBITNZ(w);
            w ^= bit[j];
            x |= g[j];
        } 
        x &= ~bit[i]; 
        hi = 0; 
        while (x) 
        { 
            j = FIRSTBITNZ(x); 
            x ^= bit[j]; 
            hi |= XBIT(j); 
        } 
        h[i] = hi; 
    } 
} 

/**************************************************************************/ 
 
static void 
makeplus5graph(graph *g, xword *h, int n, setword mask)
/* Make x-format graph for cycle lengths in mask plus 5-cycles */
{
    int v0,v3,v4,v5;
    setword w3,w4,w5;
    xword x,x3,x4;

    x3 = x4 = 0;
    if ((bit[3] & mask)) x3 = ~x3;
    if ((bit[4] & mask)) x4 = ~x4;
    for (v0 = 0; v0 < n; ++v0)
    {
        x = 0;
        w3 = g[v0];
        while (w3)
        {
            TAKEBIT(v3,w3);
            x |= (x3 & XBIT(v3));
            w4 = g[v3] & ~bit[v0];
            while (w4)
            {
                TAKEBIT(v4,w4);
                x |= (x4 & XBIT(v4));
                w5 = g[v4] & ~bit[v0] & ~bit[v3];
                while (w5)
                {
                    TAKEBIT(v5,w5);
                    x |= XBIT(v5);
                }
            }
        }
        h[v0] = x;
    }
}

/**************************************************************************/ 
 
static void 
makepgraph(graph *g, xword *h, int n)
/* Make x-format graph for 5-cycles */
{
    makeplus5graph(g,h,n,0);
}

/**************************************************************************/ 
 
static void 
makeptgraph(graph *g, xword *h, int n)
/* Make x-format graph for 3-cycles and 5-cycles */
{
    makeplus5graph(g,h,n,bit[3]);
}

/**************************************************************************/ 
 
static void 
makepfgraph(graph *g, xword *h, int n)
/* Make x-format graph for 4-cycles and 5-cycles */
{
    makeplus5graph(g,h,n,bit[4]);
}

/**************************************************************************/ 
 
static void 
makepftgraph(graph *g, xword *h, int n)
/* Make x-format graph for girth at least 6 */
{
    makeplus5graph(g,h,n,bit[3]|bit[4]);
}

/**************************************************************************/  

static xword
arith(xword a, xword b, xword c)
/* Calculate a*b/c, assuming a*b/c and (c-1)*b are representable integers */
{
    return (a/c)*b + ((a%c)*b)/c;
}

/**************************************************************************/  

static void
makeleveldata(boolean restricted)
/* make the level data for each level */
{
    long h;
    int n,nn;
    xword ncj;
    leveldata *d;
    xword *xcard,*xinv;
    xword *xset,xw,nxsets;
    xword cw;
    xword i,ilast,j;
    size_t tttn;

    for (n = 1; n < maxn; ++n)
    {
        nn = maxdeg <= n ? maxdeg : n;
        ncj = nxsets = 1;
        for (j = 1; j <= nn; ++j)
        { 
            ncj = arith(ncj,n-j+1,j);
            nxsets += ncj;
        }

        d = &data[n];
        d->ne = d->dmax = d->xlb = d->xub = -1;

        if (restricted)
        {
            d->xorb = (xword*) calloc(nxsets,sizeof(xword));
            d->xx = (xword*) calloc(nxsets,sizeof(xword));
            if (d->xorb == NULL || d->xx == NULL)
            {
                fprintf(stderr,
                   ">E geng: calloc failed in makeleveldata()\n");
                exit(2);
            }
            continue;   /* <--- NOTE THIS! */
        }

        tttn = (size_t)1 << n;
        d->xset = xset = (xword*) calloc(nxsets,sizeof(xword));
        d->xcard = xcard = (xword*) calloc(nxsets,sizeof(xword));
        d->xinv = xinv = (xword*) calloc(tttn,sizeof(xword));
        d->xorb = (xword*) calloc(nxsets,sizeof(xword));
        d->xx = d->xcard;

        if (xset==NULL || xcard==NULL || xinv==NULL || d->xorb==NULL)
        {
            fprintf(stderr,">E geng: calloc failed in makeleveldata()\n");
            exit(2);
        }

        j = 0;

        ilast = (n == WORDSIZE ? ~(setword)0 : XBIT(n)-1);
        for (i = 0;; ++i)
        {
            if ((h = XPOPCOUNT(i)) <= maxdeg)
            {
                xset[j] = i;
                xcard[j] = h;
                ++j;
            }
            if (i == ilast) break;
        }

        if (j != nxsets)
        {
            fprintf(stderr,">E geng: j=" SETWORD_DEC_FORMAT 
                               " nxsets=" SETWORD_DEC_FORMAT "\n",
                    j,nxsets);
            exit(2);
        }

        h = 1;
        do
            h = 3 * h + 1;
        while (h < nxsets);

        do     /* Shell sort, consider replacing */
        {
            for (i = h; i < nxsets; ++i)
            {
                xw = xset[i];
                cw = xcard[i];
                for (j = i; xcard[j-h] > cw ||
                            (xcard[j-h] == cw && xset[j-h] > xw); )
                {
                    xset[j] = xset[j-h];
                    xcard[j] = xcard[j-h];
                    if ((j -= h) < h) break;
                }
                xset[j] = xw;
                xcard[j] = cw;
            }
            h /= 3;
        }
        while (h > 0);

        for (i = 0; i < nxsets; ++i) xinv[xset[i]] = i;

        d->xstart[0] = 0;
        for (i = 1; i < nxsets; ++i)
            if (xcard[i] > xcard[i-1]) d->xstart[xcard[i]] = i;
        d->xstart[xcard[nxsets-1]+1] = nxsets;
    }
}

/**************************************************************************/

static void
userautomproc(int count, int *p, int *orbits,
          int numorbits, int stabvertex, int n)
/* form orbits on powerset of VG
   called by nauty;  operates on data[n] */
{
    xword i,j1,j2,moved,pi,pxi;
    xword lo,hi;
    xword *xorb,*xinv,*xset,w;

    xorb = data[n].xorb;
    xset = data[n].xset;
    xinv = data[n].xinv;
    lo = data[n].lo;
    hi = data[n].hi;

    if (count == 1)                         /* first automorphism */
        for (i = lo; i < hi; ++i) xorb[i] = i;

    moved = 0;
    for (i = 0; i < n; ++i)
        if (p[i] != i) moved |= XBIT(i);

    for (i = lo; i < hi; ++i)
    {
        if ((w = xset[i] & moved) == 0) continue;
        pxi = xset[i] & ~moved;
        while (w)
        {
            j1 = XNEXTBIT(w);
            w ^= XBIT(j1);
            pxi |= XBIT(p[j1]);
        }
        pi = xinv[pxi];

        j1 = xorb[i];
        while (xorb[j1] != j1) j1 = xorb[j1];
        j2 = xorb[pi];
        while (xorb[j2] != j2) j2 = xorb[j2];

        if      (j1 < j2) xorb[j2] = xorb[i] = xorb[pi] = j1;
        else if (j1 > j2) xorb[j1] = xorb[i] = xorb[pi] = j2;
    }
}

/**************************************************************************/

static void
userautomprocb(int count, int *p, int *orbits,
          int numorbits, int stabvertex, int n)
/* form orbits on powerset of VG
   called by nauty;  operates on data[n] */
{
    xword j1,j2,moved,pi,pxi,lo,hi,x;
    xword i,*xorb,*xx,w,xlim,xlb;

    xorb = data[n].xorb;
    xx = data[n].xx;
    xlim = data[n].xlim;

    if (count == 1)                         /* first automorphism */
    {
        j1 = 0;
        xlb = data[n].xlb;

        for (i = 0; i < xlim; ++i)
        {
            x = xx[i];
            if (XPOPCOUNT(x) >= xlb)
            {
                xx[j1] = x;
                xorb[j1] = j1;
                ++j1;
            }
        }
        data[n].xlim = xlim = j1;
    }

    moved = 0;
    for (i = 0; i < n; ++i)
        if (p[i] != i) moved |= XBIT(i);

    for (i = 0; i < xlim; ++i)
    {
        if ((w = xx[i] & moved) == 0) continue;
        pxi = xx[i] & ~moved;
        while (w)
        {
            j1 = XNEXTBIT(w);
            w ^= XBIT(j1);
            pxi |= XBIT(p[j1]);
        }
        /* pi = position of pxi */

        lo = 0;
        hi = xlim - 1;

        for (;;)
        {
            pi = (lo + hi) >> 1;
            if (xx[pi] == pxi) break;
            else if (xx[pi] < pxi) lo = pi + 1;
            else                   hi = pi - 1;
        }

        j1 = xorb[i];
        while (xorb[j1] != j1) j1 = xorb[j1];
        j2 = xorb[pi];
        while (xorb[j2] != j2) j2 = xorb[j2];

        if      (j1 < j2) xorb[j2] = xorb[i] = xorb[pi] = j1;
        else if (j1 > j2) xorb[j1] = xorb[i] = xorb[pi] = j2;
    }
}

/*****************************************************************************
*                                                                            *
*  refinex(g,lab,ptn,level,numcells,count,active,goodret,code,m,n) is a      *
*  custom version of refine() which can exit quickly if required.            *
*                                                                            *
*  Only use at level==0.                                                     *
*  goodret : whether to do an early return for code 1                        *
*  code := -1 for n-1 not max, 0 for maybe, 1 for definite                   *
*                                                                            *
*****************************************************************************/

static void
refinex(graph *g, int *lab, int *ptn, int level, int *numcells,
     int *count, set *active, boolean goodret, int *code, int m, int n)
{
    int i,c1,c2,labc1;
    setword x,lact;
    int split1,split2,cell1,cell2;
    int cnt,bmin,bmax;
    set *gptr;
    setword workset;
    int workperm[MAXN];
    int bucket[MAXN+2];

    if (n == 1)
    {
        *code = 1;
        return;
    }

    *code = 0;
    lact = *active;

    while (*numcells < n && lact)
    {
        TAKEBIT(split1,lact);
        
        for (split2 = split1; ptn[split2] > 0; ++split2) {}
        if (split1 == split2)       /* trivial splitting cell */
        {
            gptr = GRAPHROW(g,lab[split1],1);
            for (cell1 = 0; cell1 < n; cell1 = cell2 + 1)
            {
                for (cell2 = cell1; ptn[cell2] > 0; ++cell2) {}
                if (cell1 == cell2) continue;

                c1 = cell1;
                c2 = cell2;
                while (c1 <= c2)
                {
                    labc1 = lab[c1];
                    if (ISELEMENT1(gptr,labc1))
                        ++c1;
                    else
                    {
                        lab[c1] = lab[c2];
                        lab[c2] = labc1;
                        --c2;
                    }
                }
                if (c2 >= cell1 && c1 <= cell2)
                {
                    ptn[c2] = 0;
                    ++*numcells;
                    lact |= bit[c1];
                }
            }
        }

        else        /* nontrivial splitting cell */
        {
            workset = 0;
            for (i = split1; i <= split2; ++i) workset |= bit[lab[i]];

            for (cell1 = 0; cell1 < n; cell1 = cell2 + 1)
            {
                for (cell2 = cell1; ptn[cell2] > 0; ++cell2) {}
                if (cell1 == cell2) continue;
                i = cell1;
                if ((x = workset & g[lab[i]]) != 0) cnt = POPCOUNT(x);
                else                                cnt = 0;
                count[i] = bmin = bmax = cnt;
                bucket[cnt] = 1;
                while (++i <= cell2)
                {
                    if ((x = workset & g[lab[i]]) != 0)
                        cnt = POPCOUNT(x);
                    else
                        cnt = 0;

                    while (bmin > cnt) bucket[--bmin] = 0;
                    while (bmax < cnt) bucket[++bmax] = 0;
                    ++bucket[cnt];
                    count[i] = cnt;
                }
                if (bmin == bmax) continue;
                c1 = cell1;
                for (i = bmin; i <= bmax; ++i)
                    if (bucket[i])
                    {
                        c2 = c1 + bucket[i];
                        bucket[i] = c1;
                        if (c1 != cell1)
                        {
                            lact |= bit[c1];
                            ++*numcells;
                        }
                        if (c2 <= cell2) ptn[c2-1] = 0;
                        c1 = c2;
                    }
                for (i = cell1; i <= cell2; ++i)
                    workperm[bucket[count[i]]++] = lab[i];
                for (i = cell1; i <= cell2; ++i) lab[i] = workperm[i];
            }
        }

        if (ptn[n-2] == 0)
        {
            if (lab[n-1] == n-1)
            {
                *code = 1;
                if (goodret) return;
            }
            else
            {
                *code = -1;
                return;
            }
        }
        else
        {
            i = n - 1;
            while (TRUE)
            {
                if (lab[i] == n-1) break;
                --i;
                if (ptn[i] == 0)
                {
                    *code = -1;
                    return;
                }
            }
        }
    }
}

/**************************************************************************/

static void
makecanon(graph *g, graph *gcan, int n)
/* gcan := canonise(g) */
{
    int lab[MAXN],ptn[MAXN],orbits[MAXN];
    static TLS_ATTR DEFAULTOPTIONS_GRAPH(options);
    setword workspace[50];

    options.getcanon = TRUE;

    nauty(g,lab,ptn,NULL,orbits,&options,&nauty_stats,
          workspace,50,1,n,gcan);
}

/**************************************************************************/

static boolean
hask4(graph *g, int n, int maxn)
/* Return TRUE iff there is a K4 including the last vertex */
{
    setword gx,w;
    int i,j;

    gx = g[n-1];
    while (gx)
    {
        TAKEBIT(i,gx);
        w = g[i] & gx;
        while (w)
        {
            TAKEBIT(j,w);
            if ((g[j] & w)) return TRUE;
        }
    }
    return FALSE;
}

/**************************************************************************/

static boolean
hasclaw(graph *g, int n, int maxn)
/* Return TRUE if there is a claw (induced K(1,3)) involving the last vertex */
{
    int i,j,k;
    setword x,y;

    x = g[n-1];
    while (x)
    {
        TAKEBIT(j,x);
        y = x & ~g[j];
        while (y)
        {
            TAKEBIT(k,y);
            if (y & ~g[k]) return TRUE;
        }
    }

    x = g[n-1];
    while (x)
    {
        TAKEBIT(i,x);
        y = g[i] & ~(bit[n-1]|g[n-1]);
        while (y)
        {
            TAKEBIT(k,y);
            if (y & ~g[k]) return TRUE;
        }
    }

    return FALSE;
}

static boolean
hasinducedpath(graph *g, int start, setword body, setword last)
/* return TRUE if there is an induced path in g starting at start,
   extravertices within body and ending in last.
 * {start}, body and last should be disjoint. */
{
    setword gs,w;
    int i;

    gs = g[start];
    if ((gs & last)) return TRUE;

    w = gs & body;
    while (w)
    {
        TAKEBIT(i,w);
        if (hasinducedpath(g,i,body&~gs,last&~bit[i]&~gs))
            return TRUE;
    }

    return FALSE;
}

static boolean
notchordal(graph *g, int n, int maxn)
/* g is a graph of order n. Return TRUE if there is a
   chordless cycle of length at least 4 that includes
   the last vertex. */
{
    setword all,gn,gs;
    int v,s;

    all = ALLMASK(n);
    gn = g[n-1];

    while (gn)
    {
        TAKEBIT(v,gn);
        gs = g[v] & ~(bit[n-1]|g[n-1]);
        while (gs)
        {
            TAKEBIT(s,gs);
            if (hasinducedpath(g,s,all&~(g[n-1]|g[v]),gn&~g[v]))
                return TRUE;
        }
    }

    return FALSE;
}

static boolean
hasinducedpath5(graph *g, int start, setword body, setword last)
/* return TRUE if there is an induced path in g starting at start,
   extravertices within body and ending in last.
 * {start}, body and last should be disjoint. */
{
    setword gs,w;
    int i;

    gs = g[start];

    w = gs & body;
    while (w)
    {
        TAKEBIT(i,w);
        if (hasinducedpath(g,i,body&~gs,last&~bit[i]&~gs))
            return TRUE;
    }

    return FALSE;
}

static boolean
notchordal5(graph *g, int n, int maxn)
/* g is a graph of order n. Return TRUE if there is a
   chordless cycle of length at least 5 that includes
   the last vertex. */
{
    setword all,gn,gs;
    int v,s;

    all = ALLMASK(n);
    gn = g[n-1];

    while (gn)
    {
        TAKEBIT(v,gn);
        gs = g[v] & ~(bit[n-1]|g[n-1]);
        while (gs)
        {
            TAKEBIT(s,gs);
            if (hasinducedpath5(g,s,all&~(g[n-1]|g[v]),gn&~g[v]))
                return TRUE;
        }
    }

    return FALSE;
}

static boolean
notsplit(graph *g, int n, int maxn)
/* g is a graph of order n. Return TRUE if either g or its
   complement has a chordless cycle of length at least 4 that
   includes the last vertex. */
{
    graph gc[MAXN];
    setword w;
    int i;

    if (notchordal(g,n,maxn)) return TRUE;

    w = ALLMASK(n);
    for (i = 0; i < n; ++i) gc[i] = g[i] ^ w ^ bit[i];
    return notchordal(gc,n,maxn);
}

static boolean
hasinducedoddpath(graph *g, int start, setword body, setword last, boolean parity)
/* return TRUE if there is an induced path of odd length >= 3 in g
   starting at start, extravertices within body and ending in last.
   {start}, body and last should be disjoint. */
{
    setword gs,w;
    int i;

    gs = g[start];
    if ((gs & last) && parity) return TRUE;

    w = gs & body;
    while (w)
    {
        TAKEBIT(i,w);
        if (hasinducedoddpath(g,i,body&~gs,last&~bit[i]&~gs,!parity))
            return TRUE;
    }

    return FALSE;
}

static boolean
oddchordless(graph *g, int n, int maxn)
/* g is a graph of order n. Return TRUE if there is a
   chordless cycle of odd length at least 5 that includes
   the last vertex. */
{
    setword all,gn,gs;
    int v,s;

    all = ALLMASK(n);
    gn = g[n-1];

    while (gn)
    {
        TAKEBIT(v,gn);
        gs = g[v] & ~(bit[n-1]|g[n-1]);
        while (gs)
        {
            TAKEBIT(s,gs);
            if (hasinducedoddpath(g,s,all&~(g[n-1]|g[v]),gn&~g[v],FALSE))
                return TRUE;
        }
    }

    return FALSE;
}

static boolean
notperfect(graph *g, int n, int maxn)
/* g is a graph of order n. Return TRUE if either g or its
   complement has a chordless cycle of odd length at least 5 that
   includes the last vertex. I.e., if it is not perfect. */
{
    graph gc[MAXN];
    setword w;
    int i;

    if (oddchordless(g,n,maxn)) return TRUE;

    w = ALLMASK(n);
    for (i = 0; i < n; ++i) gc[i] = g[i] ^ w ^ bit[i];
    return oddchordless(gc,n,maxn);
}

/**************************************************************************/

static boolean
accept1(graph *g, int n, xword x, graph *gx, int *deg, boolean *rigid)
/* decide if n in theta(g+x) -  version for n+1 < maxn */
{
    int i;
    int lab[MAXN],ptn[MAXN],orbits[MAXN];
    int count[MAXN];
    graph h[MAXN];
    xword xw;
    int nx,numcells,code;
    int i0,i1,degn;
    set active[MAXM];
    statsblk stats;
    static TLS_ATTR DEFAULTOPTIONS_GRAPH(options);
    setword workspace[50];

#ifdef INSTRUMENT
    ++a1calls;
#endif

    nx = n + 1;
    for (i = 0; i < n; ++i) gx[i] = g[i];
    gx[n] = 0;
    deg[n] = degn = XPOPCOUNT(x);

    xw = x;
    while (xw)
    {
        i = XNEXTBIT(xw);
        xw ^= XBIT(i);
        gx[i] |= bit[n];
        gx[n] |= bit[i];
        ++deg[i];
    }

    if (k4free && hask4(gx,n+1,maxn)) return FALSE;
    if (clawfree && hasclaw(gx,n+1,maxn)) return FALSE;
#ifdef PREPRUNE
    if (PREPRUNE(gx,n+1,maxn)) return FALSE;
#endif
    if (connec == 2 && n+2 == maxn && !isconnected(gx,n+1)) return FALSE;
    if (((connec ==2 && n+2 < maxn) || (connec == 1 && n+2 <= maxn))
           && connpreprune(gx,n+1,maxn)) return FALSE;

    i0 = 0;
    i1 = n;
    for (i = 0; i < nx; ++i)
    {
        if (deg[i] == degn) lab[i1--] = i;
        else                lab[i0++] = i;
        ptn[i] = 1;
    }
    ptn[n] = 0;
    if (i0 == 0)
    {
        numcells = 1;
        active[0] = bit[0];
    }
    else
    {
        numcells = 2;
        active[0] = bit[0] | bit[i1+1];
        ptn[i1] = 0;
    }
    refinex(gx,lab,ptn,0,&numcells,count,active,FALSE,&code,1,nx);

    if (code < 0) return FALSE;

    if (numcells == nx)
    {
        *rigid = TRUE;
#ifdef INSTRUMENT
        ++a1succs;
#endif
        return TRUE;
    }

    options.getcanon = TRUE;
    options.defaultptn = FALSE;
    options.userautomproc = userautomproc;

    active[0] = 0;
#ifdef INSTRUMENT
    ++a1nauty;
#endif
    nauty(gx,lab,ptn,active,orbits,&options,&stats,workspace,50,1,nx,h);

    if (orbits[lab[n]] == orbits[n])
    {
        *rigid = stats.numorbits == nx;
#ifdef INSTRUMENT
        ++a1succs;
#endif
        return TRUE;
    }
    else
        return FALSE;
}

/**************************************************************************/

static boolean
accept1b(graph *g, int n, xword x, graph *gx, int *deg, boolean *rigid,
     void (*makeh)(graph*,xword*,int))
/* decide if n in theta(g+x)  --  version for n+1 < maxn */
{
    int i,v;
    xword z,hv,bitv,ixx;
    int lab[MAXN],ptn[MAXN],orbits[MAXN];
    int count[MAXN];
    graph gc[MAXN];
    xword h[MAXN],xw,jxx,kxx,*xx;
    int nx,numcells,code;
    int i0,i1,degn,xubx;
    set active[MAXM];
    statsblk stats;
    static TLS_ATTR DEFAULTOPTIONS_GRAPH(options);
    setword workspace[50];

#ifdef INSTRUMENT
    ++a1calls;
#endif

    nx = n + 1;
    for (i = 0; i < n; ++i) gx[i] = g[i];
    gx[n] = 0;
    deg[n] = degn = XPOPCOUNT(x);

    xw = x;
    while (xw)
    {
        i = XNEXTBIT(xw);
        xw ^= XBIT(i);
        gx[i] |= bit[n];
        gx[n] |= bit[i];
        ++deg[i];
    }

    if (k4free && hask4(gx,n+1,maxn)) return FALSE;
    if (clawfree && hasclaw(gx,n+1,maxn)) return FALSE;
#ifdef PREPRUNE
    if (PREPRUNE(gx,n+1,maxn)) return FALSE;
#endif
    if (connec == 2 && n+2 == maxn && !isconnected(gx,n+1)) return FALSE;
    if (((connec ==2 && n+2 < maxn) || (connec == 1 && n+2 <= maxe))
           && connpreprune(gx,n+1,maxn)) return FALSE;

    i0 = 0;
    i1 = n;
    for (i = 0; i < nx; ++i)
    {
        if (deg[i] == degn) lab[i1--] = i;
        else                lab[i0++] = i;
        ptn[i] = 1;
    }
    ptn[n] = 0;
    if (i0 == 0)
    {
        numcells = 1;
        active[0] = bit[0];
    }
    else
    {
        numcells = 2;
        active[0] = bit[0] | bit[i1+1];
        ptn[i1] = 0;
    }
    refinex(gx,lab,ptn,0,&numcells,count,active,FALSE,&code,1,nx);

    if (code < 0) return FALSE;

    (*makeh)(gx,h,nx);
    xx = data[nx].xx;
    xubx = data[nx].xub;

    xx[0] = 0;
    kxx = 1;
    for (v = 0; v < nx; ++v)
    {
        bitv = XBIT(v);
        hv = h[v];
        jxx = kxx;
        for (ixx = 0; ixx < jxx; ++ixx)
        if ((hv & xx[ixx]) == 0)
        {
            z = xx[ixx] | bitv;
            if (XPOPCOUNT(z) <= xubx) xx[kxx++] = z;
        }
    }
    data[nx].xlim = kxx;

    if (numcells == nx)
    {
        *rigid = TRUE;
#ifdef INSTRUMENT
        ++a1succs;
#endif
        return TRUE;
    }

    options.getcanon = TRUE;
    options.defaultptn = FALSE;
    options.userautomproc = userautomprocb;

    active[0] = 0;
#ifdef INSTRUMENT
    ++a1nauty;
#endif
    nauty(gx,lab,ptn,active,orbits,&options,&stats,workspace,50,1,nx,gc);

    if (orbits[lab[n]] == orbits[n])
    {
        *rigid = stats.numorbits == nx;
#ifdef INSTRUMENT
        ++a1succs;
#endif
        return TRUE;
    }
    else
        return FALSE;
}

/**************************************************************************/

static boolean
accept2(graph *g, int n, xword x, graph *gx, int *deg, boolean nuniq)
/* decide if n in theta(g+x)  --  version for n+1 == maxn */
{
    int i;
    int lab[MAXN],ptn[MAXN],orbits[MAXN];
    int degx[MAXN],invar[MAXN];
    setword vmax,gv;
    int qn,qv;
    int count[MAXN];
    xword xw;
    int nx,numcells,code;
    int degn,i0,i1,j,j0,j1;
    set active[MAXM];
    statsblk stats;
    static TLS_ATTR DEFAULTOPTIONS_GRAPH(options);
    setword workspace[50];
    boolean cheapacc;

#ifdef INSTRUMENT
    ++a2calls;
    if (nuniq) ++a2uniq;
#endif
    nx = n + 1;
    for (i = 0; i < n; ++i)
    {
        gx[i] = g[i];
        degx[i] = deg[i];
    }
    gx[n] = 0;
    degx[n] = degn = XPOPCOUNT(x);

    xw = x;
    while (xw)
    {
        i = XNEXTBIT(xw);
        xw ^= XBIT(i);
        gx[i] |= bit[n];
        gx[n] |= bit[i];
        ++degx[i];
    }

    if (k4free && hask4(gx,n+1,maxn)) return FALSE;
    if (clawfree && hasclaw(gx,n+1,maxn)) return FALSE;
#ifdef PREPRUNE
    if (PREPRUNE(gx,n+1,maxn)) return FALSE;
#endif
    if (connec == 2 && n+2 == maxn && !isconnected(gx,n+1)) return FALSE;
    if (((connec ==2 && n+2 < maxn) || (connec == 1 && n+2 <= maxe))
           && connpreprune(gx,n+1,maxn)) return FALSE;

    if (nuniq)
    {
#ifdef INSTRUMENT
        ++a2succs;
#endif
        if (canonise) makecanon(gx,gcan,nx);
        return TRUE;
    }

    i0 = 0;
    i1 = n;
    for (i = 0; i < nx; ++i)
    {
        if (degx[i] == degn) lab[i1--] = i;
        else                 lab[i0++] = i;
        ptn[i] = 1;
    }
    ptn[n] = 0;
    if (i0 == 0)
    {
        numcells = 1;
        active[0] = bit[0];

        if (!distinvar(gx,invar,nx)) return FALSE;
        qn = invar[n];
        j0 = 0;
        j1 = n;
        while (j0 <= j1)
        {
            j = lab[j0];
            qv = invar[j];
            if (qv < qn)
                ++j0;
            else
            {
                lab[j0] = lab[j1];
                lab[j1] = j;
                --j1;
            }
        }
        if (j0 > 0)
        {
            if (j0 == n)
            {
#ifdef INSTRUMENT
                ++a2succs;
#endif
                if (canonise) makecanon(gx,gcan,nx);
                return TRUE;
            }
            ptn[j1] = 0;
            ++numcells;
            active[0] |= bit[j0];
        }
    }
    else
    {
        numcells = 2;
        ptn[i1] = 0;
        active[0] = bit[0] | bit[i1+1];

        vmax = 0;
        for (i = i1+1; i < nx; ++i) vmax |= bit[lab[i]];

        gv = gx[n] & vmax;
        qn = POPCOUNT(gv);

        j0 = i1+1;
        j1 = n;
        while (j0 <= j1)
        {
            j = lab[j0];
            gv = gx[j] & vmax;
            qv = POPCOUNT(gv);
            if (qv > qn)
                return FALSE;
            else if (qv < qn)
                ++j0;
            else
            {
                lab[j0] = lab[j1];
                lab[j1] = j;
                --j1;
            }
        }
        if (j0 > i1+1)
        {
            if (j0 == n)
            {
#ifdef INSTRUMENT
                ++a2succs;
#endif
                if (canonise) makecanon(gx,gcan,nx);
                return TRUE;
            }
            ptn[j1] = 0;
            ++numcells;
            active[0] |= bit[j0];
        }
    }

    refinex(gx,lab,ptn,0,&numcells,count,active,TRUE,&code,1,nx);

    if (code < 0) return FALSE;

    cheapacc = FALSE;
    if (code > 0 || numcells >= nx-4)
        cheapacc = TRUE;
    else if (numcells == nx-5)
    {
        for (j1 = nx-2; j1 >= 0 && ptn[j1] > 0; --j1) {}
        if (nx - j1 != 5) cheapacc = TRUE;
    }
    else
    {
        j1 = nx;
        j0 = 0;
        for (i1 = 0; i1 < nx; ++i1)
        {
            --j1;
            if (ptn[i1] > 0)
            {
                ++j0;
                while (ptn[++i1] > 0) {}
            }
        }
        if (j1 <= j0 + 1) cheapacc = TRUE;
    }
    
    if (cheapacc)
    {
#ifdef INSTRUMENT
        ++a2succs;
#endif
        if (canonise) makecanon(gx,gcan,nx);
        return TRUE;
    }

    options.getcanon = TRUE;
    options.defaultptn = FALSE;

    active[0] = 0;
#ifdef INSTRUMENT
    ++a2nauty;
#endif
    nauty(gx,lab,ptn,active,orbits,&options,&stats,workspace,50,1,nx,gcan);

    if (orbits[lab[n]] == orbits[n])
    {
#ifdef INSTRUMENT
        ++a2succs;
#endif
        if (canonise) makecanon(gx,gcan,nx);
        return TRUE;
    }
    else
        return FALSE;
}

/**************************************************************************/

static void
xbnds(int n, int ne, int dmax)
/* find bounds on extension degree;  store answer in data[*].*  */
{
    int xlb,xub,d,nn,m,xc;

    xlb = n == 1 ? 0 : (dmax > (2*ne + n - 2)/(n - 1) ?
                        dmax : (2*ne + n - 2)/(n - 1));
    xub = n < maxdeg ? n : maxdeg;

    for (xc = xub; xc >= xlb; --xc)
    {
        d = xc;
        m = ne + d;
        for (nn = n+1; nn < maxn; ++nn)
        {
            if (d < (2*m + nn - 2)/(nn - 1)) d = (2*m + nn - 2)/(nn - 1);
            m += d;
        }
        if (d > maxdeg || m > maxe) xub = xc - 1;
        else                        break;
    }

    if (ne + xlb < mine)
        for (xc = xlb; xc <= xub; ++xc)
        {
            m = ne + xc;
            for (nn = n + 1; nn < maxn; ++nn)
                m += maxdeg < nn ? maxdeg : nn;
            if (m < mine) xlb = xc + 1;
            else          break;
        }

    data[n].ne = ne;
    data[n].dmax = dmax;
    data[n].xlb = xlb;
    data[n].xub = xub;
}

/**************************************************************************/

static void
spaextend(graph *g, int n, int *deg, int ne, boolean rigid,
      int xlb, int xub, void (*makeh)(graph*,xword*,int))
/* extend from n to n+1 -- version for restricted graphs */
{
    xword x,d,dlow;
    xword xlim,*xorb;
    int xc,nx,i,j,dmax,dcrit,xlbx,xubx;
    graph gx[MAXN];
    xword *xx,ixx;
    int degx[MAXN];
    boolean rigidx;

#ifdef INSTRUMENT
    boolean haschild;

    haschild = FALSE;
    if (rigid) ++rigidnodes[n];
#endif
    ++nodes[n];

    nx = n + 1;
    dmax = deg[n-1];
    dcrit = mindeg - maxn + n;
    d = dlow = 0;
    for (i = 0; i < n; ++i)
    {
        if (deg[i] == dmax) d |= XBIT(i);
        if (deg[i] == dcrit) dlow |= XBIT(i);
    }

    if (xlb == dmax && XPOPCOUNT(d) + dmax > n) ++xlb;
    if (nx == maxn && xlb < mindeg) xlb = mindeg;
    if (xlb > xub) return;

    if (splitgraph && notsplit(g,n,maxn)) return;
    if (chordal && ((bipartite && notchordal5(g,n,maxn))
                  || (!bipartite && notchordal(g,n,maxn)))) return;
    if (perfect && notperfect(g,n,maxn)) return;
#ifdef PRUNE
    if (PRUNE(g,n,maxn)) return;
#endif

    xorb = data[n].xorb;
    xx = data[n].xx;
    xlim = data[n].xlim;

    if (nx == maxn)
    {
        for (ixx = 0; ixx < xlim; ++ixx)
        {
            x = xx[ixx];
            xc = XPOPCOUNT(x);
            if (xc < xlb || xc > xub) continue;
            if ((rigid || xorb[ixx] == ixx) 
                && (xc > dmax || (xc == dmax && (x & d) == 0))
                && (dlow & ~x) == 0)
            {
                if (accept2(g,n,x,gx,deg,
                    xc > dmax+1 || (xc == dmax+1 && (x & d) == 0))
                    && (!connec ||
                          (connec==1 && isconnected(gx,nx)) ||
                          (connec>1 && isbiconnected(gx,nx))))
                {
                    if (splitgraph && notsplit(gx,nx,maxn)) continue;
		    if (chordal && ((bipartite && notchordal5(gx,nx,maxn))
			  || (!bipartite && notchordal(gx,nx,maxn)))) continue;
                    if (perfect && notperfect(gx,nx,maxn)) continue;
#ifdef PRUNE
                    if (!PRUNE(gx,nx,maxn))
#endif
                    {
#ifdef INSTRUMENT
                        haschild = TRUE;
#endif
                        ++ecount[ne+xc];
                        (*outproc)(outfile,canonise ? gcan : gx,nx);
                    }
                }
            }
        }
    }
    else
    {
        for (ixx = 0; ixx < xlim; ++ixx)
        {
            if (nx == splitlevel)
            {
                if (odometer-- != 0) continue;
                odometer = mod - 1;
            }
            x = xx[ixx];
            xc = XPOPCOUNT(x);
            if (xc < xlb || xc > xub) continue;
            if ((rigid || xorb[ixx] == ixx)
                && (xc > dmax || (xc == dmax && (x & d) == 0))
                && (dlow & ~x) == 0)
            {
                for (j = 0; j < n; ++j) degx[j] = deg[j];
                if (data[nx].ne != ne+xc || data[nx].dmax != xc)
                    xbnds(nx,ne+xc,xc);

                xlbx = data[nx].xlb;
                xubx = data[nx].xub;
                if (xlbx <= xubx
                    && accept1b(g,n,x,gx,degx,&rigidx,makeh))
                {
#ifdef INSTRUMENT
                    haschild = TRUE;
#endif
                    spaextend(gx,nx,degx,ne+xc,rigidx,xlbx,xubx,makeh);
                }
            }
        }
        if (n == splitlevel - 1 && n >= min_splitlevel
                                && nodes[n] >= multiplicity)
            --splitlevel;
    }
#ifdef INSTRUMENT
    if (haschild) ++fertilenodes[n];
#endif
}

/**************************************************************************/

static void
genextend(graph *g, int n, int *deg, int ne, boolean rigid, int xlb, int xub)
/* extend from n to n+1 -- version for general graphs */
{
    xword x,d,dlow;
    xword *xset,*xcard,*xorb;
    xword i,imin,imax;
    int nx,xc,j,dmax,dcrit;
    int xlbx,xubx;
    graph gx[MAXN];
    int degx[MAXN];
    boolean rigidx;

#ifdef INSTRUMENT
    boolean haschild;

    haschild = FALSE;
    if (rigid) ++rigidnodes[n];
#endif
    ++nodes[n];

    nx = n + 1;
    dmax = deg[n-1];
    dcrit = mindeg - maxn + n;
    d = dlow = 0;
    for (i = 0; i < n; ++i)
    {
        if (deg[i] == dmax) d |= XBIT(i);
        if (deg[i] == dcrit) dlow |= XBIT(i);
    }

    if (xlb == dmax && XPOPCOUNT(d) + dmax > n) ++xlb;
    if (nx == maxn && xlb < mindeg) xlb = mindeg;
    if (xlb > xub) return;

    if (splitgraph && notsplit(g,n,maxn)) return;
    if (chordal && ((bipartite && notchordal5(g,n,maxn))
	  || (!bipartite && notchordal(g,n,maxn)))) return;
    if (perfect && notperfect(g,n,maxn)) return;
#ifdef PRUNE 
    if (PRUNE(g,n,maxn)) return; 
#endif 

    imin = data[n].xstart[xlb];
    imax = data[n].xstart[xub+1];
    xset = data[n].xset;
    xcard = data[n].xcard;
    xorb = data[n].xorb;

    if (nx == maxn)
        for (i = imin; i < imax; ++i)
        {
            if (!rigid && xorb[i] != i) continue;
            x = xset[i];
            xc = (int)xcard[i];
            if (xc == dmax && (x & d) != 0) continue;
            if ((dlow & ~x) != 0) continue;

            if (accept2(g,n,x,gx,deg,
                        xc > dmax+1 || (xc == dmax+1 && (x & d) == 0)))
                if (!connec || (connec==1 && isconnected(gx,nx))
                            || (connec>1 && isbiconnected(gx,nx)))
                {
                    if (splitgraph && notsplit(gx,nx,maxn)) continue;
		    if (chordal && ((bipartite && notchordal5(gx,nx,maxn))
		        || (!bipartite && notchordal(gx,nx,maxn)))) continue;
                    if (perfect && notperfect(gx,nx,maxn)) continue;
#ifdef PRUNE
                    if (!PRUNE(gx,nx,maxn))
#endif
                    {
#ifdef INSTRUMENT
                        haschild = TRUE;
#endif
                        ++ecount[ne+xc];
                        (*outproc)(outfile,canonise ? gcan : gx,nx);
                    }
                }
        }
    else
        for (i = imin; i < imax; ++i)
        {
            if (!rigid && xorb[i] != i) continue;
            x = xset[i];
            xc = (int)xcard[i];
            if (xc == dmax && (x & d) != 0) continue;
            if ((dlow & ~x) != 0) continue;
            if (nx == splitlevel)
            {
                if (odometer-- != 0) continue;
                odometer = mod - 1;
            }

            for (j = 0; j < n; ++j) degx[j] = deg[j];
            if (data[nx].ne != ne+xc || data[nx].dmax != xc)
                xbnds(nx,ne+xc,xc);
            xlbx = data[nx].xlb;
            xubx = data[nx].xub;
            if (xlbx > xubx) continue;

            data[nx].lo = data[nx].xstart[xlbx];
            data[nx].hi = data[nx].xstart[xubx+1];
            if (accept1(g,n,x,gx,degx,&rigidx))
            {
#ifdef INSTRUMENT
                haschild = TRUE;
#endif
                genextend(gx,nx,degx,ne+xc,rigidx,xlbx,xubx);
            }
        }

    if (n == splitlevel-1 && n >= min_splitlevel
            && nodes[n] >= multiplicity)
        --splitlevel;
#ifdef INSTRUMENT
    if (haschild) ++fertilenodes[n];
#endif
}

/**************************************************************************/
/**************************************************************************/

#ifdef GENG_MAIN
int
GENG_MAIN(int argc, char *argv[])
#else
int
main(int argc, char *argv[])
#endif
{
    char *arg;
    boolean badargs,gote,gotmr,gotf,gotd,gotD,gotx,gotX;
    boolean secret,connec1,connec2,safe,sparse;
    char *outfilename,sw;
    int i,j,argnum;
    graph g[1];
    int tmaxe,deg[1];
    nauty_counter nout;
    int splitlevinc;
    double t1,t2;
    char msg[201];

    HELP; PUTVERSION;
    nauty_check(WORDSIZE,1,MAXN,NAUTYVERSIONID);

    badargs = FALSE;
    trianglefree = bipartite = squarefree = pentagonfree = FALSE;
    k4free = splitgraph = chordal = perfect = clawfree = FALSE;
    verbose = quiet = FALSE;
    nautyformat = graph6 = sparse6 = nooutput = FALSE;
    savemem = canonise = header = FALSE;
    outfilename = NULL;
    secret = safe = FALSE;
    connec1 = connec2 = FALSE;

    maxdeg = MAXN; 
    mindeg = 0;
    
    gotX = gotx = gotd = gotD = gote = gotmr = gotf = FALSE;

    argnum = 0;
    for (j = 1; !badargs && j < argc; ++j)
    {
        arg = argv[j];
        if (arg[0] == '-' && arg[1] != '\0')
        {
            ++arg;
            while (*arg != '\0')
            {
                sw = *arg++;
                     SWBOOLEAN('n',nautyformat)
                else SWBOOLEAN('u',nooutput)
                else SWBOOLEAN('g',graph6)
                else SWBOOLEAN('s',sparse6)
                else SWBOOLEAN('t',trianglefree)
                else SWBOOLEAN('f',squarefree)
                else SWBOOLEAN('p',pentagonfree)
                else SWBOOLEAN('k',k4free)
                else SWBOOLEAN('S',splitgraph)
                else SWBOOLEAN('T',chordal)
                else SWBOOLEAN('P',perfect)
                else SWBOOLEAN('F',clawfree)
                else SWBOOLEAN('b',bipartite)
                else SWBOOLEAN('v',verbose)
                else SWBOOLEAN('l',canonise)
                else SWBOOLEAN('h',header)
                else SWBOOLEAN('m',savemem)
                else SWBOOLEAN('c',connec1)
                else SWBOOLEAN('C',connec2)
                else SWBOOLEAN('q',quiet)
                else SWBOOLEAN('$',secret)
                else SWBOOLEAN('S',safe)
                else SWINT('d',gotd,mindeg,"geng -d")
                else SWINT('D',gotD,maxdeg,"geng -D")
                else SWINT('x',gotx,multiplicity,"geng -x")
                else SWINT('X',gotX,splitlevinc,"geng -X")
#ifdef PLUGIN_SWITCHES
PLUGIN_SWITCHES
#endif
                else badargs = TRUE;
            }
        }
        else if (arg[0] == '-' && arg[1] == '\0')
            gotf = TRUE;
        else
        {
            if (argnum == 0)
            {
                if (sscanf(arg,"%d",&maxn) != 1) badargs = TRUE;
                ++argnum;
            }
            else if (gotf)
                badargs = TRUE;
            else
            {
                if (!gotmr)
                {
                    if (sscanf(arg,"%d/%d",&res,&mod) == 2)
                    { 
                        gotmr = TRUE; 
                        continue; 
                    }
                }
                if (!gote)
                {
                    if (sscanf(arg,"%d:%d",&mine,&maxe) == 2
                     || sscanf(arg,"%d-%d",&mine,&maxe) == 2)
                    {
                        gote = TRUE;
                        if (maxe == 0 && mine > 0) maxe = MAXN*(MAXN-1)/2;
                        continue;
                    }
                    else if (sscanf(arg,"%d",&mine) == 1)
                    {
                        gote = TRUE;
                        maxe = mine;
                        continue;
                    }
                }
                if (!gotf)
                {
                    outfilename = arg;
                    gotf = TRUE;
                    continue;
                }
            }
        }
    }

    if (argnum == 0)
        badargs = TRUE;
    else if (maxn < 1 || maxn > MAXN || maxn > 64)
    {
        fprintf(stderr,">E geng: n must be in the range 1..%d\n",MAXN);
        badargs = TRUE;
    }

    if (!gotmr)
    {
        mod = 1;
        res = 0;
    }

    if (!gote)
    {
        mine = 0;
        maxe = (maxn*maxn - maxn) / 2;
    }

    if (trianglefree || squarefree || bipartite) k4free = FALSE;
    if (bipartite || chordal) perfect = FALSE;
		/* bipartite and chordal graphs are perfect */
    if (splitgraph && !bipartite) chordal = perfect = FALSE;
		/* split graphs are chordal */
    if (clawfree && bipartite)
    {
        clawfree = FALSE;
        if (maxdeg > 2) maxdeg = 2;
    }
    if (splitgraph && bipartite && maxe >= maxn) maxe = maxn - 1;

    if (connec1 && mindeg < 1 && maxn > 1) mindeg = 1;
    if (connec2 && mindeg < 2 && maxn > 2) mindeg = 2;
    if (maxdeg >= maxn) maxdeg = maxn - 1;
    if (maxe > maxn*maxdeg / 2) maxe = maxn*maxdeg / 2;
    if (maxdeg > maxe) maxdeg = maxe;
    if (mindeg < 0) mindeg = 0;
    if (mine < (maxn*mindeg+1) / 2) mine = (maxn*mindeg+1) / 2;
    if (maxdeg > 2*maxe - mindeg*(maxn-1)) maxdeg = 2*maxe - mindeg*(maxn-1);

    if      (connec2) connec = 2;
    else if (connec1) connec = 1;
    else              connec = 0;
    if (connec && mine < maxn-1) mine = maxn - 2 + connec;

#if defined(PRUNE) || defined(PREPRUNE)
    geng_mindeg = mindeg;
    geng_maxdeg = maxdeg;
    geng_mine = mine;
    geng_maxe = maxe;
    geng_connec = connec;
#endif

    if (!badargs && (mine > maxe || maxe < 0 || maxdeg < 0))
    {
        fprintf(stderr,
                ">E geng: impossible mine,maxe,mindeg,maxdeg values\n");
        badargs = TRUE;
    }

    if (!badargs && (res < 0 || res >= mod))
    {
        fprintf(stderr,">E geng: must have 0 <= res < mod\n");
        badargs = TRUE;
    }

    if (badargs)
    {
        fprintf(stderr,">E Usage: %s\n",USAGE);
        GETHELP;
        exit(1);
    }

    if ((nautyformat!=0) + (graph6!=0) + (sparse6!=0) + (nooutput!=0) > 1)
        gt_abort(">E geng: -ungs are incompatible\n");

#ifdef OUTPROC
    outproc = OUTPROC;
#else
    if (nautyformat)   outproc = writenauty;
    else if (nooutput) outproc = nullwrite;
    else if (sparse6)  outproc = writes6x;
    else               outproc = writeg6x;
#endif

#ifdef PLUGIN_INIT
PLUGIN_INIT
#endif

    for (i = 0; i <= maxe; ++i) ecount[i] = 0;
    for (i = 0; i < maxn; ++i)  nodes[i] = 0;

    if (nooutput)
        outfile = stdout;
    else if (!gotf || outfilename == NULL)
    {
        outfilename = "stdout";
        outfile = stdout;
    }
    else if ((outfile = fopen(outfilename,
                    nautyformat ? "wb" : "w")) == NULL)
        gt_abort_1(">E geng: can't open %s for writing\n",outfilename);

    if (bipartite)
    {
        if (squarefree)  tmaxe = findmaxe(maxebf,maxn);
        else             tmaxe = findmaxe(maxeb,maxn);
    }
    else if (trianglefree)
    {
        if (squarefree)
            if (pentagonfree) tmaxe = findmaxe(maxepft,maxn);
            else              tmaxe = findmaxe(maxeft,maxn);
        else
            if (pentagonfree) tmaxe = findmaxe(maxept,maxn);
            else              tmaxe = findmaxe(maxet,maxn);
    }
    else if (squarefree)
    {
        if (pentagonfree) tmaxe = findmaxe(maxepf,maxn);
        else              tmaxe = findmaxe(maxef,maxn);
    }
    else if (pentagonfree)
        tmaxe = findmaxe(maxep,maxn);
    else 
        tmaxe = (maxn*maxn - maxn) / 2;

    if (safe) ++tmaxe;

    if (maxe > tmaxe) maxe = tmaxe;

    if (gotx)
    {
        if (multiplicity < 3 * mod || multiplicity > 999999999)
            gt_abort(">E geng: -x value must be in [3*mod,10^9-1]\n");
    }
    else
    {
        multiplicity = PRUNEMULT * mod;
        if (multiplicity / PRUNEMULT != mod)
            gt_abort(">E geng: mod value is too large\n");
    }

    if (!gotX) splitlevinc = 0;

    if (!quiet)
    {
        msg[0] = '\0';
        if (strlen(argv[0]) > 75)
            fprintf(stderr,">A %s",argv[0]);
        else
            CATMSG1(">A %s",argv[0]);
       
        CATMSG8(" -%s%s%s%s%s%s%s%s",
            connec2      ? "C" : connec1 ? "c" : "",
            trianglefree ? "t" : "",
            squarefree   ? "f" : "",
            pentagonfree ? "p" : "",
            k4free       ? "k" : "",
            bipartite    ? "b" : "",
            canonise     ? "l" : "",
            savemem      ? "m" : "");
        if (splitgraph || chordal || perfect || clawfree)
            CATMSG4(" -%s%s%s%s",
                splitgraph ? "S" : "",
                chordal ? "T" : "",
                perfect ? "P" : "",
                clawfree ? "F" : "");
        if (mod > 1)
            CATMSG2("X%dx%d",splitlevinc,multiplicity);
        CATMSG4("d%dD%d n=%d e=%d",mindeg,maxdeg,maxn,mine);
        if (maxe > mine) CATMSG1("-%d",maxe);
        if (mod > 1) CATMSG2(" class=%d/%d",res,mod);
        CATMSG0("\n");
        fputs(msg,stderr);
        fflush(stderr);
    }

    g[0] = 0;
    deg[0] = 0;

    sparse = bipartite || squarefree || trianglefree
                       || pentagonfree || savemem;

    t1 = CPUTIME;

    if (header)
    {
        if (sparse6)
        {
            writeline(outfile,SPARSE6_HEADER);
            fflush(outfile);
        }
        else if (!nautyformat && !nooutput)
        {
            writeline(outfile,GRAPH6_HEADER);
            fflush(outfile);
        }
    }

    if (maxn == 1)
    {
        if (res == 0 && connec < 2)
        {
            ++ecount[0];
            (*outproc)(outfile,g,1);
        }
    }
    else
    {
        if (maxn > 28 || maxn+4 > 8*sizeof(xword))
            savemem = sparse = TRUE;
        if (maxn == maxe+1 && connec)
            bipartite = squarefree = sparse = TRUE;  /* trees */

        makeleveldata(sparse);

        if (maxn >= 14 && mod > 1)     splitlevel = maxn - 4;
        else if (maxn >= 6 && mod > 1) splitlevel = maxn - 3;
        else                           splitlevel = -1;

        if (splitlevel > 0) splitlevel += splitlevinc;
        if (splitlevel > maxn - 1) splitlevel = maxn - 1;
        if (splitlevel < 3) splitlevel = -1;

        min_splitlevel = 6;
        odometer = secret ? -1 : res;

        if (maxe >= mine && 
                (mod <= 1 || (mod > 1 && (splitlevel > 2 || res == 0))))
        {
            xbnds(1,0,0);   
            if (sparse)
            {
                data[1].xx[0] = 0;
                if (maxdeg > 0) data[1].xx[1] = XBIT(0);
                data[1].xlim = data[1].xub + 1;
            }

            if (bipartite)
            {
                if (squarefree)
                    spaextend(g,1,deg,0,TRUE,
                                    data[1].xlb,data[1].xub,makeb6graph);
                else
                    spaextend(g,1,deg,0,TRUE,
                                    data[1].xlb,data[1].xub,makebgraph);
            }
            else if (trianglefree)
            {
                if (squarefree)
                {
                    if (pentagonfree)
                        spaextend(g,1,deg,0,TRUE,
                                    data[1].xlb,data[1].xub,makepftgraph);
                    else
                        spaextend(g,1,deg,0,TRUE,
                                    data[1].xlb,data[1].xub,makeftgraph);
                }
                else
                {
                    if (pentagonfree)
                        spaextend(g,1,deg,0,TRUE,
                                    data[1].xlb,data[1].xub,makeptgraph);
                    else
                        spaextend(g,1,deg,0,TRUE,
                                    data[1].xlb,data[1].xub,maketgraph);
                }
            }
            else if (squarefree)
            {
                if (pentagonfree)
                    spaextend(g,1,deg,0,TRUE,
                                    data[1].xlb,data[1].xub,makepfgraph);
                else
                    spaextend(g,1,deg,0,TRUE,
                                    data[1].xlb,data[1].xub,makefgraph);
            }
            else if (pentagonfree)
            {
                    spaextend(g,1,deg,0,TRUE,
                                    data[1].xlb,data[1].xub,makepgraph);
            }
            else if (savemem)
                spaextend(g,1,deg,0,TRUE,
                                    data[1].xlb,data[1].xub,make0graph);
            else
                genextend(g,1,deg,0,TRUE,data[1].xlb,data[1].xub);
        }
    }
    t2 = CPUTIME;

    nout = 0;
    for (i = 0; i <= maxe; ++i) nout += ecount[i];

    if (verbose)
    {
        for (i = 0; i <= maxe; ++i)
            if (ecount[i] != 0)
            {
                fprintf(stderr,">C " COUNTER_FMT " graphs with %d edges\n",
                     ecount[i],i);
            }
    }

#ifdef INSTRUMENT
    fprintf(stderr,"\n>N node counts\n");
    for (i = 1; i < maxn; ++i)
    {
        fprintf(stderr," level %2d: ",i);
        fprintf(stderr,COUNTER_FMT " (" COUNTER_FMT
                       " rigid, " COUNTER_FMT " fertile)\n",
                       nodes[i],rigidnodes[i],fertilenodes[i]);
    }
    fprintf(stderr,">A1 " COUNTER_FMT " calls to accept1, "
                   COUNTER_FMT " nauty, " COUNTER_FMT " succeeded\n",
                   a1calls,a1nauty,a1succs);
    fprintf(stderr,">A2 " COUNTER_FMT " calls to accept2, " COUNTER_FMT
                   " nuniq, "COUNTER_FMT " nauty, " COUNTER_FMT " succeeded\n",
                   a2calls,a2uniq,a2nauty,a2succs);
    fprintf(stderr,"\n");
#endif

#ifdef SUMMARY
    SUMMARY(nout,t2-t1);
#endif

    if (!quiet)
    {
        fprintf(stderr,">Z " COUNTER_FMT " graphs generated in %3.2f sec\n",
                nout,t2-t1);
    }

#ifdef GENG_MAIN
    for (i = 1; i < maxn; ++i)
        if (sparse)
        {
            free(data[i].xorb); 
            free(data[i].xx);
        }
        else
        {
            free(data[i].xorb);
            free(data[i].xset);
            free(data[i].xinv);
            free(data[i].xcard);
        }
    return 0;
#else
    exit(0);
#endif
}
