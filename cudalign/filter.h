
#pragma once

#include "align.h"
#include "db/DB.h"
#include "ovlbuffer.h"
#include "types.h"

#include <aio.h>
#include <sys/time.h>

typedef unsigned int uint;

#define TIMING

#ifdef TIMING
    #define INIT_TIMING struct timeval _timing_stop, _timing_start;
    #define START_TIMING gettimeofday(&_timing_start, NULL);
    #define END_TIMING(message) gettimeofday(&_timing_stop, NULL); printf("timing> %s %lu ms\n", message, ((_timing_stop.tv_sec - _timing_start.tv_sec) * 1000000 + _timing_stop.tv_usec - _timing_start.tv_usec)/1000);
#else
    #define INIT_TIMING
    #define START_TIMING
    #define END_TIMING
#endif

#define PANEL_SIZE 50000 //  Size to break up very long A-reads
#define PANEL_OVERLAP 10000 //  Overlap of A-panels

#define MATCH_CHUNK 100 //  Max initial number of hits between two reads
#define TRACE_CHUNK 20000 //  Max initial trace points in hits between two reads

#define DAZZ_DB HITS_DB
#define DAZZ_TRACK HITS_TRACK
#define DAZZ_READ HITS_READ

//  Debug Controls

#undef TEST_KSORT
#undef TEST_PAIRS
#undef TEST_CSORT
// #define    HOW_MANY   3000   //  Print first HOW_MANY items for each of the TEST options above

#define DO_ALIGNMENT
#undef DO_BRIDGING

#undef TEST_GATHER
#undef TEST_CONTAIN
#undef TEST_BRIDGE
#undef SHOW_OVERLAP //  Show the cartoon
#undef SHOW_ALIGNMENT //  Show the alignment

// #define   ALIGN_WIDTH    80   //     Parameters for alignment
// #define   ALIGN_INDENT   20
// #define   ALIGN_BORDER   10

#ifdef SHOW_OVERLAP
#define NOTHREAD
#endif

#ifdef TEST_GATHER
#define NOTHREAD
#endif

#ifdef TEST_CONTAIN
#define NOTHREAD
#endif

extern int BIASED;
extern int VERBOSE;
extern int MINOVER;
extern int SYMMETRIC;
extern int IDENTITY;
extern int MR_tspace;

extern uint64_t MEM_LIMIT;
extern uint64_t MEM_PHYSICAL;

extern int Binshift;
extern int Kmer;

typedef struct
{
    uint64 max;
    uint64 top;
    uint16* trace;
} Trace_Buffer;

typedef struct
{
    uint64 p1; //  The lower half
    uint64 p2;
} Double;

void Set_Filter_Params( int kmer, uint binshift, int suppress, int hitmin, int nthreads, int traceSpace );

KmerPos* Sort_Kmers( DAZZ_DB* block, int* len );

void Diagonal_Span( Path* path, int* mind, int* maxd );
int Handle_Redundancies( Path* amatch, int novls, Path* bmatch, Alignment* align, Work_Data* work, Trace_Buffer* tbuf );

// Exposing function used in the GPU version
int find_tuple( uint64 x, KmerPos* a, int n );
void* count_thread( void* arg );
void* merge_thread( void* arg );
