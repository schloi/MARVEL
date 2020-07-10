
#pragma once

#include "align.h"
#include "db/DB.h"
#include "ovlbuffer.h"

#define PANEL_SIZE 50000    //  Size to break up very long A-reads
#define PANEL_OVERLAP 10000 //  Overlap of A-panels

#define MATCH_CHUNK 100   //  Max initial number of hits between two reads
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
#define DO_BRIDGING

#undef TEST_GATHER
#undef TEST_CONTAIN
#undef TEST_BRIDGE
#undef SHOW_OVERLAP   //  Show the cartoon
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
extern int HGAP_MIN;
extern int SYMMETRIC;
extern int IDENTITY;
extern char* SORT_PATH;

extern uint64_t MEM_LIMIT;
extern uint64_t MEM_PHYSICAL;

typedef struct
{
    uint64 max;
    uint64 top;
    uint16* trace;
} Trace_Buffer;

typedef struct
{
    int64 beg, end;
    int* score;
    int* lastp;
    int* lasta;
    Work_Data* work;
    FILE* ofile1;
    FILE* ofile2;
    int64 nfilt;
    int64 nlas;
#ifdef PROFILE
    int profyes[ MAXHIT + 1 ];
    int profno[ MAXHIT + 1 ];
#endif

#ifdef ENABLE_OVL_IO_BUFFER
    Overlap_IO_Buffer* iobuf;
#endif
} Report_Arg;

typedef struct
{
    uint64 p1; //  The lower half
    uint64 p2;
} Double;

typedef struct
{
    int aread;
    int bread;
    int apos;
    int diag;
} SeedPair;

void Set_Filter_Params( int kmer, int binshift, int suppress, int hitmin, int nthreads );

void* Sort_Kmers( DAZZ_DB* block, int* len );

void Match_Filter( char* aname, DAZZ_DB* ablock, char* bname, DAZZ_DB* bblock,
                   void* atable, int alen, void* btable, int blen, Align_Spec* asettings );

void Clean_Exit( int val );

void Diagonal_Span( Path* path, int* mind, int* maxd );
int Handle_Redundancies( Path* amatch, int novls, Path* bmatch,
                                Alignment* align, Work_Data* work, Trace_Buffer* tbuf );
void CopyAndComp( char* bcomp, char* bseq, int blen );