#pragma once

#define MAXGRAM 10000 //  Cap on k-mer count histogram (in count_thread, merge_thread)

#include "stdint.h"
#ifdef __cplusplus__
extern "C"
{
#include "db/DB.h"
}
#else
#include "db/DB.h"
#endif

#define DAZZ_DB HITS_DB
#define DAZZ_TRACK HITS_TRACK
#define DAZZ_READ HITS_READ

/*** PATH ABSTRACTION:

     Coordinates are *between* characters where 0 is the tick just before the first char,
     1 is the tick between the first and second character, and so on.  Our data structure
     is called a Path refering to its conceptualization in an edit graph.

     A local alignment is specified by the point '(abpos,bbpos)' at which its path in
     the underlying edit graph starts, and the point '(aepos,bepos)' at which it ends.
     In otherwords A[abpos+1..aepos] is aligned to B[bbpos+1..bepos] (assuming X[1] is
     the *first* character of X).

     There are 'diffs' differences in an optimal local alignment between the beginning and
     end points of the alignment (if computed by Compute_Trace), or nearly so (if computed
     by Local_Alignment).

     Optionally, a Path can have additional information about the exact nature of the
     aligned substrings if the field 'trace' is not NULL.  Trace points to either an
     array of integers (if computed by a Compute_Trace routine), or an array of unsigned
     short integers (if computed by Local_Alignment).

     If computed by Local_Alignment 'trace' points at a list of 'tlen' (always even) short
     values:

            d_0, b_0, d_1, b_1, ... d_n-1, b_n-1, d_n, b_n

     to be interpreted as follows.  The alignment from (abpos,bbpos) to (aepos,bepos)
     passes through the n trace points for i in [1,n]:

            (a_i,b_i) where a_i = floor(abpos/TS)*TS + i*TS
                        and b_i = bbpos + (b_0 + b_1 + b_i-1)

     where also let a_0,b_0 = abpos,bbpos and a_(n+1),b_(n+1) = aepos,bepos.  That is, the
     interior (i.e. i != 0 and i != n+1) trace points pass through every TS'th position of
     the aread where TS is the "trace spacing" employed when finding the alignment (see
     New_Align_Spec).  Typically TS is 100.  Then d_i is the number of differences in the
     portion of the alignment between (a_i,b_i) and (a_i+1,b_i+1).  These trace points allow
     the Compute_Trace routines to efficiently compute the exact alignment between the two
     reads by efficiently computing exact alignments between consecutive pairs of trace points.
     Moreover, the diff values give one an idea of the quality of the alignment along every
     segment of TS symbols of the aread.

     If computed by a Compute_Trace routine, 'trace' points at a list of 'tlen' integers
     < i1, i2, ... in > that encodes an exact alignment as follows.  A negative number j
     indicates that a dash should be placed before A[-j] and a positive number k indicates
     that a dash should be placed before B[k], where A and B are the two sequences of the
     overlap.  The indels occur in the trace in the order in which they occur along the
     alignment.  For a good example of how to "decode" a trace into an alignment, see the
     code for the routine Print_Alignment.

***/

typedef struct
{
    void* trace;
    int tlen;
    int diffs;
    int abpos, bbpos;
    int aepos, bepos;
} Path;

/*** OVERLAP ABSTRACTION:

 Externally, between modules an Alignment is modeled by an "Overlap" record, which
 (a) replaces the pointers to the two sequences with their ID's in the HITS data bases,
 (b) does not contain the length of the 2 sequences (must fetch from DB), and
 (c) contains its path as a subrecord rather than as a pointer (indeed, typically the
 corresponding Alignment record points at the Overlap's path sub-record).  The trace pointer
 is always to a sequence of trace points and can be either compressed (uint8) or
 uncompressed (uint16).  One can read and write binary records of an "Overlap".
 ***/

typedef struct
{
    Path path;    /* Path: begin- and end-point of alignment + diffs    */
    uint32 flags; /* Pipeline status and complementation flags          */
    int aread;    /* Id # of A sequence                                 */
    int bread;    /* Id # of B sequence                                 */
} Overlap;

typedef struct
{
    uint32 rpos;
    uint32 read;
    uint64 code;
} KmerPos;

typedef struct
{
    int aread;
    int bread;
    int apos;
    int diag;
} SeedPair;

typedef struct
{
    int abeg, aend;
    int bbeg, bend;
    int64 nhits;
    int limit;
    int64 hitgram[ MAXGRAM ];

    KmerPos* MG_alist;
    KmerPos* MG_blist;
    DAZZ_DB* MG_ablock;
    DAZZ_DB* MG_bblock;
    SeedPair* MG_hits;
    int MG_self;

} Merge_Arg;

typedef void Work_Data;

typedef struct
{
    // trace buffer
    int tbytes;
    uint64_t tmax;
    uint64_t ttop;
    void* trace;
    int no_trace;
    // overlap buffer
    int omax;
    int otop;
    Overlap* ovls;
} Overlap_IO_Buffer;

typedef struct
{
    double ave_corr;
    int trace_space;
    int reach;
    float freq[ 4 ];
    int ave_path;
    int16_t* score;
    int16_t* table;

    int nthreads;

    Overlap_IO_Buffer* ioBuffer;

} Align_Spec;

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

    Overlap_IO_Buffer* iobuf;

    HITS_DB* ablock;
    HITS_DB* bblock;
    SeedPair* khit;
    int two;
    Align_Spec* aspec;
    int MG_self;

    DAZZ_DB* MR_ablock;
    DAZZ_DB* MR_bblock;
    SeedPair* MR_hits;
    int MR_two;

} Report_Arg;
