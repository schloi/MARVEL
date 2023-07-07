
#ifndef _A_MODULE

#define _A_MODULE

#include "db/DB.h"
#include "types.h"
#include <inttypes.h>

#define TRACE_XOVR                                                                                                                                             \
    125 //  If the trace spacing is not more than this value, then can
        //    and do compress traces pts to 8-bit unsigned ints

#define COMP_FLAG 0x1
#define ACOMP_FLAG 0x2 //  A-sequence is complemented, not B !  Only Local_Alignment notices

#define COMP( x ) ( (x)&COMP_FLAG )
#define ACOMP( x ) ( (x)&ACOMP_FLAG )

typedef struct
{
    Path* path;
    size_t pathBufferOffset;
    uint32_t flags; /* Pipeline status and complementation flags          */
    char* aseq;     /* Pointer to A sequence                              */
    char* bseq;     /* Pointer to B sequence                              */
    int alen;       /* Length of A sequence                               */
    int blen;       /* Length of B sequence                               */
    int aread;
    int bread;
} Alignment;

void Complement_Seq( char* a, int n );

Work_Data* New_Work_Data();

void Free_Work_Data( Work_Data* work );

Align_Spec* New_Align_Spec( double ave_corr, int trace_space, float* freq, int reach, int nthreads );

void Free_Align_Spec( Align_Spec* spec );

int Trace_Spacing( Align_Spec* spec );

#define LOWERMOST -1 //   Possible modes for "mode" parameter below)
#define GREEDIEST 0
#define UPPERMOST 1

#define PLUS_ALIGN 0
#define PLUS_TRACE 1
#define DIFF_ONLY 2
#define DIFF_ALIGN 3
#define DIFF_TRACE 4

int Compute_Alignment( Alignment* align, Work_Data* work, int task, int trace_spacing );

int Write_Overlap( FILE* output, Overlap* ovl, int tbytes );

int Compress_TraceTo8( Overlap* ovl, int check );

#endif // _A_MODULE
