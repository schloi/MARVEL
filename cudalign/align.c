
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "align.h"
#include "db/DB.h"
#include "ovlbuffer.h"

#undef DEBUG_PASSES    //  Show forward / backward extension termini for Local_Alignment
#undef DEBUG_POINTS    //  Show trace points
#undef DEBUG_WAVE      //  Show waves of Local_Alignment
#undef SHOW_MATCH_WAVE //  For waves of Local_Alignment also show # of matches
#undef SHOW_TRAIL      //  Show trace at the end of forward and reverse passes
#undef SHOW_TPS        //  Show trace points as they are encountered in a wave

#undef DEBUG_EXTEND //  Show waves of Extend_Until_Overlap

#undef DEBUG_ALIGN  //  Show division points of Compute_Trace
#undef DEBUG_TRACE  //  Show trace additions for Compute_Trace
#undef DEBUG_SCRIPT //  Show script additions for Compute_Trace
#undef DEBUG_AWAVE  //  Show F/R waves of Compute_Trace

#undef SHOW_TRACE //  Show full trace for Print_Alignment

#undef WAVE_STATS

/****************************************************************************************\
*                                                                                        *
*  Working Storage Abstraction                                                           *
*                                                                                        *
\****************************************************************************************/

typedef struct //  Hidden from the user, working space for each thread
{
    int vecmax;
    void* vector;
    int celmax;
    void* cells;
    int pntmax;
    void* points;
    int tramax;
    void* trace;
    int alnmax;
    void* alnpts;
} _Work_Data;

Work_Data* New_Work_Data()
{
    _Work_Data* work;

    work = (_Work_Data*)Malloc( sizeof( _Work_Data ), "Allocating work data block" );
    if ( work == NULL )
        EXIT( NULL );
    work->vecmax = 0;
    work->vector = NULL;
    work->pntmax = 0;
    work->points = NULL;
    work->tramax = 0;
    work->trace  = NULL;
    work->alnmax = 0;
    work->alnpts = NULL;
    work->celmax = 0;
    work->cells  = NULL;
    return ( (Work_Data*)work );
}

static int enlarge_vector( _Work_Data* work, int newmax )
{
    void* vec;
    int max;

    max = ( (int)( newmax * 1.2 ) ) + 10000;
    vec = Realloc( work->vector, max, "Enlarging DP vector" );
    if ( vec == NULL )
        EXIT( 1 );
    work->vecmax = max;
    work->vector = vec;
    return ( 0 );
}

static int enlarge_alnpts( _Work_Data* work, int newmax )
{
    void* vec;
    int max;

    max = ( (int)( newmax * 1.2 ) ) + 10000;
    vec = Realloc( work->alnpts, max, "Enlarging point vector" );
    if ( vec == NULL )
        EXIT( 1 );
    work->alnmax = max;
    work->alnpts = vec;
    return ( 0 );
}

void Free_Work_Data( Work_Data* ework )
{
    _Work_Data* work = (_Work_Data*)ework;
    if ( work->vector != NULL )
        free( work->vector );
    if ( work->cells != NULL )
        free( work->cells );
    if ( work->trace != NULL )
        free( work->trace );
    if ( work->points != NULL )
        free( work->points );
    if ( work->alnpts != NULL )
        free( work->alnpts );
    free( work );
}

/****************************************************************************************\
*                                                                                        *
*  ADAPTIVE PATH FINDING                                                                 *
*                                                                                        *
\****************************************************************************************/

//  Absolute/Fixed Parameters

#define TRIM_LEN 15 //  Report as the tip, the last wave maximum for which the last
                    //     2*TRIM_LEN edits are prefix-positive at rate ave_corr*f(bias)
                    //     (max value is 20)

#define PATH_LEN 60 //  Follow the last PATH_LEN columns/edges (max value is 63)

//  Derivative fixed parameters

#define TRIM_MASK 0x7fff              //  Must be (1 << TRIM_LEN) - 1

static double Bias_Factor[ 10 ] = { .690, .690, .690, .690, .780, .850, .900, .933, .966, 1.000 };

/* Fill in bit table: TABLE[x] = 1 iff the alignment modeled by x (1 = match, 0 = mismatch)
     has a non-negative score for every suffix of the alignment under the scoring scheme
     where match = MATCH and mismatch = -1.  MATCH is set so that an alignment with TRIM_PCT
     matches has zero score ( (1-TRIM_PCT) / TRIM_PCT ).                                     */

#define FRACTION 1000 //  Implicit fractional part of scores, i.e. score = x/FRACTION

typedef struct
{
    int mscore;
    int dscore;
    int16_t* table;
    int16_t* score;
} Table_Bits;

static void set_table( int bit, int prefix, int score, int max, Table_Bits* parms )
{
    if ( bit >= TRIM_LEN )
    {
        parms->table[ prefix ] = ( int16 )( score - max );
        parms->score[ prefix ] = (int16)score;
    }
    else
    {
        if ( score > max )
            max = score;
        set_table( bit + 1, ( prefix << 1 ), score - parms->dscore, max, parms );
        set_table( bit + 1, ( prefix << 1 ) | 1, score + parms->mscore, max, parms );
    }
}

/* Create an alignment specification record including path tip tables & values */

Align_Spec* New_Align_Spec( double ave_corr, int trace_space, float* freq, int reach, int nthreads )
{
    Align_Spec* spec;
    Table_Bits parms;
    double match;
    int bias;

    spec = (Align_Spec*)Malloc( sizeof( Align_Spec ), "Allocating alignment specification" );
    if ( spec == NULL )
        EXIT( NULL );

    spec->ave_corr    = ave_corr;
    spec->trace_space = trace_space;
    spec->reach       = reach;
    spec->freq[ 0 ]   = freq[ 0 ];
    spec->freq[ 1 ]   = freq[ 1 ];
    spec->freq[ 2 ]   = freq[ 2 ];
    spec->freq[ 3 ]   = freq[ 3 ];

    match = freq[ 0 ] + freq[ 3 ];
    if ( match > .5 )
        match = 1. - match;
    bias = (int)( ( match + .025 ) * 20. - 1. );
    if ( match < .2 )
    {
        fprintf( stderr, "Warning: Base bias worse than 80/20%% ! (New_Align_Spec)\n" );
        fprintf( stderr, "         Capping bias at this ratio.\n" );
        bias = 3;
    }

    spec->ave_path = (int)( PATH_LEN * ( 1. - Bias_Factor[ bias ] * ( 1. - ave_corr ) ) );
    parms.mscore   = (int)( FRACTION * Bias_Factor[ bias ] * ( 1. - ave_corr ) );
    parms.dscore   = FRACTION - parms.mscore;

    parms.score = (int16*)Malloc( sizeof( int16 ) * ( TRIM_MASK + 1 ) * 2, "Allocating trim table" );
    if ( parms.score == NULL )
    {
        free( spec );
        EXIT( NULL );
    }
    parms.table = parms.score + ( TRIM_MASK + 1 );

    set_table( 0, 0, 0, 0, &parms );

    spec->table = parms.table;
    spec->score = parms.score;

    {
        spec->nthreads = nthreads;

        int i;
        spec->ioBuffer = malloc( sizeof( Overlap_IO_Buffer ) * nthreads );
        for ( i = 0; i < nthreads; i++ )
        {
            Overlap_IO_Buffer* ob = CreateOverlapBuffer( nthreads, ( trace_space <= TRACE_XOVR ) ? sizeof( uint8 ) : sizeof( uint16 ), 0 );
            if ( ob == NULL )
                exit( 1 );

            ( (Overlap_IO_Buffer*)( spec->ioBuffer ) )[ i ] = *ob;
            free( ob );
        }
    }

    return ( (Align_Spec*)spec );
}

void Free_Align_Spec( Align_Spec* espec )
{
    Align_Spec* spec = (Align_Spec*)espec;
    free( spec->score );
    free( spec );
}


int Trace_Spacing( Align_Spec* espec ) { return ( ( (Align_Spec*)espec )->trace_space ); }

/****************************************************************************************\
*                                                                                        *
*  OVERLAP MANIPULATION                                                                  *
*                                                                                        *
\****************************************************************************************/

static int64 PtrSize   = sizeof( void* );
static int64 OvlIOSize = sizeof( Overlap ) - sizeof( void* );

int Write_Overlap( FILE* output, Overlap* ovl, int tbytes )
{
    if ( fwrite( ( (char*)ovl ) + PtrSize, OvlIOSize, 1, output ) != 1 )
        return ( 1 );
    if ( ovl->path.trace != NULL )
        if ( fwrite( ovl->path.trace, tbytes, ovl->path.tlen, output ) != (size_t)ovl->path.tlen )
            return ( 1 );
    return ( 0 );
}

int Compress_TraceTo8( Overlap* ovl, int check )
{
    uint16* t16 = (uint16*)ovl->path.trace;
    uint8* t8   = (uint8*)ovl->path.trace;
    int j, x;

    if ( check )
        for ( j = 0; j < ovl->path.tlen; j++ )
        {
            x = t16[ j ];
            if ( x > 255 )
            {
                fprintf( stderr, "%s: Compression of trace to bytes fails, value too big\n", Prog_Name );
                EXIT( 1 );
            }
            t8[ j ] = (uint8)x;
        }
    else
        for ( j = 0; j < ovl->path.tlen; j++ )
            t8[ j ] = ( uint8 )( t16[ j ] );
    return ( 0 );
}

/****************************************************************************************\
*                                                                                        *
*  ALIGNMENT PRINTING                                                                    *
*                                                                                        *
\****************************************************************************************/

/* Complement the sequence in fragment aseq.  The operation does the
   complementation/reversal in place.  Calling it a second time on a
   given fragment restores it to its original state.                */

void Complement_Seq( char* aseq, int len )
{
    char *s, *t;
    int c;

    s = aseq;
    t = aseq + ( len - 1 );
    while ( s < t )
    {
        c    = 3 - *s;
        *s++ = (char)( 3 - *t );
        *t-- = (char)c;
    }
    if ( s == t )
        *s = (char)( 3 - *s );
}


/****************************************************************************************\
*                                                                                        *
*  O(ND) trace algorithm                                                                 *
*                                                                                        *
\****************************************************************************************/


typedef struct
{
    int* Stop;         //  Ongoing stack of alignment indels
    uint16* Trace;     //  Base of Trace Vector
    char *Aabs, *Babs; //  Absolute base of A and B sequences

    int **PVF, **PHF; //  List of waves for iterative np algorithms
    int mida, midb;   //  mid point division for mid-point algorithms

    int *VF, *VB; //  Forward/Reverse waves for nd algorithms
} Trace_Waves;

static int split_nd( char* A, int M, char* B, int N, Trace_Waves* wave, int* px, int* py )
{
    int x, y;
    int D;

    int* VF = wave->VF;
    int* VB = wave->VB;
    int flow; //  fhgh == D !
    int blow, bhgh;
    char* a;

    y = 0;
    if ( N < M )
        while ( y < N && B[ y ] == A[ y ] )
            y += 1;
    else
    {
        while ( y < M && B[ y ] == A[ y ] )
            y += 1;
        if ( y >= M && N == M )
        {
            *px = *py = M;
            return ( 0 );
        }
    }

    flow     = 0;
    VF[ 0 ]  = y;
    VF[ -1 ] = -2;

    x = N - M;
    a = A - x;
    y = N - 1;
    if ( N > M )
        while ( y >= x && B[ y ] == a[ y ] )
            y -= 1;
    else
        while ( y >= 0 && B[ y ] == a[ y ] )
            y -= 1;

    blow = bhgh = -x;
    VB += x;
    VB[ blow ]     = y;
    VB[ blow - 1 ] = N + 1;

    for ( D = 1; 1; D += 1 )
    {
        int k, r;
        int am, ac, ap;

        //  Forward wave

        flow -= 1;
        am = ac = VF[ flow - 1 ] = -2;

        a = A + D;
        x = M - D;
        for ( k = D; k >= flow; k-- )
        {
            ap = ac;
            ac = am + 1;
            am = VF[ k - 1 ];

            if ( ac < am )
                if ( ap < am )
                    y = am;
                else
                    y = ap;
            else if ( ap < ac )
                y = ac;
            else
                y = ap;

            if ( blow <= k && k <= bhgh )
            {
                r = VB[ k ];
                if ( y > r )
                {
                    D = ( D << 1 ) - 1;
                    if ( ap > r )
                        y = ap;
                    else if ( ac > r )
                        y = ac;
                    else
                        y = r + 1;
                    x   = k + y;
                    *px = x;
                    *py = y;
                    return ( D );
                }
            }

            if ( N < x )
                while ( y < N && B[ y ] == a[ y ] )
                    y += 1;
            else
                while ( y < x && B[ y ] == a[ y ] )
                    y += 1;

            VF[ k ] = y;
            a -= 1;
            x += 1;
        }

#ifdef DEBUG_AWAVE
        print_awave( VF, flow, D );
#endif

        //  Reverse Wave

        bhgh += 1;
        blow -= 1;
        am = ac = VB[ blow - 1 ] = N + 1;

        a = A + bhgh;
        x = -bhgh;
        for ( k = bhgh; k >= blow; k-- )
        {
            ap = ac + 1;
            ac = am;
            am = VB[ k - 1 ];

            if ( ac > am )
                if ( ap > am )
                    y = am;
                else
                    y = ap;
            else if ( ap > ac )
                y = ac;
            else
                y = ap;

            if ( flow <= k && k <= D )
            {
                r = VF[ k ];
                if ( y <= r )
                {
                    D = ( D << 1 );
                    if ( ap <= r )
                        y = ap;
                    else if ( ac <= r )
                        y = ac;
                    else
                        y = r;
                    x   = k + y;
                    *px = x;
                    *py = y;
                    return ( D );
                }
            }

            y -= 1;
            if ( x > 0 )
                while ( y >= x && B[ y ] == a[ y ] )
                    y -= 1;
            else
                while ( y >= 0 && B[ y ] == a[ y ] )
                    y -= 1;

            VB[ k ] = y;
            a -= 1;
            x += 1;
        }

#ifdef DEBUG_AWAVE
        print_awave( VB, blow, bhgh );
#endif
    }
}

static int trace_nd( char* A, int M, char* B, int N, Trace_Waves* wave, int tspace )
{
    int x, y;
    int D, s;

#ifdef DEBUG_ALIGN
    printf( "%*s %ld,%ld: %d vs %d\n", depth, "", A - wave->Aabs, B - wave->Babs, M, N );
    fflush( stdout );
#endif

    if ( M <= 0 )
    {
        y = ( ( ( A - wave->Aabs ) / tspace ) << 1 );
        wave->Trace[ y ] += N;
        wave->Trace[ y + 1 ] += N;
#ifdef DEBUG_TRACE
        printf( "%*s Adding1 (%d,%d) to tp %d(%d,%d)\n", depth, "", N, N, y >> 1, wave->Trace[ y + 1 ], wave->Trace[ y ] );
        fflush( stdout );
#endif
        return ( N );
    }

    if ( N <= 0 )
    {
        x = A - wave->Aabs;
        y = x / tspace;
        x = ( y + 1 ) * tspace - x;
        y <<= 1;
        for ( s = M; s > 0; s -= x, x = tspace )
        {
            if ( x > s )
                x = s;
            wave->Trace[ y ] += x;
#ifdef DEBUG_TRACE
            printf( "%*s Adding2 (0,%d) to tp %d(%d,%d)\n", depth, "", x, y >> 1, wave->Trace[ y + 1 ], wave->Trace[ y ] );
            fflush( stdout );
#endif
            y += 2;
        }
        return ( M );
    }

    D = split_nd( A, M, B, N, wave, &x, &y );

    if ( D > 1 )
    {
#ifdef DEBUG_ALIGN
        printf( "%*s (%d,%d) @ %d\n", depth, "", x, y, D );
        fflush( stdout );
        depth += 2;
#endif

        s = A - wave->Aabs;
        if ( ( s / tspace + 1 ) * tspace - s >= x )
        {
            s = ( ( s / tspace ) << 1 );
            wave->Trace[ s ] += ( D + 1 ) / 2;
            wave->Trace[ s + 1 ] += y;
#ifdef DEBUG_TRACE
            printf( "%*s Adding3 (%d,%d) to tp %d(%d,%d)\n", depth, "", y, ( D + 1 ) / 2, s >> 1, wave->Trace[ s + 1 ], wave->Trace[ s ] );
            fflush( stdout );
#endif
        }
        else
            trace_nd( A, x, B, y, wave, tspace );

        s = ( A + x ) - wave->Aabs;
        if ( ( s / tspace + 1 ) * tspace - s >= M - x )
        {
            s = ( ( s / tspace ) << 1 );
            wave->Trace[ s ] += D / 2;
            wave->Trace[ s + 1 ] += N - y;
#ifdef DEBUG_TRACE
            printf( "%*s Adding4 (%d,%d)) to tp %d(%d,%d)\n", depth, "", N - y, D / 2, s >> 1, wave->Trace[ s + 1 ], wave->Trace[ s ] );
            fflush( stdout );
#endif
        }
        else
            trace_nd( A + x, M - x, B + y, N - y, wave, tspace );

#ifdef DEBUG_ALIGN
        depth -= 2;
#endif
    }

    else
    {
        int u, v;

        if ( D == 0 || M < N )
            s = x;
        else
            s = x - 1;
        if ( s > 0 )
        {
            u = A - wave->Aabs;
            v = u / tspace;
            u = ( v + 1 ) * tspace - u;
            for ( v <<= 1; s > 0; s -= u, u = tspace )
            {
                if ( u > s )
                    u = s;
                wave->Trace[ v + 1 ] += u;
#ifdef DEBUG_TRACE
                printf( "%*s Adding5 (%d,0)) to tp %d(%d,%d)\n", depth, "", u, v >> 1, wave->Trace[ v + 1 ], wave->Trace[ v ] );
                fflush( stdout );
#endif
                v += 2;
            }
        }

        if ( D == 0 )
            return ( D );

        if ( M < N )
            y = ( ( ( ( A + x ) - wave->Aabs ) / tspace ) << 1 );
        else
            y = ( ( ( ( A + ( x - 1 ) ) - wave->Aabs ) / tspace ) << 1 );
        wave->Trace[ y ] += 1;
        if ( M <= N )
            wave->Trace[ y + 1 ] += 1;
#ifdef DEBUG_TRACE
        printf( "%*s Adding5 (%d,1)) to tp %d(%d,%d)\n", depth, "", N >= M, y >> 1, wave->Trace[ y + 1 ], wave->Trace[ y ] );
        fflush( stdout );
#endif

        s = M - x;
        if ( s > 0 )
        {
            u = ( A + x ) - wave->Aabs;
            v = u / tspace;
            u = ( v + 1 ) * tspace - u;
            for ( v <<= 1; s > 0; s -= u, u = tspace )
            {
                if ( u > s )
                    u = s;
                wave->Trace[ v + 1 ] += u;
#ifdef DEBUG_TRACE
                printf( "%*s Adding5 (%d,0)) to tp %d(%d,%d)\n", depth, "", u, v >> 1, wave->Trace[ v + 1 ], wave->Trace[ v ] );
                fflush( stdout );
#endif
                v += 2;
            }
        }
    }

    return ( D );
}

static int dandc_nd( char* A, int M, char* B, int N, Trace_Waves* wave )
{
    int x, y;
    int D;

#ifdef DEBUG_ALIGN
    printf( "%*s %ld,%ld: %d vs %d\n", depth, "", A - wave->Aabs, B - wave->Babs, M, N );
#endif

    if ( M <= 0 )
    {
        x = ( wave->Aabs - A ) - 1;
        for ( y = 1; y <= N; y++ )
        {
            *wave->Stop++ = x;
#ifdef DEBUG_SCRIPT
            printf( "%*s *I %ld(%ld)\n", depth, "", y + ( B - wave->Babs ), ( A - wave->Aabs ) + 1 );
#endif
        }
        return ( N );
    }

    if ( N <= 0 )
    {
        y = ( B - wave->Babs ) + 1;
        for ( x = 1; x <= M; x++ )
        {
            *wave->Stop++ = y;
#ifdef DEBUG_SCRIPT
            printf( "%*s *D %ld(%ld)\n", depth, "", x + ( A - wave->Aabs ), ( B - wave->Babs ) + 1 );
#endif
        }
        return ( M );
    }

    D = split_nd( A, M, B, N, wave, &x, &y );

    if ( D > 1 )
    {
#ifdef DEBUG_ALIGN
        printf( "%*s (%d,%d) @ %d\n", depth, "", x, y, D );
        fflush( stdout );
        depth += 2;
#endif

        dandc_nd( A, x, B, y, wave );
        dandc_nd( A + x, M - x, B + y, N - y, wave );

#ifdef DEBUG_ALIGN
        depth -= 2;
#endif
    }

    else if ( D == 1 )

    {
        if ( M > N )
        {
            *wave->Stop++ = ( B - wave->Babs ) + y + 1;
#ifdef DEBUG_SCRIPT
            printf( "%*s  D %ld(%ld)\n", depth, "", ( A - wave->Aabs ) + x, ( B - wave->Babs ) + y + 1 );
#endif
        }

        else if ( M < N )
        {
            *wave->Stop++ = ( wave->Aabs - A ) - x - 1;
#ifdef DEBUG_SCRIPT
            printf( "%*s  I %ld(%ld)\n", depth, "", ( B - wave->Babs ) + y, ( A - wave->Aabs ) + x + 1 );
#endif
        }

#ifdef DEBUG_SCRIPT
        else
            printf( "%*s  %ld S %ld\n", depth, "", ( wave->Aabs - A ) + x, ( B - wave->Babs ) + y );
#endif
    }

    return ( D );
}

int Compute_Alignment( Alignment* align, Work_Data* ework, int task, int tspace )
{
    _Work_Data* work = (_Work_Data*)ework;
    Trace_Waves wave;

    int L, D;
    int asub, bsub;
    char *aseq, *bseq;
    Path* path;
    int* trace;
    uint16* strace;

    path = align->path;
    asub = path->aepos - path->abpos;
    bsub = path->bepos - path->bbpos;
    aseq = align->aseq + path->abpos;
    bseq = align->bseq + path->bbpos;

    L = 0;
    if ( task != DIFF_ONLY )
    {
        if ( task == DIFF_TRACE || task == PLUS_TRACE )
            L = 2 * ( ( ( path->aepos + ( tspace - 1 ) ) / tspace - path->abpos / tspace ) + 1 ) * sizeof( uint16 );
        else if ( asub < bsub )
            L = bsub * sizeof( int );
        else
            L = asub * sizeof( int );
        if ( L > work->alnmax )
            if ( enlarge_alnpts( work, L ) )
                EXIT( 1 );
    }

    trace  = ( (int*)work->alnpts );
    strace = ( (uint16*)work->alnpts );

    if ( asub > bsub )
        D = ( 4 * asub + 6 ) * sizeof( int );
    else
        D = ( 4 * bsub + 6 ) * sizeof( int );
    if ( D > work->vecmax )
        if ( enlarge_vector( work, D ) )
            EXIT( 1 );

    if ( asub > bsub )
    {
        wave.VF = ( (int*)work->vector ) + ( asub + 1 );
        wave.VB = wave.VF + ( 2 * asub + 3 );
    }
    else
    {
        wave.VF = ( (int*)work->vector ) + ( bsub + 1 );
        wave.VB = wave.VF + ( 2 * bsub + 3 );
    }

    wave.Aabs = align->aseq;
    wave.Babs = align->bseq;

    if ( task == DIFF_ONLY )
    {
        wave.mida = -1;
        if ( asub <= 0 )
            path->diffs = bsub;
        else if ( bsub <= 0 )
            path->diffs = asub;
        else
            path->diffs = split_nd( aseq, asub, bseq, bsub, &wave, &wave.mida, &wave.midb );
        path->trace = NULL;
        path->tlen  = -1;
        return ( 0 );
    }

    else if ( task < DIFF_ONLY && wave.mida >= 0 )
    {
        int x = wave.mida;
        int y = wave.midb;

        if ( task == PLUS_ALIGN )
        {
            wave.Stop = trace;
            dandc_nd( aseq, x, bseq, y, &wave );
            dandc_nd( aseq + x, asub - x, bseq + y, bsub - y, &wave );
            path->tlen = wave.Stop - trace;
        }
        else
        {
            int i, n;

            wave.Trace = strace - 2 * ( path->abpos / tspace );
            n          = L / sizeof( uint16 );
            for ( i = 0; i < n; i++ )
                strace[ i ] = 0;

            trace_nd( aseq, x, bseq, y, &wave, tspace );
            trace_nd( aseq + x, asub - x, bseq + y, bsub - y, &wave, tspace );

            if ( strace[ n - 1 ] != 0 ) //  Last element is to capture all inserts on TP boundary
            {
                strace[ n - 3 ] += strace[ n - 1 ];
                strace[ n - 4 ] += strace[ n - 2 ];
            }
            path->tlen = n - 2;

#ifdef DEBUG_SCRIPT
            printf( "  Trace:\n" );
            for ( i = 0; i < path->tlen; i += 2 )
                printf( "    %3d  %3d\n", strace[ i ], strace[ i + 1 ] );
            fflush( stdout );
#endif
        }
    }

    else
    {
        if ( task == DIFF_ALIGN )
        {
            wave.Stop   = trace;
            path->diffs = dandc_nd( aseq, asub, bseq, bsub, &wave );
            path->tlen  = wave.Stop - trace;
        }
        else
        {
            int i, n;

            wave.Trace = strace - 2 * ( path->abpos / tspace );
            n          = L / sizeof( uint16 );
            for ( i = 0; i < n; i++ )
                strace[ i ] = 0;
            path->diffs = trace_nd( aseq, asub, bseq, bsub, &wave, tspace );

            if ( strace[ n - 1 ] != 0 ) //  Last element is to capture all inserts on TP boundary
            {
                strace[ n - 3 ] += strace[ n - 1 ];
                strace[ n - 4 ] += strace[ n - 2 ];
            }
            path->tlen = n - 2;

#ifdef DEBUG_SCRIPT
            printf( "  Trace:\n" );
            for ( i = 0; i < path->tlen; i += 2 )
                printf( "    %3d  %3d\n", strace[ i ], strace[ i + 1 ] );
            fflush( stdout );
#endif
        }
    }

    path->trace = trace;
    return ( 0 );
}
