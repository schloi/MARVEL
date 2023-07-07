
//  A complete threaded code for the filter

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "align.h"
#include "db/DB.h"
#include "filter.h"
#include "radix.h"
#include "types.h"

#if 1
#define likely( cond ) __builtin_expect( ( cond ), 1 )
#define unlikely( cond ) __builtin_expect( ( cond ), 0 )
#else
#define likely( cond ) ( cond )
#define unlikely( cond ) ( cond )
#endif

#undef PROFILE //  WHen running sensitivity trials, compute histogram of
// #define MAXHIT   1000  //    false & true positive hit scores

//  K-mer selection strategy control:
//       MODIFER off: Select minimizers in window of size WINDOW
//       MODIFER on : Select modimizers mod MODULUS < MODTHR (best)

#define MODIFER
// #define WINDOW   6

#define MODULUS 101
#define MODTHR 28

//  Algorithm constants & global data types

#define THREAD pthread_t

// #define MAX_CODE_16  0xffffu
// #define MAX_CODE_32  0xffffffffu
#define MAX_CODE_64 0xffffffffffffffffllu

#define SIGN_BIT 0x1u

/*******************************************************************************************
 *
 *  PARAMETER SETUP
 *
 ********************************************************************************************/

int Kmer;
static int Koff;     //  Kmer-2
static int Kshift;   //  2*Kmer
static uint64 Kmask; //  2^Kshift - 1

static uint64 LFmask; //  4^floor(Kmer/2)-1
static uint64 HFmask; //  Kmask - LFmask;
static uint64 LRmask; //  4^ceil(Kmer/2)-1
static uint64 HRmask; //  Kmask - LRmask;

int Hitmin;
int Binshift;
static int Suppress;
static int TooFrequent; //  (Suppress != 0) ? Suppress : INT32_MAX

static int NTHREADS; //  # of threads to use
int MR_tspace;

void Set_Filter_Params( int kmer, uint binshift, int suppress, int hitmin, int nthread, int traceSpace )
{
    if ( kmer > 32 )
    {
        fprintf( stderr, "%s: Kmer length must be <= 32\n", Prog_Name );
        exit( 1 );
    }
    MR_tspace = traceSpace;
    Kmer      = kmer;
    Koff      = kmer - 2;
    Binshift  = binshift;
    Suppress  = suppress;
    Hitmin    = hitmin;

    Kmer = kmer;
    if ( Kmer >= 32 )
    {
        Kshift = 64;
        Kmask  = MAX_CODE_64;
    }
    else
    {
        Kshift = 2 * Kmer;
        Kmask  = ( 0x1llu << Kshift ) - 1;
    }

    LFmask = ( 0x1llu << ( Kshift / 2 ) ) - 1;
    HFmask = Kmask - LFmask;

    LRmask = ( 0x1llu << ( ( Kshift + 1 ) / 2 ) ) - 1;
    HRmask = Kmask - LRmask;

    if ( Suppress == 0 )
        TooFrequent = INT32_MAX;
    else
        TooFrequent = Suppress;

    NTHREADS = nthread;
}

/*******************************************************************************************
 *
 *  INDEX BUILD
 *
 ********************************************************************************************/

typedef struct
{
    int beg;
    int end;
    int fill;
    KmerPos* FR_src;
    KmerPos* FR_trg;
    DAZZ_DB* TA_block;
    DAZZ_TRACK* TA_track;
    uint64_t Cumber[ 4 ];

} Tuple_Arg;

//  for reads [beg,end) computing how many k-tuples are not masked

static void* mask_thread( void* arg )
{
    Tuple_Arg* data  = (Tuple_Arg*)arg;
    DAZZ_READ* reads = data->TA_block->reads;
    int km1          = Kmer - 1;
    int beg, end, idx;
    int64_t a, b, f;
    int i, p, q;

#ifndef MODIFER
    uint64 min1[ WINDOW ], min2[ WINDOW ];
    int w, ny, m1, m2;
#endif
    int x;
    uint64_t c, u;
    uint64_t d, v;
    char* s;

    beg = data->beg;
    end = data->end;
    idx = 0;

    s = ( (char*)( data->TA_block->bases ) ) + data->TA_block->reads[ beg ].boff;
    if ( data->TA_track != NULL )
    {
        int64* anno1 = ( (int64*)( data->TA_track->anno ) ) + 1;
        int* point   = (int*)( data->TA_track->data );

        q = 0;
        f = anno1[ beg - 1 ];
        for ( i = beg; i < end; i++ )
        {
            b = f;
            f = anno1[ i ];
            for ( a = b; a <= f; a += 2 )
            {
                if ( a == b )
                    p = 0;
                else
                    p = point[ a - 1 ];
                if ( a == f )
                    q = reads[ i ].rlen;
                else
                    q = point[ a ];
                if ( q - p > km1 )
                {
                    c = u = 0;
                    for ( x = 1; x < Kmer; x++ )
                    {
                        c = ( c << 2 ) | s[ p ];
                        u = ( u >> 2 ) | data->Cumber[ (int)s[ p++ ] ];
                    }
#ifndef MODIFER
                    ny = 1;
                    w  = 0;
#endif
                    while ( p < q )
                    {
                        x = s[ p ];

                        d = ( c & HFmask );
                        c = ( ( c << 2 ) | x ) & Kmask;
                        d = d | ( c & LFmask );

                        v = ( u & LRmask );
                        u = ( u >> 2 ) | data->Cumber[ x ];
                        v = v | ( u & HRmask );

#ifdef MODIFER
                        if ( u < c )
                        {
                            if ( u % MODULUS < MODTHR )
                                idx += 1;
                        }
                        else
                        {
                            if ( c % MODULUS < MODTHR )
                                idx += 1;
                        }

                        if ( v < d )
                        {
                            if ( v % MODULUS < MODTHR )
                                idx += 1;
                        }
                        else
                        {
                            if ( d % MODULUS < MODTHR )
                                idx += 1;
                        }
#else
                        if ( u < c )
                            min1[ w ] = u;
                        else
                            min1[ w ] = c;
                        if ( v < d )
                            min2[ w ] = v;
                        else
                            min2[ w ] = d;

                        if ( ny )
                        {
                            w += 1;
                            if ( w < WINDOW )
                                continue;
                            m1 = m2 = w = w - 1;
                            ny          = 0;
                        }

                        if ( w == m1 )
                        {
                            m1 = 0;
                            for ( x = 1; x < WINDOW; x++ )
                                if ( min1[ x ] < min1[ m1 ] )
                                    m1 = x;
                            idx += 1;
                        }
                        else if ( min1[ w ] < min1[ m1 ] )
                        {
                            m1 = w;
                            idx += 1;
                        }

                        if ( w == m2 )
                        {
                            m2 = 0;
                            for ( x = 1; x < WINDOW; x++ )
                                if ( min2[ x ] < min2[ m2 ] )
                                    m2 = x;
                            idx += 1;
                        }
                        else if ( min2[ w ] < min2[ m2 ] )
                        {
                            m2 = w;
                            idx += 1;
                        }

                        w += 1;
                        if ( w == WINDOW )
                            w = 0;
#endif
                        p += 1;
                    }
                }
            }
            s += ( q + 1 );
        }
    }
    else
    {
        // aid compiler in optimizing access to Cumber
        const uint64 _Cumber[4] = { data->Cumber[0], data->Cumber[1], data->Cumber[2], data->Cumber[3] };

        for ( i = beg; i < end; i++ )
        {
            q = reads[ i ].rlen;
            c = u = 0;
            for ( p = 0; p < km1; p++ )
            {
                x = s[ p ];
                c = ( c << 2 ) | x;
                u = ( u >> 2 ) | _Cumber[ x ];
            }
#ifndef MODIFER
            ny = 1;
            w  = 0;
#endif
            for ( p = km1; p < q; p++ )
            {
                x = s[ p ];

                d = ( c & HFmask );
                c = ( ( c << 2 ) | x ) & Kmask;
                d = d | ( c & LFmask );

                v = ( u & LRmask );
                u = ( u >> 2 ) | _Cumber[ x ];
                v = v | ( u & HRmask );

#ifdef MODIFER
                if ( u < c )
                {
                    if ( unlikely(u % MODULUS < MODTHR ) )
                        idx += 1;
                }
                else
                {
                    if ( unlikely(c % MODULUS < MODTHR ) )
                        idx += 1;
                }

                if ( v < d )
                {
                    if ( unlikely(v % MODULUS < MODTHR ) )
                        idx += 1;
                }
                else
                {
                    if ( unlikely(d % MODULUS < MODTHR ) )
                        idx += 1;
                }
#else
                if ( u < c )
                    min1[ w ] = u;
                else
                    min1[ w ] = c;
                if ( v < d )
                    min2[ w ] = v;
                else
                    min2[ w ] = d;

                if ( ny )
                {
                    w += 1;
                    if ( w < WINDOW )
                        continue;
                    m1 = m2 = w = w - 1;
                    ny = 0;
                }

                if ( w == m1 )
                {
                    m1 = 0;
                    for ( x = 1; x < WINDOW; x++ )
                        if ( min1[ x ] < min1[ m1 ] )
                            m1 = x;
                    idx += 1;
                }
                else if ( min1[ w ] < min1[ m1 ] )
                {
                    m1 = w;
                    idx += 1;
                }

                if ( w == m2 )
                {
                    m2 = 0;
                    for ( x = 1; x < WINDOW; x++ )
                        if ( min2[ x ] < min2[ m2 ] )
                            m2 = x;
                    idx += 1;
                }
                else if ( min2[ w ] < min2[ m2 ] )
                {
                    m2 = w;
                    idx += 1;
                }

                w += 1;
                if ( w == WINDOW )
                    w = 0;
#endif
            }
            s += ( q + 1 );
        }
    }

    data->fill = idx;
    return ( NULL );
}

// for reads [beg,end) place their k-tuples in list starting at index idx

static void* tuple_thread( void* arg )
{
    Tuple_Arg* data  = (Tuple_Arg*)arg;
    DAZZ_READ* reads = data->TA_block->reads;
    int km1          = Kmer - 1;
    KmerPos* list    = data->FR_src;
    int beg, end, idx;
    int64 a, b, f;
    int i, p, q, x, r;
    uint64 c, u;
    uint64 d, v;
    char* s;

#ifndef MODIFER
    uint64 min1[ WINDOW ], min2[ WINDOW ];
    int sgn1[ WINDOW ], sgn2[ WINDOW ];
    int pos[ WINDOW ];
    int w, ny, m1, m2;
#endif

    beg = data->beg;
    end = data->end;
    idx = data->fill;

    s = ( (char*)( data->TA_block->bases ) ) + data->TA_block->reads[ beg ].boff;
    if ( data->TA_track != NULL )
    {
        int64* anno1 = ( (int64*)( data->TA_track->anno ) ) + 1;
        int* point   = (int*)( data->TA_track->data );

        q = 0;
        f = anno1[ beg - 1 ];
        for ( i = beg; i < end; i++ )
        {
            r = ( i << 1 );
            b = f;
            f = anno1[ i ];
            for ( a = b; a <= f; a += 2 )
            {
                if ( a == b )
                    p = 0;
                else
                    p = point[ a - 1 ];
                if ( a == f )
                    q = reads[ i ].rlen;
                else
                    q = point[ a ];
                if ( q - p > km1 )
                {
                    c = 0;
                    u = 0;
                    for ( x = 1; x < Kmer; x++ )
                    {
                        c = ( ( c << 2 ) | s[ p ] );
                        u = ( u >> 2 ) | data->Cumber[ (int)s[ p ] ];
                        p += 1;
                    }
#ifndef MODIFER
                    ny = 1;
                    w  = 0;
#endif
                    while ( p < q )
                    {
                        x = s[ p ];

                        d = ( c & HFmask );
                        c = ( ( c << 2 ) | x ) & Kmask;
                        d = d | ( c & LFmask );

                        v = ( u & LRmask );
                        u = ( u >> 2 ) | data->Cumber[ x ];
                        v = v | ( u & HRmask );
#ifdef MODIFER
                        if ( u < c )
                        {
                            if ( u % MODULUS < MODTHR )
                            {
                                list[ idx ].code = u;
                                list[ idx ].read = r | SIGN_BIT;
                                list[ idx ].rpos = p;
                                idx += 1;
                            }
                        }
                        else
                        {
                            if ( c % MODULUS < MODTHR )
                            {
                                list[ idx ].code = c;
                                list[ idx ].read = r;
                                list[ idx ].rpos = p;
                                idx += 1;
                            }
                        }

                        if ( v < d )
                        {
                            if ( v % MODULUS < MODTHR )
                            {
                                list[ idx ].code = v;
                                list[ idx ].read = r | SIGN_BIT;
                                list[ idx ].rpos = p;
                                idx += 1;
                            }
                        }
                        else
                        {
                            if ( d % MODULUS < MODTHR )
                            {
                                list[ idx ].code = d;
                                list[ idx ].read = r;
                                list[ idx ].rpos = p;
                                idx += 1;
                            }
                        }
#else
                        if ( u < c )
                        {
                            min1[ w ] = u;
                            sgn1[ w ] = 0x1;
                        }
                        else
                        {
                            min1[ w ] = c;
                            sgn1[ w ] = 0x0;
                        }
                        if ( v < d )
                        {
                            min2[ w ] = v;
                            sgn2[ w ] = 0x1;
                        }
                        else
                        {
                            min2[ w ] = d;
                            sgn2[ w ] = 0x0;
                        }
                        pos[ w ] = p;

                        if ( ny )
                        {
                            w += 1;
                            if ( w < WINDOW )
                                continue;
                            m1 = m2 = w = w - 1;
                            ny = 0;
                        }

                        if ( w == m1 )
                        {
                            m1 = 0;
                            for ( x = 1; x < WINDOW; x++ )
                                if ( min1[ x ] < min1[ m1 ] )
                                    m1 = x;
                            list[ idx ].read = r | sgn1[ m1 ];
                            list[ idx ].code = min1[ m1 ];
                            list[ idx ].rpos = pos[ m1 ];
                            idx += 1;
                        }
                        else if ( min1[ w ] < min1[ m1 ] )
                        {
                            m1 = w;
                            list[ idx ].read = r | sgn1[ m1 ];
                            list[ idx ].code = min1[ m1 ];
                            list[ idx ].rpos = pos[ m1 ];
                            idx += 1;
                        }

                        if ( w == m2 )
                        {
                            m2 = 0;
                            for ( x = 1; x < WINDOW; x++ )
                                if ( min2[ x ] < min2[ m2 ] )
                                    m2 = x;
                            list[ idx ].read = r | sgn2[ m2 ];
                            list[ idx ].code = min2[ m2 ];
                            list[ idx ].rpos = pos[ m2 ];
                            idx += 1;
                        }
                        else if ( min2[ w ] < min2[ m2 ] )
                        {
                            m2 = w;
                            list[ idx ].read = r | sgn2[ m2 ];
                            list[ idx ].code = min2[ m2 ];
                            list[ idx ].rpos = pos[ m2 ];
                            idx += 1;
                        }

                        w += 1;
                        if ( w == WINDOW )
                            w = 0;
#endif
                        p += 1;
                    }
                }
            }
            s += ( q + 1 );
        }
    }

    else
    {
        // aid compiler in optimizing access to Cumber
        const uint64 _Cumber[4] = { data->Cumber[0], data->Cumber[1], data->Cumber[2], data->Cumber[3] };

        for ( i = beg; i < end; i++ )
        {
            q = reads[ i ].rlen;
            r = ( i << 1 );
            c = 0;
            u = 0;
            for ( p = 0; p < km1; p++ )
            {
                x = s[ p ];
                c = ( c << 2 ) | x;
                u = ( u >> 2 ) | _Cumber[ x ];
            }
#ifndef MODIFER
            ny = 1;
            w  = 0;
#endif
            for ( p = km1; p < q; p++ )
            {
                x = s[ p ];

                d = ( c & HFmask );
                c = ( ( c << 2 ) | x ) & Kmask;
                d = d | ( c & LFmask );

                v = ( u & LRmask );
                u = ( u >> 2 ) | _Cumber[ x ];
                v = v | ( u & HRmask );

#ifdef MODIFER
                if ( u < c )
                {
                    if ( unlikely(u % MODULUS < MODTHR ))
                    {
                        list[ idx ].code = u;
                        list[ idx ].read = r | SIGN_BIT;
                        list[ idx ].rpos = p;
                        idx += 1;
                    }
                }
                else
                {
                    if ( unlikely(c % MODULUS < MODTHR ))
                    {
                        list[ idx ].code = c;
                        list[ idx ].read = r;
                        list[ idx ].rpos = p;
                        idx += 1;
                    }
                }

                if ( v < d )
                {
                    if ( unlikely(v % MODULUS < MODTHR ))
                    {
                        list[ idx ].code = v;
                        list[ idx ].read = r | SIGN_BIT;
                        list[ idx ].rpos = p;
                        idx += 1;
                    }
                }
                else
                {
                    if ( unlikely(d % MODULUS < MODTHR ))
                    {
                        list[ idx ].code = d;
                        list[ idx ].read = r;
                        list[ idx ].rpos = p;
                        idx += 1;
                    }
                }
#else
                if ( u < c )
                {
                    min1[ w ] = u;
                    sgn1[ w ] = 0x1;
                }
                else
                {
                    min1[ w ] = c;
                    sgn1[ w ] = 0x0;
                }
                if ( v < d )
                {
                    min2[ w ] = v;
                    sgn2[ w ] = 0x1;
                }
                else
                {
                    min2[ w ] = d;
                    sgn2[ w ] = 0x0;
                }
                pos[ w ] = p;

                if ( ny )
                {
                    w += 1;
                    if ( w < WINDOW )
                        continue;
                    m1 = m2 = w = w - 1;
                    ny = 0;
                }

                if ( w == m1 )
                {
                    m1 = 0;
                    for ( x = 1; x < WINDOW; x++ )
                        if ( min1[ x ] < min1[ m1 ] )
                            m1 = x;
                    list[ idx ].read = r | sgn1[ m1 ];
                    list[ idx ].code = min1[ m1 ];
                    list[ idx ].rpos = pos[ m1 ];
                    idx += 1;
                }
                else if ( min1[ w ] < min1[ m1 ] )
                {
                    m1 = w;
                    list[ idx ].read = r | sgn1[ m1 ];
                    list[ idx ].code = min1[ m1 ];
                    list[ idx ].rpos = pos[ m1 ];
                    idx += 1;
                }

                if ( w == m2 )
                {
                    m2 = 0;
                    for ( x = 1; x < WINDOW; x++ )
                        if ( min2[ x ] < min2[ m2 ] )
                            m2 = x;
                    list[ idx ].read = r | sgn2[ m2 ];
                    list[ idx ].code = min2[ m2 ];
                    list[ idx ].rpos = pos[ m2 ];
                    idx += 1;
                }
                else if ( min2[ w ] < min2[ m2 ] )
                {
                    m2 = w;
                    list[ idx ].read = r | sgn2[ m2 ];
                    list[ idx ].code = min2[ m2 ];
                    list[ idx ].rpos = pos[ m2 ];
                    idx += 1;
                }

                w += 1;
                if ( w == WINDOW )
                    w = 0;
#endif
            }
            s += ( q + 1 );
        }
    }

    return ( NULL );
}

static void* compsize_thread( void* arg )
{
    Tuple_Arg* data = (Tuple_Arg*)arg;
    const int end         = data->end;
    const KmerPos* const src    = data->FR_src;
    int n, i, c, p;
    uint64 h, g;

    i = data->beg;
    h = src[ i ].code;
    n = 0;
    while ( i < end )
    {
        p = i++;
        while ( 1 )
        {
            g = src[ i ].code;
            if ( g != h )
                break;
            i += 1;
        }
        if ( ( c = ( i - p ) ) < TooFrequent )
            n += c;
        h = g;
    }

    data->fill = n;
    return ( NULL );
}

static void* compress_thread( void* arg )
{
    Tuple_Arg* data = (Tuple_Arg*)arg;
    const int end         = data->end;
    const KmerPos* const src    = data->FR_src;
    KmerPos* trg    = data->FR_trg;
    int n, i, p;
    uint64 h, g;

    i = data->beg;
    h = src[ i ].code;
    n = data->fill;
    while ( i < end )
    {
        p = i++;
        while ( 1 )
        {
            g = src[ i ].code;
            if ( g != h )
                break;
            i += 1;
        }
        if ( i - p < TooFrequent )
        {
            while ( p < i )
                trg[ n++ ] = src[ p++ ];
        }
        h = g;
    }

    return ( NULL );
}

KmerPos * Sort_Kmers( DAZZ_DB* block, int* len )
{
    THREAD threads[ NTHREADS ];
    Tuple_Arg parmt[ NTHREADS ];

    KmerPos *src, *trg, *rez;
    int kmers, nreads;

    nreads = block->nreads;

    if ( block->reads[ nreads ].boff > 0x7fffffffll )
    {
        fprintf( stderr, "%s: Fatal error, DB blocks are greater than 2Gbp!\n", Prog_Name );
        exit( 1 );
    }

    if ( nreads <= 0 )
        goto no_mers;

    for ( int i = 0; i < NTHREADS; i++ )
    {

        parmt[ i ].TA_block = block;
        parmt[ i ].TA_track = block->tracks;

        parmt[ i ].Cumber[ 0 ] = ( 0x3llu << ( Kshift - 2 ) );
        parmt[ i ].Cumber[ 1 ] = ( 0x2llu << ( Kshift - 2 ) );
        parmt[ i ].Cumber[ 2 ] = ( 0x1llu << ( Kshift - 2 ) );
        parmt[ i ].Cumber[ 3 ] = ( 0x0llu << ( Kshift - 2 ) );
    }

    //  Determine how many k-tuples will be listed for each thread
    //    and use that to set up index drop points


    INIT_TIMING
    START_TIMING

    {
        int i, x, z;

        parmt[ 0 ].beg = 0;
        for ( i = 1; i < NTHREADS; i++ )
            parmt[ i ].beg = parmt[ i - 1 ].end = ( ( (int64)nreads ) * i ) / NTHREADS;
        parmt[ NTHREADS - 1 ].end = nreads;

        for ( i = 0; i < NTHREADS; i++ )
            pthread_create( threads + i, NULL, mask_thread, parmt + i );
        for ( i = 0; i < NTHREADS; i++ )
            pthread_join( threads[ i ], NULL );

        x = 0;
        for ( i = 0; i < NTHREADS; i++ )
        {
            z               = parmt[ i ].fill;
            parmt[ i ].fill = x;
            x += z;
        }
        kmers = x;

        if ( kmers <= 0 )
            goto no_mers;
    }

    END_TIMING("mask_thread")

    //  Allocate k-mer sorting arrays now that # of kmers is known

    if ( ( ( Kshift - 1 ) / 8 + ( TooFrequent < INT32_MAX ) ) & 0x1 )
    {
        src = (KmerPos*)Malloc( sizeof( KmerPos ) * ( kmers + 2 ), "Allocating Sort_Kmers vectors" );
        trg = (KmerPos*)Malloc( sizeof( KmerPos ) * ( kmers + 2 ), "Allocating Sort_Kmers vectors" );
    }
    else
    {
        trg = (KmerPos*)Malloc( sizeof( KmerPos ) * ( kmers + 2 ), "Allocating Sort_Kmers vectors" );
        src = (KmerPos*)Malloc( sizeof( KmerPos ) * ( kmers + 2 ), "Allocating Sort_Kmers vectors" );
    }
    if ( src == NULL || trg == NULL )
        exit( 1 );

    if ( VERBOSE )
    {
        printf( "\n   Kmer count = " );
        Print_Number( (int64)kmers, 0, stdout );
        printf( "\n   Using %.2fGb of space\n", ( 1. * kmers ) / ( 0x20000000 / sizeof( KmerPos ) ) );
        fflush( stdout );
    }

    //  Build the k-mer list

    START_TIMING

    {
        int i;
        for ( i = 0; i < NTHREADS; i++ )
        {
            parmt[ i ].FR_src = src;
        }

        for ( i = 0; i < NTHREADS; i++ )
            pthread_create( threads + i, NULL, tuple_thread, parmt + i );
        for ( i = 0; i < NTHREADS; i++ )
            pthread_join( threads[ i ], NULL );
    }

    END_TIMING("tuple_thread")

    //  Sort the k-mer list

    START_TIMING

    {
        int i;
        int mersort[ 11 ];

#if __ORDER_LITTLE_ENDIAN__ == __BYTE_ORDER__
        for ( i = 0; i < ( Kmer - 1 ) / 4 + 1; i++ )
            mersort[ i ] = 8 + i;
#else
        for ( i = 0; i < ( Kmer - 1 ) / 4 + 1; i++ )
            mersort[ i ] = 17 - i;
#endif
        mersort[ i ] = -1;

        rez = (KmerPos*)Radix_Sort( kmers, src, trg, mersort );
    }

    END_TIMING("radix_sort")

    //  Compress frequent tuples if requested

    START_TIMING

    if ( TooFrequent < INT32_MAX && kmers > 0 )
    {
        int i, x, z;
        uint64 h;

        parmt[ 0 ].beg = 0;
        for ( i = 1; i < NTHREADS; i++ )
        {
            x = ( ( (int64)i ) * kmers ) / NTHREADS;
            h = rez[ x - 1 ].code;
            while ( rez[ x ].code == h )
                x += 1;
            parmt[ i - 1 ].end = parmt[ i ].beg = x;
        }
        parmt[ NTHREADS - 1 ].end = kmers;

        if ( rez[ kmers - 1 ].code == MAX_CODE_64 )
            rez[ kmers ].code = 0;
        else
            rez[ kmers ].code = MAX_CODE_64;

        if ( src == rez )
        {
            for ( i = 0; i < NTHREADS; i++ )
            {
                parmt[ i ].FR_src = src;
                parmt[ i ].FR_trg = rez = trg;
            }
        }
        else
        {
            for ( i = 0; i < NTHREADS; i++ )
            {
                parmt[ i ].FR_src = trg;
                parmt[ i ].FR_trg = rez = src;
            }
        }

        for ( i = 0; i < NTHREADS; i++ )
            pthread_create( threads + i, NULL, compsize_thread, parmt + i );

        for ( i = 0; i < NTHREADS; i++ )
            pthread_join( threads[ i ], NULL );

        x = 0;
        for ( i = 0; i < NTHREADS; i++ )
        {
            z               = parmt[ i ].fill;
            parmt[ i ].fill = x;
            x += z;
        }
        kmers = x;

        for ( i = 0; i < NTHREADS; i++ )
            pthread_create( threads + i, NULL, compress_thread, parmt + i );

        for ( i = 0; i < NTHREADS; i++ )
            pthread_join( threads[ i ], NULL );
    }

    rez[ kmers ].code     = MAX_CODE_64;
    rez[ kmers + 1 ].code = 0;

    if ( src != rez )
        free( src );
    else
        free( trg );

    END_TIMING("compsize + compress_thread")

#ifdef TEST_KSORT
    {
        int i;

        printf( "\nKMER SORT:\n" );
        for ( i = 0 /*100000000*/; i < 100000000 + HOW_MANY && i < kmers; i++ )
        {
            KmerPos* c = rez + i;
            printf( " %9d:  %6d%c / %6d / %016llx\n", i, c->read >> 1, ( c->read & 0x1 ) ? 'c' : 'n', c->rpos, c->code );
        }
        fflush( stdout );
    }
#endif

#ifdef HISTOGRAM_KSORT
    {
        int hist[ 100 ];
        uint64 ca;
        int i, j;

        for ( i = 0; i < 100; i++ )
            hist[ i ] = 0;

        i = 0;
        while ( i < kmers )
        {
            ca = rez[ i ].code;
            j  = i++;
            while ( rez[ i ].code == ca )
                i += 1;
            if ( i - j >= 100 )
                hist[ 99 ] += 1;
            else
                hist[ i - j ] += 1;
        }

        for ( i = 99; i >= 0; i-- )
            printf( " %2d: %6d\n", i, hist[ i ] );
    }
#endif

    if ( VERBOSE )
    {
        if ( TooFrequent < INT32_MAX )
        {
            printf( "   Revised kmer count = " );
            Print_Number( (int64)kmers, 0, stdout );
            printf( "\n" );
        }
        printf( "   Index occupies %.2fGb\n", ( 1. * kmers ) / ( 0x40000000 / sizeof( KmerPos ) ) );
        fflush( stdout );
    }

    if ( kmers <= 0 )
    {
        free( rez );
        goto no_mers;
    }

    if ( kmers > ( int64 )( MEM_LIMIT / ( 4 * sizeof( KmerPos ) ) ) )
    {
        fprintf( stderr, "Warning: Block size too big, index occupies more than 1/4 of" );
        if ( MEM_LIMIT == MEM_PHYSICAL )
            fprintf( stderr, " physical memory (%.1fGb)\n", ( 1. * MEM_LIMIT ) / 0x40000000ll );
        else
            fprintf( stderr, " desired memory allocation (%.1fGb)\n", ( 1. * MEM_LIMIT ) / 0x40000000ll );
        fflush( stderr );
    }

    *len = kmers;
    return ( rez );

no_mers:
    *len = 0;
    return ( NULL );
}

/*******************************************************************************************
 *
 *  FILTER MATCH
 *
 ********************************************************************************************/

int find_tuple( uint64 x, KmerPos* a, int n )
{
    int l, r, m;

    // smallest k s.t. a[k].code >= x (or n if does not exist)

    l = 0;
    r = n;
    while ( l < r )
    {
        m = ( ( l + r ) >> 1 );
        if ( a[ m ].code < x )
            l = m + 1;
        else
            r = m;
    }
    return ( l );
}

//  Determine what *will* be the size of the merged list and histogram of sizes for given cutoffs

void* count_thread( void* arg )
{
    Merge_Arg* data = (Merge_Arg*)arg;
    KmerPos* asort  = data->MG_alist;
    KmerPos* bsort  = data->MG_blist;
    int64* gram     = data->hitgram;
    int64 nhits     = 0;
    int aend        = data->aend;

    int64 ct;
    int ia, ja;
    uint64 ca, da;

    ia = data->abeg;
    ca = asort[ ia ].code;
    if ( data->MG_self )
    {
        uint32 ar;
        int ka;

        while ( 1 )
        {
            ja = ka = ia++;
            ct      = 0;
            if ( IDENTITY )
                while ( 1 )
                {
                    da = asort[ ia ].code;
                    if ( da != ca )
                        break;
                    ct += ( ia - ja );
                    ia += 1;
                }
            else
                while ( 1 )
                {
                    da = asort[ ia ].code;
                    if ( da != ca )
                        break;
                    ar = ( asort[ ia ].read & ~0x1u );
                    while ( ka < ia && asort[ ka ].read < ar )
                        ka += 1;
                    ct += ( ka - ja );
                    ia += 1;
                }

            ca = da;
            if ( ia > aend )
            {
                if ( ja >= aend )
                    break;
                ia = aend;
                ca = asort[ ia ].code;
                ct -= ( ka - ja );
            }

            nhits += ct;
            if ( ct < MAXGRAM )
                gram[ ct ] += 1;
        }
    }
    else
    {
        int ib, jb;
        uint64 cb;

        ib = data->bbeg;
        cb = bsort[ ib ].code;
        while ( 1 )
        {
            ja = ia++;
            while ( 1 )
            {
                da = asort[ ia ].code;
                if ( da != ca )
                    break;
                ia += 1;
            }

            if ( ia > aend )
            {
                if ( ja >= aend )
                    break;
                ia = aend;
                da = asort[ ia ].code;
            }

            while ( cb < ca )
            {
                ib += 1;
                cb = bsort[ ib ].code;
            }
            if ( cb != ca )
            {
                ca = da;
                continue;
            }

            jb = ib++;
            while ( 1 )
            {
                cb = bsort[ ib ].code;
                if ( cb != ca )
                    break;
                ib += 1;
            }
            ca = da;

            ct = ( ( int64 )( ia - ja ) ) * ( ib - jb );
            nhits += ct;
            if ( ct < MAXGRAM )
                gram[ ct ] += 1;
        }
    }

    data->nhits = nhits;

    return ( NULL );
}

//  Produce the merged list now that the list has been allocated and
//    the appropriate cutoff determined.

void* merge_thread( void* arg )
{
    Merge_Arg* data  = (Merge_Arg*)arg;
    const KmerPos* const asort   = data->MG_alist;
    const KmerPos* const bsort   = data->MG_blist;
    const DAZZ_READ* const reads = data->MG_bblock->reads;
    SeedPair* hits   = data->MG_hits;
    int64 nhits      = data->nhits;
    int aend         = data->aend;
    int limit        = data->limit;

    int64 ct;
    int ia, ja;
    uint64 ca, da;
    int nread = data->MG_ablock->nreads;

    ia = data->abeg;
    ca = asort[ ia ].code;
    if ( data->MG_self )
    {
        uint32 ar, br;
        uint32 ap, bp;
        uint32 as, bs;
        int a, ka;

        while ( 1 )
        {
            ja = ka = ia++;
            ct      = 0;
            if ( IDENTITY )
                while ( 1 )
                {
                    da = asort[ ia ].code;
                    if ( da != ca )
                        break;
                    ct += ( ia - ja );
                    ia += 1;
                }
            else
                while ( 1 )
                {
                    da = asort[ ia ].code;
                    if ( da != ca )
                        break;
                    ar = ( asort[ ia ].read & ~0x1u );
                    while ( ka < ia && asort[ ka ].read < ar )
                        ka += 1;
                    ct += ( ka - ja );
                    ia += 1;
                }

            ca = da;
            if ( ia > aend )
            {
                if ( ja >= aend )
                    break;
                ia = aend;
                ca = asort[ ia ].code;
                ct -= ( ka - ja );
            }

            if ( ct >= limit )
                continue;

            if ( IDENTITY )
                for ( ka = ja + 1; ka < ia; ka++ )
                {
                    ar = asort[ ka ].read;
                    as = ( ar & SIGN_BIT );
                    ar >>= 1;
                    ap = asort[ ka ].rpos;
                    for ( a = ja; a < ka; a++ )
                    {
                        br = asort[ a ].read;
                        bs = ( br & SIGN_BIT );
                        br >>= 1;
                        bp = asort[ a ].rpos;
                        if ( bs == as )
                            hits[ nhits ].aread = ar;
                        else
                        {
                            bp                  = ( reads[ br ].rlen - bp ) + Koff;
                            hits[ nhits ].aread = ar + nread;
                        }
                        hits[ nhits ].bread = br;
                        hits[ nhits ].apos  = ap;
                        hits[ nhits ].diag  = ap - bp;
                        nhits += 1;
                    }
                }
            else
                for ( ka = ja + 1; ka < ia; ka++ )
                {
                    ar = asort[ ka ].read;
                    as = ( ar & SIGN_BIT );
                    ar >>= 1;
                    ap = asort[ ka ].rpos;
                    for ( a = ja; a < ka; a++ )
                    {
                        br = asort[ a ].read;
                        bs = ( br & SIGN_BIT );
                        br >>= 1;
                        if ( br >= ar )
                            break;
                        bp = asort[ a ].rpos;
                        if ( bs == as )
                            hits[ nhits ].aread = ar;
                        else
                        {
                            bp                  = ( reads[ br ].rlen - bp ) + Koff;
                            hits[ nhits ].aread = ar + nread;
                        }
                        hits[ nhits ].bread = br;
                        hits[ nhits ].apos  = ap;
                        hits[ nhits ].diag  = ap - bp;
                        nhits += 1;
                    }
                }
        }
    }
    else
    {
        int ib, jb;
        uint64 cb;
        uint32 ar, br;
        uint32 ap, bp;
        uint32 as, bs;
        int a, b;

        ib = data->bbeg;
        cb = bsort[ ib ].code;
        while ( 1 )
        {
            ja = ia++;
            while ( 1 )
            {
                da = asort[ ia ].code;
                if ( da != ca )
                    break;
                ia += 1;
            }

            if ( unlikely(ia > aend ))
            {
                if ( ja >= aend )
                    break;
                ia = aend;
                da = asort[ ia ].code;
            }

            while ( cb < ca )
            {
                ib += 1;
                cb = bsort[ ib ].code;
            }
            if ( cb != ca )
            {
                ca = da;
                continue;
            }

            jb = ib++;
            while ( 1 )
            {
                cb = bsort[ ib ].code;
                if ( cb != ca )
                    break;
                ib += 1;
            }
            ca = da;

            if ( unlikely( ( ( int64 )( ia - ja ) ) * ( ib - jb ) >= limit ) )
                continue;

            for ( a = ja; a < ia; a++ )
            {
                ar = asort[ a ].read;
                as = ( ar & SIGN_BIT );
                ar >>= 1;
                ap = asort[ a ].rpos;
                for ( b = jb; b < ib; b++ )
                {
                    br = bsort[ b ].read;
                    bs = ( br & SIGN_BIT );
                    br >>= 1;
                    bp = bsort[ b ].rpos;
                    if ( bs == as )
                        hits[ nhits ].aread = ar;
                    else
                    {
                        bp                  = ( reads[ br ].rlen - bp ) + Koff;
                        hits[ nhits ].aread = ar + nread;
                    }
                    hits[ nhits ].bread = br;
                    hits[ nhits ].apos  = ap;
                    hits[ nhits ].diag  = ap - bp;
                    nhits += 1;
                }
            }
        }
    }

    return ( NULL );
}

//  Report threads: given a segment of merged list, find all seeds and from them all alignments.

//DAZZ_DB* MR_ablock;
//DAZZ_DB* MR_bblock;
//SeedPair* MR_hits;
//static int MR_two;
//Align_Spec* MR_spec;

#ifdef DO_BRIDGING

static inline int MapToTPAbove( Path* path, int* x, int isA, Trace_Buffer* tbuf )
{
    uint16* trace = tbuf->trace + (uint64)path->trace;
    int a, b, i;

    a = ( path->abpos / MR_tspace ) * MR_tspace;
    b = path->bbpos;
    for ( i = 1; i < path->tlen; i += 2 )
    {
        a += MR_tspace;
        b += trace[ i ];
        if ( a > path->aepos )
            a = path->aepos;
        if ( isA )
        {
            if ( a >= *x )
            {
                *x = a;
                return ( b );
            }
        }
        else
        {
            if ( b >= *x )
            {
                *x = b;
                return ( a );
            }
        }
    }
    if ( isA )
    {
        *x = a;
        return ( b );
    }
    else
    {
        *x = b;
        return ( a );
    }
}

static inline int MapToTPBelow( Path* path, int* x, int isA, Trace_Buffer* tbuf )
{
    uint16* trace = tbuf->trace + (uint64)path->trace;
    int a, b, i;

    a = ( ( path->aepos + ( MR_tspace - 1 ) ) / MR_tspace ) * MR_tspace;
    b = path->bepos;
    for ( i = path->tlen - 1; i >= 0; i -= 2 )
    {
        a -= MR_tspace;
        b -= trace[ i ];
        if ( a < path->abpos )
            a = path->abpos;
        if ( isA )
        {
            if ( a <= *x )
            {
                *x = a;
                return ( b );
            }
        }
        else
        {
            if ( b <= *x )
            {
                *x = b;
                return ( a );
            }
        }
    }
    if ( isA )
    {
        *x = a;
        return ( b );
    }
    else
    {
        *x = b;
        return ( a );
    }
}

static int Check_Bridge( Path* path )
{
    uint16* trace = (uint16*)path->trace;
    int i;

    if ( MR_tspace <= TRACE_XOVR )
    {
        for ( i = 0; i < path->tlen; i++ )
            if ( trace[ i ] > 250 )
                return ( 1 );
    }
    return ( 0 );
}

static void Compute_Bridge_Path( Path* path1, Path* path2, Alignment* align, int comp, int aovl, int bovl, Work_Data* work, Trace_Buffer* tbuf )
{
    Path* apath;
    int ain, aout;
    int bin, bout, boff;
    int err;
    int i, j, p;
    uint16* trk;

    apath = align->path;

    if ( bovl > aovl )
    {
        bin  = path2->bbpos;
        bout = path1->bepos;
        ain  = MapToTPBelow( path1, &bin, 0, tbuf );
        aout = MapToTPAbove( path2, &bout, 0, tbuf );
    }
    else
    {
        ain  = path2->abpos;
        aout = path1->aepos;
        bin  = MapToTPBelow( path1, &ain, 1, tbuf );
        bout = MapToTPAbove( path2, &aout, 1, tbuf );
    }
#ifdef TEST_BRIDGE
    printf( "\n  Tangle [%5d..%5d] vs [%5d..%5d]  %4d\n", path1->abpos, path1->aepos, path2->abpos, path2->aepos, abs( aovl - bovl ) );
    printf( "         [%5d..%5d] vs [%5d..%5d]  %4d vs %4d\n", path1->bbpos, path1->bepos, path2->bbpos, path2->bepos, aovl, bovl );
    printf( "      (%d,%d) to (%d,%d)\n", ain, bin, aout, bout );
    fflush( stdout );
#endif

    apath->abpos = ain - 2 * MR_tspace;
    apath->aepos = aout + 2 * MR_tspace;
    apath->bbpos = MapToTPBelow( path1, &( apath->abpos ), 1, tbuf );
    apath->bepos = MapToTPAbove( path2, &( apath->aepos ), 1, tbuf );

    if ( comp )
    {
        boff = MR_tspace - apath->aepos % MR_tspace;

        p            = align->alen - apath->abpos;
        apath->abpos = align->alen - apath->aepos;
        apath->aepos = p;
        p            = align->blen - apath->bbpos;
        apath->bbpos = align->blen - apath->bepos;
        apath->bepos = p;

        boff = boff - apath->abpos % MR_tspace;
        align->aseq -= boff;
        apath->abpos += boff;
        apath->aepos += boff;
        align->alen += boff;
    }

#ifdef TEST_BRIDGE
    printf( "\n      (%d,%d) to (%d,%d)\n", apath->abpos, apath->bbpos, apath->aepos, apath->bepos );
    fflush( stdout );

    Compute_Alignment( align, work, DIFF_ALIGN, 0 );
    Print_Reference( stdout, align, work, 8, 100, 10, 0, 6 );
    fflush( stdout );
#endif

    Compute_Alignment( align, work, DIFF_TRACE, MR_tspace );

    trk = (uint16*)apath->trace;
    if ( comp )
    {
        j = apath->tlen - 2;
        i = 0;
        while ( i < j )
        {
            p            = trk[ i ];
            trk[ i ]     = trk[ j ];
            trk[ j ]     = p;
            p            = trk[ i + 1 ];
            trk[ i + 1 ] = trk[ j + 1 ];
            trk[ j + 1 ] = p;
            i += 2;
            j -= 2;
        }

        align->aseq += boff;
        apath->abpos -= boff;
        apath->aepos -= boff;
        align->alen -= boff;

        p            = align->alen - apath->abpos;
        apath->abpos = align->alen - apath->aepos;
        apath->aepos = p;
        p            = align->blen - apath->bbpos;
        apath->bbpos = align->blen - apath->bepos;
        apath->bepos = p;
    }

    bin  = apath->bbpos;
    bout = apath->bepos;
    err  = apath->diffs;

    p = 2 * ( ain / MR_tspace - apath->abpos / MR_tspace );
    for ( i = 0; i < p; i += 2 )
    {
        bin += trk[ i + 1 ];
        err -= trk[ i ];
    }

    p = 2 * ( apath->aepos / MR_tspace - aout / MR_tspace );
    for ( i = align->path->tlen, p = i - p; i > p; i -= 2 )
    {
        bout -= trk[ i - 1 ];
        err -= trk[ i - 2 ];
    }

#ifdef TEST_BRIDGE
    printf( "      (%d,%d) to (%d,%d)\n", ain, bin, aout, bout );
    printf( "  Box %d vs %d -> %d %d%%\n", aout - ain, bout - bin, err, ( 200 * err ) / ( ( aout - ain ) + ( bout - bin ) ) );
    fflush( stdout );
#endif
}

#endif // DO_BRIDGING

static int Entwine( Path* jpath, Path* kpath, Trace_Buffer* tbuf, int* where )
{
    int ac, b2, y2, ae;
    int i, j, k;
    int num, den, min;
#ifdef SEE_ENTWINEF
    int strt = 1;
    int iflare, oflare;
#endif

    uint16* ktrace = tbuf->trace + ( uint64 )( kpath->trace );
    uint16* jtrace = tbuf->trace + ( uint64 )( jpath->trace );

    min = 10000;
    num = 0;
    den = 0;

#ifdef SEE_ENTWINE
    printf( "\n" );
#endif

    y2 = jpath->bbpos;
    j  = jpath->abpos / MR_tspace;

    b2 = kpath->bbpos;
    k  = kpath->abpos / MR_tspace;

    if ( jpath->abpos == kpath->abpos )
    {
        min = abs( y2 - b2 );
        if ( min == 0 )
            *where = kpath->abpos;
    }

    if ( j < k )
    {
        ac = k * MR_tspace;

        j = 1 + 2 * ( k - j );
        k = 1;

        for ( i = 1; i < j; i += 2 )
            y2 += jtrace[ i ];
    }
    else
    {
        ac = j * MR_tspace;

        k = 1 + 2 * ( j - k );
        j = 1;

        for ( i = 1; i < k; i += 2 )
            b2 += ktrace[ i ];
    }

    ae = jpath->aepos;
    if ( ae > kpath->aepos )
        ae = kpath->aepos;

    while ( 1 )
    {
        ac += MR_tspace;
        if ( ac >= ae )
            break;
        y2 += jtrace[ j ];
        b2 += ktrace[ k ];
        j += 2;
        k += 2;

#ifdef SEE_ENTWINE
        printf( "   @ %5d : %5d %5d = %4d\n", ac, y2, b2, abs( b2 - y2 ) );
#endif

        i = abs( y2 - b2 );
        if ( i <= min )
        {
            min = i;
            if ( i == 0 )
                *where = ac;
        }
        num += i;
        den += 1;
#ifdef SEE_ENTWINE
        if ( strt )
        {
            strt   = 0;
            iflare = i;
        }
        oflare = i;
#endif
    }

    if ( jpath->aepos == kpath->aepos )
    {
        i = abs( jpath->bepos - kpath->bepos );
        if ( i <= min )
        {
            min = i;
            if ( i == 0 )
                *where = kpath->aepos;
        }
    }

#ifdef SEE_ENTWINE
    if ( den == 0 )
        printf( "Nothing\n" );
    else
        printf( "MINIM = %d AVERAGE = %d  IFLARE = %d  OFLARE = %d\n", min, num / den, iflare, oflare );
#endif

    if ( den == 0 )
        return ( -1 );
    else
        return ( min );
}

//  Produce the concatentation of path1 and path2 where they are known to meet at
//    the trace point with coordinate ap. Place this result in a big growing buffer,
//    that gets reset when fusion is called with path1 = NULL

static void Fusion( Path* path1, int ap, Path* path2, Trace_Buffer* tbuf )
{
    int k, k1, k2;
    int len, diff;
    uint16* trace;

    k1 = 2 * ( ( ap / MR_tspace ) - ( path1->abpos / MR_tspace ) );
    k2 = 2 * ( ( ap / MR_tspace ) - ( path2->abpos / MR_tspace ) );

    len = k1 + ( path2->tlen - k2 );

    if ( tbuf->top + len >= tbuf->max )
    {
        tbuf->max   = 1.2 * ( tbuf->top + len ) + 1000;
        tbuf->trace = (uint16*)Realloc( tbuf->trace, sizeof( uint16 ) * tbuf->max, "Allocating alignments" );
        if ( tbuf->trace == NULL )
            exit( 1 );
    }

    trace = tbuf->trace + tbuf->top;
    tbuf->top += len;

    diff = 0;
    len  = 0;
    if ( k1 > 0 )
    {
        uint16* t = tbuf->trace + ( uint64 )( path1->trace );
        for ( k = 0; k < k1; k += 2 )
        {
            trace[ len++ ] = t[ k ];
            trace[ len++ ] = t[ k + 1 ];
            diff += t[ k ];
        }
    }
    if ( k2 < path2->tlen )
    {
        uint16* t = tbuf->trace + ( uint64 )( path2->trace );
        for ( k = k2; k < path2->tlen; k += 2 )
        {
            trace[ len++ ] = t[ k ];
            trace[ len++ ] = t[ k + 1 ];
            diff += t[ k ];
        }
    }

    path1->aepos = path2->aepos;
    path1->bepos = path2->bepos;
    path1->diffs = diff;
    path1->trace = (void*)( trace - tbuf->trace );
    path1->tlen  = len;
}

#ifdef DO_BRIDGING

//  Produce the concatentation of path1, path2, and path3 where they are known to meet at
//    the ends of path2 which was produced by Compute-Alignment. Place this result in
//    a big growing buffer.

static void Bridge( Path* path1, Path* path2, Path* path3, Trace_Buffer* tbuf )
{
    int k, k1, k2;
    int len, diff;
    uint16* trace;

    k1 = 2 * ( ( path2->abpos / MR_tspace ) - ( path1->abpos / MR_tspace ) );
    if ( path2->aepos == path3->aepos )
        k2 = path3->tlen;
    else
        k2 = 2 * ( ( path2->aepos / MR_tspace ) - ( path3->abpos / MR_tspace ) );

    len = k1 + path2->tlen + ( path3->tlen - k2 );

    if ( tbuf->top + len >= tbuf->max )
    {
        tbuf->max   = 1.2 * ( tbuf->top + len ) + 1000;
        tbuf->trace = (uint16*)Realloc( tbuf->trace, sizeof( uint16 ) * tbuf->max, "Allocating alignments" );
        if ( tbuf->trace == NULL )
            exit( 1 );
    }

    trace = tbuf->trace + tbuf->top;
    tbuf->top += len;

    diff = 0;
    len  = 0;
    if ( k1 > 0 )
    {
        uint16* t = tbuf->trace + ( uint64 )( path1->trace );
        for ( k = 0; k < k1; k += 2 )
        {
            trace[ len++ ] = t[ k ];
            trace[ len++ ] = t[ k + 1 ];
            diff += t[ k ];
        }
    }
    if ( path2->tlen > 0 )
    {
        uint16* t = (uint16*)( path2->trace );
        for ( k = 0; k < path2->tlen; k += 2 )
        {
            trace[ len++ ] = t[ k ];
            trace[ len++ ] = t[ k + 1 ];
            diff += t[ k ];
        }
    }
    if ( k2 < path3->tlen )
    {
        uint16* t = tbuf->trace + ( uint64 )( path3->trace );
        for ( k = k2; k < path3->tlen; k += 2 )
        {
            trace[ len++ ] = t[ k ];
            trace[ len++ ] = t[ k + 1 ];
            diff += t[ k ];
        }
    }

    path1->aepos = path3->aepos;
    path1->bepos = path3->bepos;
    path1->diffs = diff;
    path1->trace = (void*)( trace - tbuf->trace );
    path1->tlen  = len;
}

#endif // DO_BRIDGING

int Handle_Redundancies( Path* amatch, int novls, Path* bmatch, Alignment* align, Work_Data* work, Trace_Buffer* tbuf )
{
    Path *jpath, *kpath, *apath;
    Path _bpath, *bpath      = &_bpath;
    Alignment _blign, *blign = &_blign;

    int j, k, no;
    int dist;
    int awhen = 0, bwhen = 0;
    int hasB, comp;

#if defined( TEST_CONTAIN ) || defined( TEST_BRIDGE )
    for ( j = 0; j < novls; j++ )
        printf( "  %3d: [%5d,%5d] x [%5d,%5d]\n", j, amatch[ j ].abpos, amatch[ j ].aepos, amatch[ j ].bbpos, amatch[ j ].bepos );
#endif

    (void)work;

    //  Loop to catch LA's that share a common trace point and fuse them

    hasB = ( bmatch != NULL );
    if ( hasB )
    {
        blign->aseq = align->bseq;
        blign->bseq = align->aseq;
        blign->alen = align->blen;
        blign->blen = align->alen;
        blign->path = bpath;
    }
    apath = align->path;
    comp  = COMP( align->flags );

    (void)apath->tlen; //  Just to shut up stupid compilers

    for ( j = 1; j < novls; j++ )
    {
        jpath = amatch + j;
        for ( k = j - 1; k >= 0; k-- )
        {
            kpath = amatch + k;

            if ( kpath->abpos < 0 )
                continue;

            if ( jpath->abpos < kpath->abpos )

            {
                if ( kpath->abpos <= jpath->aepos && kpath->bbpos <= jpath->bepos )
                {
                    dist = Entwine( jpath, kpath, tbuf, &awhen );
                    if ( dist == 0 )
                    {
                        if ( kpath->aepos > jpath->aepos )
                        {
                            if ( hasB )
                            {
                                if ( comp )
                                {
                                    dist = Entwine( bmatch + k, bmatch + j, tbuf, &bwhen );
                                    if ( dist != 0 )
                                        continue;
                                    Fusion( jpath, awhen, kpath, tbuf );
                                    Fusion( bmatch + k, bwhen, bmatch + j, tbuf );
                                    bmatch[ j ] = bmatch[ k ];
#ifdef TEST_CONTAIN
                                    printf( "  Really 1" );
#endif
                                }
                                else
                                {
                                    dist = Entwine( bmatch + j, bmatch + k, tbuf, &bwhen );
                                    if ( dist != 0 )
                                        continue;
                                    Fusion( jpath, awhen, kpath, tbuf );
                                    Fusion( bmatch + j, bwhen, bmatch + k, tbuf );
#ifdef TEST_CONTAIN
                                    printf( "  Really 2" );
#endif
                                }
                            }
                            else
                            {
                                Fusion( jpath, awhen, kpath, tbuf );
#ifdef TEST_CONTAIN
                                printf( "  Really 3" );
#endif
                            }
                            k = j;
                        }
                        kpath->abpos = -1;
#ifdef TEST_CONTAIN
                        printf( "  Fuse! A %d %d\n", j, k );
#endif
                    }
                }
            }

            else // kpath->abpos <= jpath->abpos

            {
                if ( jpath->abpos <= kpath->aepos && jpath->bbpos <= kpath->bepos )
                {
                    dist = Entwine( kpath, jpath, tbuf, &awhen );
                    if ( dist == 0 )
                    {
                        if ( kpath->abpos == jpath->abpos )
                        {
                            if ( kpath->aepos > jpath->aepos )
                            {
                                *jpath = *kpath;
                                if ( hasB )
                                    bmatch[ j ] = bmatch[ k ];
                            }
                        }
                        else if ( jpath->aepos > kpath->aepos )
                        {
                            if ( hasB )
                            {
                                if ( comp )
                                {
                                    dist = Entwine( bmatch + j, bmatch + k, tbuf, &bwhen );
                                    if ( dist != 0 )
                                        continue;
                                    Fusion( kpath, awhen, jpath, tbuf );
                                    *jpath = *kpath;
                                    Fusion( bmatch + j, bwhen, bmatch + k, tbuf );
#ifdef TEST_CONTAIN
                                    printf( "  Really 4" );
#endif
                                }
                                else
                                {
                                    dist = Entwine( bmatch + k, bmatch + j, tbuf, &bwhen );
                                    if ( dist != 0 )
                                        continue;
                                    Fusion( kpath, awhen, jpath, tbuf );
                                    *jpath = *kpath;
                                    Fusion( bmatch + k, bwhen, bmatch + j, tbuf );
                                    bmatch[ j ] = bmatch[ k ];
#ifdef TEST_CONTAIN
                                    printf( "  Really 5" );
#endif
                                }
                            }
                            else
                            {
                                Fusion( kpath, awhen, jpath, tbuf );
                                *jpath = *kpath;
#ifdef TEST_CONTAIN
                                printf( "  Really 6" );
#endif
                            }
                            k = j;
                        }
                        else
                        {
                            *jpath = *kpath;
                            if ( hasB )
                                bmatch[ j ] = bmatch[ k ];
                        }
                        kpath->abpos = -1;
#ifdef TEST_CONTAIN
                        printf( "  Fuse! B %d %d\n", j, k );
#endif
                    }
                }
            }
        }
    }

#ifdef DO_BRIDGING

    //  Loop to catch LA's that have a narrow parallel overlap and bridge them

    for ( j = 1; j < novls; j++ )
    {
        jpath = amatch + j;
        if ( jpath->abpos < 0 )
            continue;

        for ( k = j - 1; k >= 0; k-- )
        {
            Path *path1, *path2;
            Path *bath1, *bath2;
            int aovl, bovl;
            Path jback, kback;

            kpath = amatch + k;
            if ( kpath->abpos < 0 )
                continue;

            if ( jpath->abpos < kpath->abpos )
            {
                path1 = jpath;
                path2 = kpath;
            }
            else
            {
                path1 = kpath;
                path2 = jpath;
            }

            if ( path2->abpos >= path1->aepos || path1->aepos >= path2->aepos || path1->bbpos >= path2->bbpos || path2->bbpos >= path1->bepos ||
                 path1->bepos >= path2->bepos )
                continue;
            aovl = path1->aepos - path2->abpos;
            bovl = path1->bepos - path2->bbpos;
            if ( abs( aovl - bovl ) > .2 * ( aovl + bovl ) )
                continue;

            if ( hasB )
            {
                if ( comp == ( jpath->abpos < kpath->abpos ) )
                {
                    bath1 = bmatch + k;
                    bath2 = bmatch + j;
                }
                else
                {
                    bath1 = bmatch + j;
                    bath2 = bmatch + k;
                }
                if ( bath1->abpos > bath2->abpos )
                {
                    printf( "  SYMFAIL %d %d\n", j, k );
                    continue;
                }
            }

            Compute_Bridge_Path( path1, path2, align, 0, aovl, bovl, work, tbuf );

            if ( Check_Bridge( apath ) )
                continue;

            jback = *jpath;
            kback = *kpath;

            Bridge( path1, apath, path2, tbuf );
            *jpath       = *path1;
            kpath->abpos = -1;

#ifdef TEST_BRIDGE
            {
                Alignment extra;
                Path pcopy;

                pcopy       = *jpath;
                extra       = *align;
                pcopy.trace = tbuf->trace + (uint64)jpath->trace;
                extra.path  = &pcopy;
                Compute_Trace_PTS( &extra, work, MR_tspace, GREEDIEST );
                Print_Reference( stdout, &extra, work, 8, 100, 10, 0, 6 );
                fflush( stdout );
            }
#endif

            if ( hasB )
            {
                Compute_Bridge_Path( bath1, bath2, blign, comp, bovl, aovl, work, tbuf );

                if ( Check_Bridge( bpath ) )
                {
                    *jpath = jback;
                    *kpath = kback;
                    continue;
                }

                Bridge( bath1, bpath, bath2, tbuf );
                bmatch[ j ] = *bath1;

#ifdef TEST_BRIDGE
                {
                    Alignment extra;
                    Path pcopy;

                    pcopy       = bmatch[ j ];
                    extra       = *blign;
                    pcopy.trace = tbuf->trace + (uint64)bmatch[ j ].trace;
                    extra.path  = &pcopy;
                    if ( comp )
                    {
                        Complement_Seq( extra.aseq, extra.alen );
                        Complement_Seq( extra.bseq, extra.blen );
                    }
                    Compute_Trace_PTS( &extra, work, MR_tspace, GREEDIEST );
                    Print_Reference( stdout, &extra, work, 8, 100, 10, 0, 6 );
                    fflush( stdout );
                    if ( comp )
                    {
                        Complement_Seq( extra.aseq, extra.alen );
                        Complement_Seq( extra.bseq, extra.blen );
                    }
                }
#endif
            }
        }
    }

#endif // DO_BRIDGING

    no = 0;
    for ( j = 0; j < novls; j++ )
        if ( amatch[ j ].abpos >= 0 )
        {
            if ( hasB )
                bmatch[ no ] = bmatch[ j ];
            amatch[ no++ ] = amatch[ j ];
        }
    novls = no;

#if defined( TEST_CONTAIN ) || defined( TEST_BRIDGE )
    for ( j = 0; j < novls; j++ )
        printf( "  %3d: [%5d,%5d] x [%5d,%5d]\n", j, amatch[ j ].abpos, amatch[ j ].aepos, amatch[ j ].bbpos, amatch[ j ].bepos );
#endif

    return ( novls );
}

void Diagonal_Span( Path* path, int* mind, int* maxd )
{
    uint16* points;
    int i, tlen;
    int dd, low, hgh;

    points = path->trace;
    tlen   = path->tlen;

    dd  = path->abpos - path->bbpos;
    low = hgh = dd;

    dd = path->aepos - path->bepos;
    if ( dd < low )
        low = dd;
    else if ( dd > hgh )
        hgh = dd;

    dd = ( path->abpos / MR_tspace ) * MR_tspace - path->bbpos;
    tlen -= 2;
    for ( i = 1; i < tlen; i += 2 )
    {
        dd += MR_tspace - points[ i ];
        if ( dd < low )
            low = dd;
        else if ( dd > hgh )
            hgh = dd;
    }

    *mind = ( low >> Binshift ) - 1;
    *maxd = ( hgh >> Binshift ) + 1;
}
