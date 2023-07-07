
#include "DB.h"
#include "lib/utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEF_ARG_B 1000

extern char* optarg;
extern int optind, opterr, optopt;

static int cmp_int64(const void* a, const void* b)
{
    int64 x = * ( (int64*)a );
    int64 y = * ( (int64*)b );

    if (x < y)
    {
        return -1;
    }
    else if ( x > y)
    {
        return 1;
    }

    return 0;
}

static void usage()
{
    fprintf( stderr, "usage: [-enr] [-bg <int>] <name:db>\n" );
    fprintf( stderr, "options: -b ... bucket size of histogram length (%d)\n", DEF_ARG_B );
    fprintf( stderr, "         -e ... show empty bins\n");
    fprintf( stderr, "         -g ... genome size\n" );
    fprintf( stderr, "         -r ... raw output\n" );
}

int main( int argc, char* argv[] )
{
    HITS_DB _db, *db = &_db;

    int nbin, *hist;
    int64* bsum;

    int BIN     = DEF_ARG_B;
    int64 GSIZE = -1;
    int raw     = 0;
    int emptybins = 0;

    // parse arguments

    int c;
    opterr = 0;

    while ( ( c = getopt( argc, argv, "erb:g:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'e':
                emptybins = 1;
                break;

            case 'r':
                raw = 1;
                break;

            case 'g':
                GSIZE = bp_parse( optarg );
                if ( GSIZE <= 0 )
                {
                    fprintf( stderr, "Invalid genome size of %lld\n", GSIZE );
                    exit( 1 );
                }
                break;

            case 'b':
                BIN = atoi( optarg );
                if ( BIN <= 0 )
                {
                    fprintf( stderr, "Invalid histogram bucket size of %d\n", BIN );
                    exit( 1 );
                }
                break;

            default:
                fprintf( stderr, "Unsupported argument: %s\n", argv[ optind ] );
                usage();
                exit( 1 );
        }
    }

    if ( optind + 1 > argc )
    {
        fprintf( stderr, "[ERROR]: Database is required\n" );
        usage();
        exit( 1 );
    }

    int i, status;

    status = Open_DB( argv[ optind ], db );
    if ( status < 0 )
    {
        exit( 1 );
    }

    int64 totlen;
    int nreads, maxlen;
    int64 ave, dev;
    HITS_READ* reads;
    int64* cum;
    int64* btot;

    nreads = db->nreads;
    totlen = db->totlen;
    maxlen = db->maxlen;
    reads  = db->reads;

    nbin = maxlen / BIN + 1;
    hist = (int*)Malloc( sizeof( int ) * nbin, "Allocating histograms" );
    bsum = (int64*)Malloc( sizeof( int64 ) * nbin, "Allocating histograms" );
    btot = malloc( sizeof( int64 ) * nbin );
    cum  = malloc( sizeof( int64 ) * nbin );
    if ( !hist || !bsum || !btot || !cum )
    {
        exit( 1 );
    }

    bzero( hist, sizeof( int ) * nbin );
    bzero( bsum, sizeof( uint64 ) * nbin );
    bzero( cum, sizeof( uint64 ) * nbin );
    bzero( btot, sizeof( uint64 ) * nbin );

    for ( i = 0; i < nreads; i++ )
    {
        int rlen = reads[ i ].rlen;
        hist[ rlen / BIN ] += 1;
        bsum[ rlen / BIN ] += rlen;
    }

    int64* lengths = malloc( sizeof(int64) * nreads );

    nbin = ( maxlen - 1 ) / BIN + 1;
    ave  = totlen / nreads;
    dev  = 0;
    for ( i = 0; i < nreads; i++ )
    {
        int rlen = lengths[i] = reads[ i ].rlen;
        dev += ( rlen - ave ) * ( rlen - ave );
    }
    dev = (int64)sqrt( ( 1. * dev ) / nreads );

    qsort(lengths, nreads, sizeof(int64), cmp_int64);
    int64 sum = 0;
    i = 0;
    while ( sum < totlen/2 )
    {
        sum += lengths[i];
        i += 1;
    }

    int64 n50 = lengths[i];

    free(lengths);

    int64 _cum  = 0;
    int64 _btot = 0;

    for ( i = nbin - 1; i >= 0; i-- )
    {
        _cum += hist[ i ];
        _btot += bsum[ i ];

        cum[ i ]  = _cum;
        btot[ i ] = _btot;
    }

    if ( raw )
    {
        printf( "%d %lld %lld %lld %lld\n", nreads, totlen, ave, dev, n50 );
    }
    else
    {
        printf( "%d reads %lld base pairs\n", nreads, totlen );
        printf( "%lld average read length with %lld standard deviation and n50 %lld\n", ave, dev, n50 );
    }

    if ( raw )
    {
        printf( "A %.3f C %.3f G %.3f T %.3f\n", db->freq[ 0 ], db->freq[ 1 ], db->freq[ 2 ], db->freq[ 3 ] );
    }
    else
    {
        printf( "Base composition: %.3f(A) %.3f(C) %.3f(G) %.3f(T)\n", db->freq[ 0 ], db->freq[ 1 ], db->freq[ 2 ],
                db->freq[ 3 ] );
        printf( "\nDistribution of Read Lengths (Bin size = %d)\n\n", BIN );

        printf( "%11s %11s %7s %7s %9s", "Bin", "Count", "% Reads", "% Bases", "Average" );
        if ( GSIZE > 0 )
        {
            printf( " %11s", "Coverage" );
        }

        printf( "\n" );
    }

    for ( i = nbin - 1; i >= 0; i-- )
    {
        if ( emptybins || hist[ i ] > 0 )
        {
            if ( raw )
            {
                printf( "%d %d %.1f %.1f %lld", ( i * BIN ), hist[ i ], ( 100. * cum[ i ] ) / nreads,
                        ( 100. * btot[ i ] ) / totlen, btot[ i ] / cum[ i ] );

                if ( GSIZE > 0 )
                {
                    printf( " %.2f", btot[ i ] * 1.0 / GSIZE );
                }
                printf( "\n" );
            }
            else
            {
                printf( "%11d %11d %7.1f %7.1f %9lld", ( i * BIN ), hist[ i ], ( 100. * cum[ i ] ) / nreads,
                        ( 100. * btot[ i ] ) / totlen, btot[ i ] / cum[ i ] );

                if ( GSIZE > 0 )
                {
                    printf( " %11.2f", btot[ i ] * 1.0 / GSIZE );
                }
                printf( "\n" );
            }
        }

        if ( cum[ i ] == nreads )
        {
            break;
        }
    }

    free( hist );
    free( bsum );
    free( btot );
    free( cum );

    Close_DB( db );

    exit( 0 );
}
