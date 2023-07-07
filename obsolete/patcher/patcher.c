
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/param.h>

#include "lib/colors.h"
#include "lib/lasidx.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/utils.h"

#include "db/DB.h"

#define DEF_ARG_E 12

#define VERBOSE
#undef DEBUG
#undef DEBUG_CLASSIFY

typedef uint64_t KMER;

typedef uint8_t KMER_COUNT;
#define KMER_COUNT_MAX 255

typedef struct
{
    uint64_t bases;
} KC_Header;

typedef struct
{
    HITS_DB* db;
    char* pathDb;

    // k-mer k
    int k;

    uint16_t error;
    uint64_t genomesize;

    // kmer occurance counts. capped at KMER_COUNT_MAX
    KMER_COUNT* kcounts;
    uint64_t kcount;
    uint64_t bases;

    // kmer occurance histogram
    uint64_t* histo_counts;
    uint64_t histo_sum;

} KmersContext;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

static int kmer_count_noise(KmersContext* kctx)
{
    double cov = kctx->bases / kctx->genomesize;
    double err = kctx->error / 100.0;
    int k = kctx->k;

    // double prob_kmer_conserved = exp( -1.0 * (kctx->error / 100.0) * kctx->k );
    double prob_kmer_conserved = pow(1-err, k+1) + (1.0/k) * ( pow(1-err, k-1) - pow(1-err, 2*k - 1) );

    printf( "pkc %lf\n", prob_kmer_conserved );

    return ceill( cov * 0.5 * prob_kmer_conserved );
}

static void print_bits( size_t const size, void const* const ptr )
{
    unsigned char* b = (unsigned char*)ptr;
    unsigned char byte;
    int i, j;

    for ( i = size - 1; i >= 0; i-- )
    {
        for ( j = 7; j >= 0; j-- )
        {
            byte = ( b[ i ] >> j ) & 1;
            printf( "%u", byte );
        }

        printf( " " );
    }
    puts( "" );
}

static void print_histograms( KmersContext* kctx )
{
    uint64_t total, count;
    total = count = 0;

    uint64_t* bins = kctx->histo_counts;
    uint16_t cov   = kctx->bases / kctx->genomesize;
    uint16_t noise = kmer_count_noise( kctx );

    uint64_t i;
    for ( i = 0; i <= KMER_COUNT_MAX; i++ )
    {
        printf( "%3lld %3lld", i, bins[ i ] );

        if ( i == cov )
        {
            printf( " COV\n" );
        }
        else if ( i == noise )
        {
            printf( " NOISE\n" );
        }
        else
        {
            printf( "\n" );
        }

        if ( i > 0 )
        {
            total += i * bins[ i ];
            count += bins[ i ];
        }
    }

    printf( "avg    %.2f (with %d)\n", (double)( total ) / count, KMER_COUNT_MAX );
    total -= KMER_COUNT_MAX * bins[ KMER_COUNT_MAX ];
    count -= bins[ KMER_COUNT_MAX ];
    printf( "avg    %.2f\n", (double)( total ) / count );
}

static int load_kcounts( KmersContext* kctx )
{
    int k        = kctx->k;
    char* pathDb = kctx->pathDb;
    HITS_DB* db  = kctx->db;

    uint64_t kcount = kctx->kcount = ( 1llu << ( 2 * k ) );
    uint64_t kmask                 = kcount - 1;

    KMER_COUNT* kcounts = kctx->kcounts = malloc( sizeof( KMER_COUNT ) * kcount );
    bzero( kcounts, sizeof( KMER_COUNT ) * kcount );

    if ( kcounts == NULL )
    {
        fprintf( stderr, "failed to allocate kcounts\n" );
        return 0;
    }

    int read = 0;

    char pcPathKmers[ PATH_MAX ];
    sprintf( pcPathKmers, "%s.%dmers", pathDb, k );
    FILE* fileKmers = fopen( pcPathKmers, "r" );
    KC_Header kch;

    if ( fileKmers )
    {
#ifdef VERBOSE
        printf( "loading k-mer counts\n" );
#endif

        if ( fread( &kch, sizeof( KC_Header ), 1, fileKmers ) != 1 )
        {
            fprintf( stderr, "ERROR: failed to read file header\n" );
            exit( 1 );
        }

        if ( fread( kcounts, sizeof( KMER_COUNT ), kcount, fileKmers ) != kcount )
        {
            fprintf( stderr, "ERROR: failed to load kmer counts\n" );
            exit( 1 );
        }

        fclose( fileKmers );
    }
    else
    {
#ifdef VERBOSE
        printf( "computing k-mer counts\n" );
#endif

        int nblocks = DB_Blocks( pathDb );
        kch.bases   = 0;

        int block;
        for ( block = 1; block <= nblocks; block++ )
        {
            fprintf( stderr, "%s %d/%d\n", pathDb, block, nblocks );

            Open_DB_Block( pathDb, db, block );

            // pass

            kch.bases += db->totlen;

            Read_All_Sequences( db, 0 );

            for ( read = 0; read < db->nreads; read++ )
            {
                char* bases = db->bases + db->reads[ read ].boff;
                int rlen    = db->reads[ read ].rlen;

                int idx   = 0;
                KMER kmer = 0;
                while ( idx < k - 1 )
                {
                    kmer = ( kmer << 2 ) + bases[ idx ];
                    idx++;
                }

                while ( idx < rlen )
                {
                    kmer = ( ( kmer << 2 ) + bases[ idx ] ) & kmask;
                    idx++;

                    if ( kcounts[ kmer ] < KMER_COUNT_MAX )
                    {
                        kcounts[ kmer ] += 1;
                    }
                }
            }

            Close_DB( db );
        }

        FILE* fileOut = fopen( pcPathKmers, "w" );
        fwrite( &kch, sizeof( KC_Header ), 1, fileOut );
        fwrite( kcounts, sizeof( KMER_COUNT ), kcount, fileOut );
        fclose( fileOut );
    }

    kctx->bases = kch.bases;

// create k-mer occurance count histogram

#ifdef VERBOSE
    printf( "computing k-mer histogram for %s bases\n", bp_format(kch.bases, 1) );
#endif

    uint64_t* bins = kctx->histo_counts = malloc( sizeof( uint64_t ) * ( KMER_COUNT_MAX + 1 ) );
    bzero( bins, sizeof( uint64_t ) * ( KMER_COUNT_MAX + 1 ) );

    uint64_t i;
    for ( i = 0; i < kcount; i++ )
    {
        bins[ kcounts[ i ] ] += 1;
    }

    uint64_t sum = 0;
    for ( i = 1; i < KMER_COUNT_MAX; i++ )
    {
        sum += bins[ i ];
    }

    kctx->histo_sum = sum;

    return 1;
}

static void usage()
{
    printf( "usage:   [-e <error>] <db> <k> <genome.mb>\n" );
    printf( "options: -e sequence error rate (default %d)\n", DEF_ARG_E );
}

int main( int argc, char* argv[] )
{
    KmersContext ctx;
    HITS_DB db;
    int c;

    bzero( &ctx, sizeof( KmersContext ) );
    ctx.db    = &db;
    ctx.error = DEF_ARG_E;

    // process arguments

    opterr = 0;

    while ( ( c = getopt( argc, argv, "e:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'e':
                ctx.error = atoi( optarg );
                break;

            default:
                fprintf( stderr, "unknow option: %s\n", argv[ optind - 1 ] );
                usage();
                exit( 1 );
        }
    }

    if ( argc - optind < 3 )
    {
        usage();
        exit( 1 );
    }

    ctx.pathDb     = argv[ optind++ ];
    ctx.k          = atoi( argv[ optind++ ] );
    ctx.genomesize = bp_parse( argv[ optind++ ] );

    if ( !load_kcounts( &ctx ) )
    {
        exit( 1 );
    }

    print_histograms( &ctx );

    free( ctx.kcounts );
    free( ctx.histo_counts );

    return 0;
}
