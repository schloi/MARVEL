
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>

#include "lib/colors.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/trim.h"
#include "lib/utils.h"

#include "dalign/align.h"
#include "db/DB.h"

// constants

#define DEF_ARG_B 0
#define DEF_ARG_M 0
#define DEF_ARG_O 0
#define DEF_ARG_LL 0

// toggles

#define VERBOSE
#undef DEBUG


typedef struct
{
    HITS_DB* db;
    HITS_TRACK* tracktrim;

    // arguments

    int min_aln_len;
    int min_rlen;
    int cov_res;

    int show_bins;
    int show_max;

    FILE* fout;

    // read coverage tracking

    uint64_t* cov;
    size_t cov_max;

    // read sets

    int* read2set;
    int nsets;

    // overall coverage tracking - set version

    uint64_t** set_histo;
    size_t* set_histo_max;
    uint64_t* set_nhisto;

    // events

    int emax;
    int dmax;
    int dcur;
    int* events;

} CoverageContext;

extern char* optarg;
extern int optind, opterr, optopt;

static int cmp_coverage_events( const void* x, const void* y )
{
    int* e1 = (int*)x;
    int* e2 = (int*)y;

    int cmp = abs( *e1 ) - abs( *e2 );

    if ( cmp == 0 )
    {
        cmp = ( *e2 ) - ( *e1 );
    }

    return cmp;
}

static void pre_coverage( CoverageContext* ctx )
{
#ifdef VERBOSE
    printf( ANSI_COLOR_GREEN "PASS coverage\n" ANSI_COLOR_RESET );
#endif

    ctx->emax   = 1024;
    ctx->events = malloc( sizeof( int ) * ctx->emax );

    ctx->cov_max = 64 * 1024;
    ctx->cov = calloc(ctx->cov_max, sizeof(uint64_t));
}

static void post_coverage( CoverageContext* ctx )
{
    size_t* set_nhisto = ctx->set_nhisto;
    uint64_t** set_histo = ctx->set_histo;
    int nsets = ctx->nsets;
    int show_max = ctx->show_max;
    int show_bins = ctx->show_bins;
    FILE* fout = ctx->fout;

    uint64_t i, j;

    if (show_max)
    {
        fprintf(fout, "MAX");

        for ( i = 0 ; i < nsets + 1 ; i++ )
        {
            uint64_t max = 0;
            uint64_t max_val = 0;

            for ( j = 0 ; j <= set_nhisto[i] ; j++ )
            {
                if ( j > 0 && set_histo[i][j] > max_val )
                {
                    max = j;
                    max_val = set_histo[i][j];
                }
            }

            fprintf(fout, " %" PRIu64, max);
        }

        fprintf(fout, "\n");
    }

    if ( show_bins )
    {
        int exhausted = 0;

        j = 0;
        while ( exhausted == 0 )
        {
            fprintf(fout, "HIST %" PRIu64, j);
            exhausted = 1;

            for ( i = 0 ; i < nsets + 1 ; i++ )
            {
                if ( j < set_nhisto[i] )
                {
                    exhausted = 0;
                    fprintf(fout, " %" PRIu64, set_histo[i][j]);
                }
                else
                {
                    fprintf(fout, " 0");
                }
            }

            fprintf(fout, "\n");

            j += 1;
        }
    }

    free( ctx->events );
    free( ctx->cov );

#ifdef VERBOSE

#endif
}

static int handler_coverage( void* _ctx, Overlap* ovl, int novl )
{
    CoverageContext* ctx       = (CoverageContext*)_ctx;
    int* events           = ctx->events;
    int cov_res = ctx->cov_res;
    uint64_t* cov = ctx->cov;
    size_t cov_max = ctx->cov_max;
    int aread = ovl->aread;
    int atrimb, atrime;
    int nset = ctx->read2set[aread];

    uint64_t* histo = ctx->set_histo[nset];
    size_t histo_max = ctx->set_histo_max[nset];

    int alen = DB_READ_LEN( ctx->db, aread );

    if ( ctx->tracktrim )
    {
        get_trim(ctx->db, ctx->tracktrim, aread, &atrimb, &atrime);
    }
    else
    {
        atrimb = 0;
        atrime = alen;
    }

    // ensure space in events buffer and create entry/exit events

    if ( 2 * novl >= ctx->emax )
    {
        ctx->emax   = 2.2 * novl + 1000;
        ctx->events = events = (int*)realloc( events, sizeof( int ) * ctx->emax );
    }

    int i;
    unsigned int j = 0;
    for ( i = 0; i < novl; i++ )
    {
        // int abpos = MAX(atrimb, ovl[ i ].path.abpos);
        // int aepos = MIN(atrime, ovl[ i ].path.aepos);

        int abpos = ovl[i].path.abpos;
        int aepos = ovl[i].path.aepos;

        if ( ovl[ i ].aread == ovl[ i ].bread
          || aepos - abpos <= ctx->min_aln_len
          || alen < ctx->min_rlen || DB_READ_LEN( ctx->db, ovl[ i ].bread ) < ctx->min_rlen
           )
        {
            continue;
        }

        assert( abpos >= 0);
        assert( aepos <= alen );
        assert( abpos < aepos );

        events[ j++ ] = abpos;
        events[ j++ ] = -( aepos - 1 );
    }

    novl = j / 2;

    qsort( events, 2 * novl, sizeof( int ), cmp_coverage_events );

    // coverage across read at cov_res resolution

    int span            = 0;
    int span_prev = 0;
    int pos_prev = abs(events[0]);
    int pos;
    int maxj = 0;
    unsigned int maxcov = 0;

    bzero(cov, cov_max * sizeof(uint64_t));

    for ( i = 0; i < 2 * novl; i++ )
    {
        if ( events[ i ] < 0 )
        {
            pos = -1 * events[i];
            span -= 1;
        }
        else
        {
            pos = events[i];
            span += 1;
        }

        j = pos_prev / cov_res;
        do
        {
            if ( j >= cov_max )
            {
                size_t temp = j * 1.2 + 100;
                cov = ctx->cov = realloc(cov, temp * sizeof(uint64_t));

                bzero(cov + cov_max, (temp - cov_max) * sizeof(uint64_t));

                ctx->cov_max = cov_max = temp;
            }

            cov[j] = MAX(cov[j], span_prev);
            maxcov = MAX(cov[j], maxcov);

            j += 1;
        } while ( j < (pos + cov_res - 1) / cov_res );

        maxj = MAX(maxj, j - 1);

        pos_prev = pos;
        span_prev = span;
    }

    if ( maxcov >= histo_max )
    {
        size_t temp = maxcov * 1.2 + 100;
        histo = ctx->set_histo[nset] = realloc(histo, temp * sizeof(uint64_t));
        bzero(histo + histo_max, (temp - histo_max) * sizeof(uint64_t));
        histo_max = ctx->set_histo_max[nset] = temp;
    }

    for (i = 0; i <= maxj; i++)
    {
        histo[ cov[i] ] += 1;
    }

    ctx->set_nhisto[nset] = MAX(ctx->set_nhisto[nset], maxcov);

    return 1;
}

static void usage()
{
    printf( "usage: [-bm] [-Os file] database input.las\n\n" );

    printf( "Coverage statistics.\n\n" );

    printf( "options: -b print bins\n");
    printf( "         -m print maximum (most likely coverage)\n" );
    printf( "         -O output results to file instead of stdout\n");
    printf( "         -s separate coverage statistics by sets of read ids\n");

}

int main( int argc, char* argv[] )
{
    HITS_DB db;
    PassContext* pctx;
    CoverageContext rctx;
    FILE* fileOvlIn;

    bzero( &rctx, sizeof( CoverageContext ) );
    rctx.db = &db;

    // process arguments

    char* pathout = NULL;
    char* nametrim = NULL;
    char* pathsets = NULL;

    rctx.min_aln_len    = DEF_ARG_O;
    rctx.min_rlen       = DEF_ARG_LL;
    rctx.show_bins      = DEF_ARG_B;
    rctx.show_max       = DEF_ARG_M;
    rctx.fout = stdout;
    rctx.cov_res = 100;

    int c;

    opterr = 0;

    while ( ( c = getopt( argc, argv, "bmL:o:O:s:t:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'b':
                rctx.show_bins = 1;
                break;

            case 'm':
                rctx.show_max = 1;
                break;

            case 'L':
                rctx.min_rlen = atoi( optarg );
                break;

            case 'o':
                rctx.min_aln_len = atoi( optarg );
                break;

            case 'O':
                pathout = optarg;
                break;

            case 's':
                pathsets = optarg;
                break;

            case 't':
                nametrim = optarg;
                break;

            default:
                usage();
                exit( 1 );
        }
    }

    if ( argc - optind != 2 )
    {
        usage();
        exit( 1 );
    }

    if ( rctx.show_max == 0 && rctx.show_bins == 0 )
    {
        fprintf( stderr, "nothing to do\n\n");
        usage();
        exit( 1 );
    }

    if ( pathout != NULL )
    {
        rctx.fout = fopen(pathout, "w");
        if ( rctx.fout == NULL )
        {
            fprintf(stderr, "could not open %s\n", pathout);
            exit( 1 );
        }
    }

    char* pcPathReadsIn  = argv[ optind++ ];
    char* pcPathOverlaps = argv[ optind++ ];

    if ( ( fileOvlIn = fopen( pcPathOverlaps, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open '%s'\n", pcPathOverlaps );
        exit( 1 );
    }

    // init

    pctx = pass_init( fileOvlIn, NULL );

    pctx->split_b      = 0;
    pctx->load_trace   = 0;
    pctx->unpack_trace = 0;
    pctx->data         = &rctx;

    if ( Open_DB( pcPathReadsIn, &db ) )
    {
        fprintf( stderr, "failed to open database '%s'\n", pcPathReadsIn );
        exit( 1 );
    }

    if ( nametrim != NULL )
    {
        rctx.tracktrim = track_load(&db, nametrim);

        if ( rctx.tracktrim == NULL )
        {
            fprintf( stderr, "failed to load track '%s'\n", nametrim);
            exit(1);
        }
    }

    rctx.read2set = malloc( db.nreads * sizeof(int) );
    bzero(rctx.read2set, db.nreads * sizeof(int) );

    if ( pathsets != NULL )
    {
        FILE* fin = fopen(pathsets, "r");
        if ( fin == NULL )
        {
            fprintf(stderr, "could not open %s\n", pathsets);
            exit( 1 );
        }

        int64_t* values;
        uint64_t* sets;
        size_t nsets;

        nsets = fread_integer_sets(fin, &values, &sets);

        size_t i;
        for ( i = 0; i < nsets; i++ )
        {
            uint64_t beg = sets[ i ];
            uint64_t end = sets[ i + 1 ];

            while ( beg < end )
            {
                rctx.read2set[ values[beg] ] = i + 1;
                beg += 1;
            }
        }

        rctx.nsets = nsets;

        free(values);
        free(sets);

        fclose(fin);
    }
    else
    {
        rctx.nsets = 0;
    }

    rctx.set_histo = malloc( sizeof(uint64_t*) * (rctx.nsets + 1) );
    rctx.set_histo_max = malloc( sizeof(size_t) * (rctx.nsets + 1) );
    rctx.set_nhisto = malloc( sizeof(uint64_t) * (rctx.nsets + 1) );

    int i;
    for ( i = 0 ; i < rctx.nsets + 1 ; i++ )
    {
        rctx.set_nhisto[i] = 0;
        rctx.set_histo_max[i] = 64 * 1024;
        rctx.set_histo[i] = calloc(rctx.set_histo_max[i], sizeof(uint64_t));
    }

    // passes

    pre_coverage( &rctx );
    pass( pctx, handler_coverage );
    post_coverage( &rctx );

    // cleanup

    pass_free( pctx );

    fclose( fileOvlIn );

    free(rctx.read2set);

    for ( i = 0 ; i < rctx.nsets + 1 ; i++ )
    {
        free( rctx.set_histo[i] );
    }
    free(rctx.set_histo);
    free(rctx.set_histo_max);
    free(rctx.set_nhisto);

    Close_DB( &db );

    return 0;
}
