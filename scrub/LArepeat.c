/*******************************************************************************************
 *
 *  creates a repeat annotation track (named -t) based on coverage statistics
 *
 *  intervals are tagged as repetitive when the -h coverage threshold is reached
 *  and terminated when the coverage drops below -l
 *
 *  -n ... max number of overlap groups that should be used for the coverage estimate
 *  -m ... merge consecutive repeats with distance less than -m
 *
 *  Author :  MARVEL Team
 *
 *******************************************************************************************/

#include "dalign/align.h"
#include "db/DB.h"
#include "lib/colors.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/utils.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>

// constants

#define BINSIZE_COVERAGE 100

#define DEFAULT_RP_XCOV_ENTER 2.0
#define DEFAULT_RP_XCOV_LEAVE 1.7
#define DEFAULT_RP_MERGE_DIST -1
#define DEFAULT_COV           -1
#define DEFAULT_COV_MAX_READS -1

#define DEF_ARG_IC 0
#define DEF_ARG_O  0
#define DEF_ARG_LL 0

#define DEF_ARG_R 0
#define DEF_ARG_I 0
#define DEF_ARG_S 0.0

// toggles

#undef VERBOSE
#undef DEBUG

// macros

typedef struct
{
    HITS_DB* db;
    HITS_TRACK* tracktrim;

    int cov;      // expected/estimated coverage
    int avg_rlen; // average read length
    int inccov;   // include coverage in the resulting track
    int min_aln_len;
    int min_rlen;
    int max_repeat_len;
    int min_repeat_isolation;

    uint64_t bases;
    uint64_t reads;

    // repeat pass

    size_t rp_emax;
    size_t rp_dmax;
    int rp_dcur;
    track_data* rp_data;
    track_anno* rp_anno;
    int* rp_events;

    int rp_merge_dist;

    int rp_merged;
    uint64_t rp_bases;
    uint64_t rp_repeat_bases;

    track_data* rp_working;
    size_t rp_working_max;

    char* rp_track;
    int rp_block;

    double rp_xcov_enter;
    double rp_xcov_leave;

} RepeatContext;

extern char* optarg;
extern int optind, opterr, optopt;

static int cmp_repeats_events( const void* x, const void* y )
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

static void pre_repeats( RepeatContext* ctx )
{
#ifdef VERBOSE
    printf( ANSI_COLOR_GREEN "PASS repeats\n" ANSI_COLOR_RESET );
#endif

    ctx->rp_emax   = 100;
    ctx->rp_events = (int*)malloc( sizeof( int ) * ctx->rp_emax );
    ctx->rp_anno   = (track_anno*)malloc( sizeof( track_anno ) * ( DB_NREADS( ctx->db ) + 1 ) );
    bzero( ctx->rp_anno, sizeof( track_anno ) * ( DB_NREADS( ctx->db ) + 1 ) );

    ctx->rp_dcur = 0;
    ctx->rp_dmax = 100;
    ctx->rp_data = (track_data*)malloc( sizeof( track_data ) * ctx->rp_dmax );

    ctx->rp_working_max = 100;
    ctx->rp_working = malloc(sizeof(track_data) * ctx->rp_working_max);
}

static void post_repeats( RepeatContext* ctx )
{
    int j;
    track_anno coff, off;

    off = 0;

    for ( j = 0; j <= DB_NREADS( ctx->db ); j++ )
    {
        coff              = ctx->rp_anno[ j ];
        ctx->rp_anno[ j ] = off;
        off += coff;
    }

    track_write( ctx->db, ctx->rp_track, ctx->rp_block, ctx->rp_anno, ctx->rp_data, ctx->rp_dcur );

    free( ctx->rp_anno );
    free( ctx->rp_data );
    free( ctx->rp_events );

    free( ctx->rp_working );

#ifdef VERBOSE
    printf( "COV_ENTER %.1f\n", ctx->rp_xcov_enter );
    printf( "COV_LEAVE %.1f\n", ctx->rp_xcov_leave );
    printf( "REGIONS %d\n", ( ctx->rp_dcur ) / ( 2 + ctx->inccov ) );
    printf( "MERGED %d\n", ctx->rp_merged );
    printf( "BASES_TOTAL %" PRIu64 "\n", ctx->rp_bases );
    printf( "BASES_REPEAT %" PRIu64 "\n", ctx->rp_repeat_bases );
    printf( "BASES_REPEAT_PERCENT %d%%\n", (int)( ctx->rp_repeat_bases * 100.0 / ctx->rp_bases ) );
#endif
}

static int handler_repeats( void* _ctx, Overlap* ovl, int novl )
{
    RepeatContext* ctx       = (RepeatContext*)_ctx;
    int* rp_events           = ctx->rp_events;
    uint64_t rp_repeat_bases = 0;
    uint64_t rp_merged       = 0;
    int rp_merge_dist        = ctx->rp_merge_dist;
    int alen = DB_READ_LEN( ctx->db, ovl->aread );

    ctx->rp_bases += alen;

    if ( (size_t)(2 * novl) > ctx->rp_emax )
    {
        ctx->rp_emax   = 2.2 * novl + 1000;
        ctx->rp_events = rp_events = (int*)realloc( rp_events, sizeof( int ) * ctx->rp_emax );
    }

    int i;
    int j = 0;
    for ( i = 0; i < novl; i++ )
    {
        // TODO ... check if OVL_DISCARD flag is needed

        if ( ovl[ i ].flags & OVL_DISCARD )
        {
            continue;
        }

        if ( ovl[ i ].aread == ovl[ i ].bread )
        {
            continue;
        }

        if ( ovl[ i ].path.aepos - ovl[ i ].path.abpos < ctx->min_aln_len )
        {
            continue;
        }

        if ( alen < ctx->min_rlen || DB_READ_LEN( ctx->db, ovl[ i ].bread ) < ctx->min_rlen )
        {
            continue;
        }

        rp_events[ j++ ] = ovl[ i ].path.abpos;
        rp_events[ j++ ] = -( ovl[ i ].path.aepos - 1 );
    }

    novl = j / 2;

    qsort( rp_events, 2 * novl, sizeof( int ), cmp_repeats_events );

    int span            = 0;
    int span_leave      = ctx->cov * ctx->rp_xcov_leave;
    int span_enter      = ctx->cov * ctx->rp_xcov_enter;
    int inccov          = ctx->inccov;
    int span_max        = 0;
    int in_repeat       = 0;
    int a               = ovl->aread;

    track_data* rp_working = ctx->rp_working;
    size_t rp_working_max = ctx->rp_working_max;
    size_t rp_working_cur = 0;

    for ( i = 0; i < 2 * novl; i++ )
    {
        if ( rp_events[ i ] < 0 )
        {
            span--;
        }
        else
        {
            span++;

            if ( span > span_max )
            {
                span_max = span;
            }
        }

        if ( in_repeat )
        {
            if ( span < span_leave )
            {
                int rexit = -( rp_events[ i ] );
#ifdef DEBUG
                printf( "repeat <- %d @ %d\n", a, rexit );
#endif

                rp_working[ rp_working_cur++ ] = rexit;

                if ( inccov )
                {
                    rp_working[ rp_working_cur++ ] = span_max;
                }

                in_repeat = 0;
            }
        }
        else
        {
            if ( span > span_enter )
            {
                int rentry = rp_events[ i ];

                {

#ifdef DEBUG
                    printf( "repeat -> %d @ %d\n", a, rentry );
#endif

                    if ( rp_working_cur + 3 >= rp_working_max )
                    {
                        rp_working_max = ctx->rp_working_max = 1.2 * rp_working_cur + 20;
                        ctx->rp_working = rp_working = realloc( rp_working, sizeof( track_data ) * rp_working_max );
                    }

                    if ( rp_working_cur >= (size_t)(inccov + 2)
                         &&
                         rentry - rp_working[ rp_working_cur - ( 1 + inccov ) ] < rp_merge_dist
                       )
                    {
#ifdef DEBUG
                        printf( "  merge\n" );
#endif

                        span_max = rp_working[ rp_working_cur - 1];
                        rp_working_cur -= (inccov + 1);

                        rp_merged++;
                    }
                    else
                    {
                        span_max = 0;
                        rp_working[ rp_working_cur++ ] = rentry;
                    }

                    in_repeat = 1;
                }

            }
        }
    }

    int min_repeat_isolation = ctx->min_repeat_isolation;
    int max_repeat_len = ctx->max_repeat_len;
    track_data* rp_data = ctx->rp_data;
    track_anno* rp_anno = ctx->rp_anno;
    int rp_dcur         = ctx->rp_dcur;
    int trim_b, trim_e;

    if ( ctx->tracktrim != NULL )
    {
        get_trim(ctx->db, ctx->tracktrim, a, &trim_b, &trim_e);

        if ( trim_b == 0 && trim_e == 0 )
        {
            trim_e = DB_READ_LEN(ctx->db, a);
        }
    }
    else
    {
        trim_b = 0;
        trim_e = DB_READ_LEN(ctx->db, a);
    }

    int rp_dcur_prev = rp_dcur;

    if ( rp_dcur + rp_working_cur >= ctx->rp_dmax )
    {
        ctx->rp_dmax = 1.2 * ( rp_dcur + rp_working_cur ) + 20;
        ctx->rp_data = rp_data = (int*)realloc( rp_data, sizeof( int ) * ctx->rp_dmax );
    }

    for ( i = 0 ; i < (int)rp_working_cur ; i += ( inccov + 2 ) )
    {
        int lastend;
        if ( rp_dcur == rp_dcur_prev )
        {
            lastend = trim_b;
        }
        else
        {
            lastend = rp_data[ rp_dcur - (1 + inccov) ];
        }

        int rlen = rp_working[i + 1] - rp_working[i];

        if ( min_repeat_isolation == 0
             ||
             (
                rp_working[i] - lastend > min_repeat_isolation
                &&
                rlen < max_repeat_len
                &&
                trim_e - rp_working[i+1] > min_repeat_isolation
             )
           )
        {
            rp_data[ rp_dcur++ ] = rp_working[i];
            rp_data[ rp_dcur++ ] = rp_working[i + 1];

            if ( inccov )
            {
                rp_data[ rp_dcur++ ] = rp_working[i + 2];
            }

            rp_anno[ a ] += ( inccov + 2 ) * sizeof( track_data );
            rp_repeat_bases += rlen;
        }
    }

#ifdef DEBUG
    if ( ctx->rp_dcur != rp_dcur )
    {
        for ( i = ctx->rp_dcur; i < rp_dcur; i += ( 2 + inccov ) )
        {
            if ( i != ctx->rp_dcur )
            {
                printf( " " );
            }

            if ( inccov )
            {
                printf( "(%d %d @ %d)", rp_data[ i ], rp_data[ i + 1 ], rp_data[ i + 2 ] );
            }
            else
            {
                printf( "(%d %d)", rp_data[ i ], rp_data[ i + 1 ] );
            }
        }

        printf( "\n" );
    }
#endif

    ctx->rp_dcur = rp_dcur;
    ctx->rp_repeat_bases += rp_repeat_bases;
    ctx->rp_merged += rp_merged;

    return 1;
}

static void usage()
{
    printf( "usage: [-hl f] [-t track] [-bcmnorRs n] database input.las\n\n" );

    printf( "Detects repeat elements based on coverage anomalies in reads and creates an annotation track with them.\n\n" );

    printf( "options: -h f  above which multiple of the expected coverage the start of a repeat is reported (%.1f)\n", DEFAULT_RP_XCOV_ENTER );
    printf( "         -l f  below which multiple of the expected coverage a repeat ends (%.1f)\n", DEFAULT_RP_XCOV_LEAVE );

    printf( "         -m n  merge repeats less then n bases apart (%d)\n", DEFAULT_RP_MERGE_DIST );

    printf( "         -o n  only use overlaps longer than n\n" );
    printf( "         -L n  only use reads longer then n\n" );

    printf( "         -c n  expected coverage of the dataset (%d)\n", DEFAULT_COV );
    printf( "         -n n  number of a reads used for coverage estimation (%d)\n", DEFAULT_COV_MAX_READS );

    printf( "         -t track  name of the repeat annotation track (%s)\n", TRACK_REPEATS );
    printf( "         -b n  block number\n" );

    printf( "         -i n  repeat isolation (%d)\n", DEF_ARG_I );
    printf( "         -r n  maximun repeat length (%d)\n", DEF_ARG_R );
    printf( "         -T track  name of the trim track\n");

    printf( "         -s n  annotate regions as repeat if the coverage deviates n stdev from the mean (%.2f)\n", DEF_ARG_S );
}

int main( int argc, char* argv[] )
{
    HITS_DB db;
    PassContext* pctx;
    RepeatContext rctx;
    FILE* fileOvlIn;

    bzero( &rctx, sizeof( RepeatContext ) );
    rctx.db = &db;

    // process arguments

    char* tracktrim = NULL;

    rctx.rp_xcov_enter        = DEFAULT_RP_XCOV_ENTER;
    rctx.rp_xcov_leave        = DEFAULT_RP_XCOV_LEAVE;
    rctx.rp_merge_dist        = DEFAULT_RP_MERGE_DIST;
    rctx.cov                  = DEFAULT_COV;
    rctx.rp_track             = TRACK_REPEATS;
    rctx.rp_block             = 0;
    rctx.inccov               = DEF_ARG_IC;
    rctx.min_aln_len          = DEF_ARG_O;
    rctx.min_rlen             = DEF_ARG_LL;
    rctx.max_repeat_len       = DEF_ARG_R;
    rctx.min_repeat_isolation = DEF_ARG_I;

    int c;

    opterr = 0;

    while ( ( c = getopt( argc, argv, "Ch:i:l:L:m:c:t:T:b:o:r:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'T':
                tracktrim = optarg;
                break;

            case 'i':
                rctx.min_repeat_isolation = atoi( optarg );
                break;

            case 'r':
                rctx.max_repeat_len = atoi( optarg );
                break;

            case 'L':
                rctx.min_rlen = atoi( optarg );
                break;

            case 'o':
                rctx.min_aln_len = atoi( optarg );
                break;

            case 'C':
                rctx.inccov = 1;
                break;

            case 'b':
                rctx.rp_block = atoi( optarg );
                break;

            case 'h':
                rctx.rp_xcov_enter = atof( optarg );
                break;

            case 'l':
                rctx.rp_xcov_leave = atof( optarg );
                break;

            case 'm':
                rctx.rp_merge_dist = atoi( optarg );
                break;

            case 'c':
                rctx.cov = atoi( optarg );
                break;

            case 't':
                rctx.rp_track = optarg;
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

    char* pcPathReadsIn  = argv[ optind++ ];
    char* pcPathOverlaps = argv[ optind++ ];

    if ( ( rctx.min_repeat_isolation == 0 && rctx.max_repeat_len != 0 ) ||
         ( rctx.min_repeat_isolation != 0 && rctx.max_repeat_len == 0 ) )
    {
        fprintf( stderr, "repeat isolation and maximum repeat length must either be both set or neither set\n" );
        exit( 1 );
    }

    if ( rctx.rp_xcov_enter < rctx.rp_xcov_leave )
    {
        fprintf( stderr, "invalid arguments: low %.2f > high %.2f\n", rctx.rp_xcov_leave, rctx.rp_xcov_enter );
        exit( 1 );
    }

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

    Open_DB( pcPathReadsIn, &db );

    if ( tracktrim )
    {
        rctx.tracktrim = track_load(&db, tracktrim );
        if ( rctx.tracktrim == NULL )
        {
            fprintf( stderr, "failed to open track '%s'\n", tracktrim );
            exit(1);
        }
    }
    else if ( rctx.min_repeat_isolation != 0 )
    {
        printf("warning: running repeat detection in isolation mode without trim track\n");
    }

    // pass

    pre_repeats( &rctx );
    pass( pctx, handler_repeats );
    post_repeats( &rctx );

    // cleanup

    pass_free( pctx );

    fclose( fileOvlIn );

    Close_DB( &db );

    return 0;
}
