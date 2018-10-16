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
#include "lib/utils.h"

#include "dalign/align.h"
#include "db/DB.h"

// constants

#define BINSIZE_COVERAGE 100

#define MAX_COVERAGE 100        // max for coverage histogram
#define MIN_OVERLAP_GROUPS 10 // min number of groups for coverage estimate

// #define EDGE_TAGGING_DIST 1000 // max distance of the repeat to the repeat ends
// #define EDGE_TAGGING_FUZZ 200  // number of wiggle bases for alignment termination at repeat ends

#define DEFAULT_RP_XCOV_ENTER 2.0
#define DEFAULT_RP_XCOV_LEAVE 1.7
#define DEFAULT_RP_MERGE_DIST -1
#define DEFAULT_COV -1
#define DEFAULT_COV_MAX_READS -1

#define DEF_ARG_IC 0
#define DEF_ARG_O 0
#define DEF_ARG_LL 0

#define DEF_ARG_R 0
#define DEF_ARG_RR 0

// toggles

#define VERBOSE
#undef DEBUG

// macros

typedef struct
{
    HITS_DB* db;

    int cov;      // expected/estimated coverage
    int avg_rlen; // average read length
    int inccov;   // include coverage in the resulting track
    int min_aln_len;
    int min_rlen;
    int min_repeat_len;
    int max_repeat_len;

    // coverage pass

    uint64_t cov_histo[ MAX_COVERAGE ];
    uint64_t cov_bases;
    uint64_t cov_inactive_bases;
    char* cov_read_active;

    uint64_t* cov_binned;

    int cov_areads;
    int cov_max_areads;

    uint64_t bases;
    uint64_t reads;

    // repeat pass

    int rp_emax;
    int rp_dmax;
    int rp_dcur;
    track_data* rp_data;
    track_anno* rp_anno;
    int* rp_events;

    int rp_merge_dist;

    int rp_merged;
    uint64_t rp_bases;
    uint64_t rp_repeat_bases;

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
        cmp = ( *e1 ) - ( *e2 );
    }

    return cmp;
}

static void pre_coverage( RepeatContext* ctx )
{
#ifdef VERBOSE
    printf( ANSI_COLOR_GREEN "PASS estimate coverage\n" ANSI_COLOR_RESET );
#endif

    ctx->cov_read_active = malloc( DB_READ_MAXLEN( ctx->db ) );
    ctx->cov_binned = malloc( sizeof(uint64_t) * ( DB_READ_MAXLEN(ctx->db) + BINSIZE_COVERAGE ) / BINSIZE_COVERAGE );

    bzero( ctx->cov_histo, sizeof( int64_t ) * MAX_COVERAGE );
}

static void post_coverage( RepeatContext* ctx )
{
    int max = 0;
    uint64_t cov = ctx->cov_histo[ 0 ];

    ctx->avg_rlen = ctx->bases / ctx->reads;

    int j;
    for ( j = 1; j < MAX_COVERAGE; j++ )
    {
        if ( cov < ctx->cov_histo[ j ] )
        {
            cov = ctx->cov_histo[ j ];
            max = j;
        }
    }

    ctx->cov = max;
    free( ctx->cov_read_active );
    free( ctx->cov_binned );

#ifdef VERBOSE
    for ( j = 0; j < MAX_COVERAGE; j++ )
    {
        printf( "COV %d READS %" PRIu64 "\n", j, ctx->cov_histo[ j ] );
    }

    printf( "MAX %d\n", max );
    printf( "INACTIVE %" PRIu64 " (%d%%) OF %" PRIu64 "\n",
            ctx->cov_inactive_bases, (int)( 100.0 * ctx->cov_inactive_bases / ctx->cov_bases ), ctx->cov_bases );
    printf( "AVG_RLEN %d\n", ctx->avg_rlen );
#endif
}

static int handler_coverage( void* _ctx, Overlap* ovls, int novl )
{
    RepeatContext* ctx = (RepeatContext*)_ctx;
    char* cov_read_active = ctx->cov_read_active;
    uint64_t* cov_binned = ctx->cov_binned;
    uint64_t* cov_histo = ctx->cov_histo;
    int aread = ovls->aread;

    int ovlArlen = DB_READ_LEN( ctx->db, aread );
    ctx->bases += ovlArlen;
    ctx->reads++;

    int i;
    int64_t cov;
    int64_t bases = 0;

    bzero( cov_read_active, ovlArlen );
    bzero( cov_binned, sizeof(uint64_t) * ( DB_READ_MAXLEN(ctx->db) + BINSIZE_COVERAGE ) / BINSIZE_COVERAGE );

    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl = ovls + i;

        if ( !( DB_READ_FLAGS( ctx->db, ovl->bread ) & DB_BEST ) || ( ovl->flags & OVL_DISCARD ) )
        {
            continue;
        }

        if ( aread == ovl->bread )
            continue;

        bases += ovl->path.bepos - ovl->path.bbpos;

        memset( cov_read_active + ovl->path.abpos, 1, ovl->path.aepos - ovl->path.abpos );

        int j;
        for ( j = ovl->path.abpos / BINSIZE_COVERAGE;
              j < ovl->path.aepos / BINSIZE_COVERAGE;
              j += 1)
        {
            cov_binned[j] += 1;
        }

    }

    int active = 0;
    int bin_active = 0;
    for ( i = 0; i < ovlArlen; i++ )
    {
        if ( i % BINSIZE_COVERAGE == 0 && i > 0 )
        {
            if ( bin_active == BINSIZE_COVERAGE )
            {
                cov = cov_binned[ i / BINSIZE_COVERAGE - 1];

                if ( cov < MAX_COVERAGE )
                {
                    cov_histo[ cov ] += 1;
                }
            }

            bin_active = 0;
        }

        bin_active += cov_read_active[ i ];
        active += cov_read_active[ i ];
    }

    /*
    if ( active > 0 )
    {
        cov = bases / active;
    }
    else
    {
        cov = 0;
    }

    if ( cov < MAX_COVERAGE )
    {
        cov_histo[ cov ]++;
    }
    */

    ctx->cov_bases += ovlArlen;
    ctx->cov_inactive_bases += ovlArlen - active;

    ctx->cov_areads++;

    printf("%" PRIu64 " %" PRIu64 "\n", ctx->cov_bases, ctx->cov_inactive_bases);

    if ( ctx->cov_areads > ctx->cov_max_areads )
    {
        return 0;
    }

    return 1;
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

    // if (ovl->aread > 1000) return 0;

    int alen = DB_READ_LEN( ctx->db, ovl->aread );
    ctx->rp_bases += alen;

    if ( 2 * novl > ctx->rp_emax )
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
    int a               = ovl->aread;
    int inccov          = ctx->inccov;
    track_data* rp_data = ctx->rp_data;
    track_anno* rp_anno = ctx->rp_anno;
    int span_max        = 0;
    int in_repeat       = 0;
    int rp_dcur         = ctx->rp_dcur;

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
#ifdef DEBUG
                printf( "repeat <- %d @ %d\n", a, -( rp_events[ i ] ) );
#endif

                int rlen = ( -1 ) * rp_events[ i ] - rp_data[ rp_dcur - 1 ];

                if ( ( ctx->min_repeat_len == 0 || rlen > ctx->min_repeat_len ) &&
                     ( ctx->max_repeat_len == 0 || rlen < ctx->max_repeat_len ) )
                {
                    // rp_anno[ a ] += ( inccov + 1 ) * sizeof( track_data );
                    rp_anno[ a ] += ( inccov + 2 ) * sizeof( track_data );

                    rp_data[ rp_dcur++ ] = -( rp_events[ i ] );

                    rp_repeat_bases += rlen;

                    if ( inccov )
                    {
                        rp_data[ rp_dcur++ ] = span_max;
                    }
                }
                else
                {
                    rp_dcur -= 1;
                }

                in_repeat = 0;
            }
        }
        else
        {
            if ( span > span_enter )
            {
#ifdef DEBUG
                printf( "repeat -> %d @ %d\n", a, rp_events[ i ] );
#endif

                if ( rp_dcur + 3 >= ctx->rp_dmax )
                {
                    ctx->rp_dmax = 1.2 * rp_dcur + 20;
                    ctx->rp_data = rp_data = (int*)realloc( rp_data, sizeof( int ) * ctx->rp_dmax );
                }

                if ( rp_dcur - ctx->rp_dcur >= ( inccov + 2 ) &&
                     rp_events[ i ] - rp_data[ rp_dcur - ( 1 + inccov ) ] < rp_merge_dist )
                {
#ifdef DEBUG
                    printf( "  merge\n" );
#endif

                    span_max = rp_data[ rp_dcur - 1 ];
                    rp_dcur -= ( inccov + 1 );
                    // rp_anno[ a ] -= 2 * sizeof( track_data );

                    rp_merged++;
                }
                else
                {
                    span_max = 0;
                    // rp_anno[ a ] += 1 * sizeof( track_data );
                    rp_data[ rp_dcur++ ] = rp_events[ i ];
                }

                in_repeat = 1;
            }
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

    // TODO: CHECHK --- predates repeat homogenization and is now obsolete
    /*
    for ( i = ctx->rp_dcur; i < rp_dcur; i += ( 2 + inccov ) )
    {
        int rb = rp_data[ i ];
        int re = rp_data[ i + 1 ];

        if ( rb > 0 && rb < EDGE_TAGGING_DIST && re < alen - EDGE_TAGGING_DIST )
        {
            int support = 0;

            for ( j = 0; j < novl; j++ )
            {
                Overlap* o = ovl + j;

                if ( o->path.aepos > re - EDGE_TAGGING_FUZZ &&
                     o->path.aepos < re + EDGE_TAGGING_FUZZ &&
                     o->path.abpos == 0 )
                {
                    support += 1;

                    if ( support > 2 )
                    {
#ifdef DEBUG
                        printf("%7d repeat %5d..%5d extended to %5d..%5d\n", a, rp_data[i], rp_data[i+1], 0, rp_data[i+1]);
#endif
                        rp_data[ i ] = 0;
                        break;
                    }
                }
            }
        }

        if ( re < alen - 1 && re > alen - EDGE_TAGGING_DIST && rb > EDGE_TAGGING_DIST )
        {
            int support = 0;

            for ( j = 0; j < novl && support <= 2; j++ )
            {
                Overlap* o = ovl + j;

                if ( o->path.abpos > rb - EDGE_TAGGING_FUZZ &&
                     o->path.abpos < rb + EDGE_TAGGING_FUZZ &&
                     o->path.aepos == alen )
                {
                    support += 1;

                    if ( support > 2 )
                    {
#ifdef DEBUG
                        printf("%7d repeat %5d..%5d extended to %5d..%5d\n", a, rp_data[i], rp_data[i+1], rp_data[i], alen);
#endif
                        rp_data[ i + 1 ] = alen;
                        break;
                    }
                }
            }
        }
    }
    */

    ctx->rp_dcur = rp_dcur;
    ctx->rp_repeat_bases += rp_repeat_bases;
    ctx->rp_merged += rp_merged;

    return 1;
}

static void usage()
{
    printf( "usage: [-hl f] [-t track] [-bcmnorR n] database input.las\n\n" );

    printf( "Detects repeat elements based on coverage anomalies in reads and creates an annotation track with them.\n\n" );

    printf( "options: -h f  above which multiple of the expected coverage the start of a repeat is reported (%.1f)\n", DEFAULT_RP_XCOV_ENTER );
    printf( "         -l f  below which multiple of the expected coverage a repeat ends (%.1f)\n", DEFAULT_RP_XCOV_LEAVE );

    printf( "         -m n  merge repeats less then n bases apart (%d)\n", DEFAULT_RP_MERGE_DIST );

    printf( "         -o n  only use overlaps longer than n\n" );
    printf( "         -L n  only use reads longer then n\n" );

    printf( "         -c n  expected coverage of the dataset. -1 auto-detect. (%d)\n", DEFAULT_COV );
    printf( "         -n n  number of a reads used for coverage estimation (%d)\n", DEFAULT_COV_MAX_READS );

    printf( "         -t track  name of the repeat annotation track (%s)\n", TRACK_REPEATS );
    printf( "         -b n  block number\n" );

    printf( "         -r n  minimum repeat length (%d)\n", DEF_ARG_R );
    printf( "         -R n  maximum repeat length (%d)\n", DEF_ARG_RR );
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

    rctx.rp_xcov_enter  = DEFAULT_RP_XCOV_ENTER;
    rctx.rp_xcov_leave  = DEFAULT_RP_XCOV_LEAVE;
    rctx.rp_merge_dist  = DEFAULT_RP_MERGE_DIST;
    rctx.cov            = DEFAULT_COV;
    rctx.cov_max_areads = DEFAULT_COV_MAX_READS;
    rctx.rp_track       = TRACK_REPEATS;
    rctx.rp_block       = 0;
    rctx.inccov         = DEF_ARG_IC;
    rctx.min_aln_len    = DEF_ARG_O;
    rctx.min_rlen       = DEF_ARG_LL;
    rctx.min_repeat_len = DEF_ARG_R;
    rctx.max_repeat_len = DEF_ARG_RR;

    int c;

    opterr = 0;

    while ( ( c = getopt( argc, argv, "Ch:l:L:m:c:n:t:b:o:r:R:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'r':
                rctx.min_repeat_len = atoi( optarg );
                break;

            case 'R':
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

            case 'n':
                rctx.cov_max_areads = atoi( optarg );
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

    if ( rctx.rp_xcov_enter < rctx.rp_xcov_leave )
    {
        fprintf( stderr, "invalid arguments: low %.2f > high %.2f\n", rctx.rp_xcov_leave, rctx.rp_xcov_enter );
        exit( 1 );
    }

    if ( rctx.cov_max_areads != -1 && rctx.cov_max_areads < MIN_OVERLAP_GROUPS )
    {
        fprintf( stderr, "invalid arguments: number of overlap groups tested should be larger than %d\n", MIN_OVERLAP_GROUPS );
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

    if ( rctx.cov_max_areads == -1 )
    {
        rctx.cov_max_areads = db.nreads;
    }

    // passes

    if ( rctx.cov <= 0 )
    {
        pre_coverage( &rctx );
        pass( pctx, handler_coverage );
        post_coverage( &rctx );

        if ( rctx.cov <= 0 )
        {
            fprintf( stderr, "ERROR: coverage estimation resulted in %d\n", rctx.cov );
            fprintf( stderr, "       bypass estimation using the -c <coverage> argument\n" );

            exit( 1 );
        }
    }

    pre_repeats( &rctx );
    pass( pctx, handler_repeats );
    post_repeats( &rctx );

    // cleanup

    pass_free( pctx );

    fclose( fileOvlIn );

    Close_DB( &db );

    return 0;
}
