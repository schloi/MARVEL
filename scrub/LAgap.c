/*******************************************************************************************
 *
 * TODO: break resolution strategies... side with "longest alignment", "most read-mass", ...
 *
 *******************************************************************************************/

#include <assert.h>
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

// argument defaults

#define DEF_ARG_P 0
#define DEF_ARG_S 0
#define DEF_ARG_SS 1
#define DEF_ARG_M 2

// thresholds

#define BIN_SIZE 100
#define MIN_LR 450

// switches

#define VERBOSE
#undef DEBUG_GAPS

// constants

typedef struct
{
    HITS_DB* db;
    HITS_TRACK* trackExclude;

    // statistics

    uint64_t stats_contained;
    uint64_t stats_breaks;
    uint64_t stats_breaks_novl;

    // arguments

    int stitch;
    uint16_t stitch_min_reads;
    int use_read_loader;
    HITS_TRACK* trackTrim;
    int min_spanning;

    // working storage for gap detection

    uint64* rm_bins;
    int rm_maxbins;

    int* rc_left;
    int rc_maxright;

    int* rc_right;
    int rc_maxleft;

    // read trimmer

    TRIM* trim;

    // read loader

    Read_Loader* rl;

} GapContext;

// for getopt()

extern char* optarg;
extern int optind, opterr, optopt;

static int loader_handler( void* _ctx, Overlap* ovl, int novl )
{
    GapContext* ctx = (GapContext*)_ctx;
    Read_Loader* rl = ctx->rl;

    int i;
    for ( i = 0; i < novl; i++ )
    {
        Overlap* o = ovl + i;
        int b      = ovl[ i ].bread;

        int trim_b_left, trim_b_right;
        get_trim( ctx->db, ctx->trackTrim, b, &trim_b_left, &trim_b_right );

        if ( ovl[ i ].flags & OVL_COMP )
        {
            int tmp      = trim_b_left;
            int blen     = DB_READ_LEN( ctx->db, ovl[ i ].bread );
            trim_b_left  = blen - trim_b_right;
            trim_b_right = blen - tmp;
        }

        if ( trim_b_left >= trim_b_right )
        {
            continue;
        }

        if ( intersect( o->path.bbpos, o->path.bepos, trim_b_left, trim_b_right ) &&
             !( o->path.bbpos >= trim_b_left && o->path.bepos <= trim_b_right ) )
        {
            rl_add( rl, ovl[ i ].aread );
            rl_add( rl, ovl[ i ].bread );
        }
    }

    return 1;
}

static int drop_break( Overlap* pOvls, int nOvls, int p1, int p2 )
{
    int max  = 0;
    int left = 0;
    int i;

    for ( i = 0; i < nOvls; i++ )
    {
        if ( pOvls[ i ].flags & OVL_DISCARD )
        {
            continue;
        }

        int len = pOvls[ i ].path.aepos - pOvls[ i ].path.abpos;
        if ( len > max )
        {
            max = len;

            if ( pOvls[ i ].path.abpos < p1 )
            {
                left = 1;
            }
            else
            {
                left = 0;
            }
        }
    }

    if ( max == 0 )
    {
        return 0;
    }

    int dropped = 0;

    if ( !left )
    {
#ifdef DEBUG_GAPS
        printf( "DROP L @ %5d..%5d\n", p1, p2 );
#endif

        for ( i = 0; i < nOvls; i++ )
        {
            if ( pOvls[ i ].path.abpos < p1 )
            {
                if ( !( pOvls[ i ].flags & OVL_DISCARD ) )
                {
                    dropped++;
                }

                pOvls[ i ].flags |= OVL_DISCARD | OVL_GAP;
            }
        }
    }
    else
    {
#ifdef DEBUG_GAPS
        printf( "DROP R @ %5d..%5d\n", p1, p2 );
#endif

        for ( i = 0; i < nOvls; i++ )
        {
            if ( pOvls[ i ].flags & OVL_DISCARD )
                continue;

            if ( pOvls[ i ].path.aepos > p2 )
            {
                if ( !( pOvls[ i ].flags & OVL_DISCARD ) )
                {
                    dropped++;
                }

                pOvls[ i ].flags |= OVL_DISCARD | OVL_GAP;
            }
        }
    }

    return dropped;
}

static int stitchable( Overlap* pOvls, int n, int fuzz, int beg, int end )
{
    if ( n < 2 )
    {
        return 0;
    }

    int t = 0;
    int k, b;
    int ab2, ae1;
    int bb2, be1;

    const int ignore_mask = OVL_TEMP | OVL_CONT | OVL_TRIM | OVL_STITCH;

    int i;
    for ( i = 0; i < n; i++ )
    {
        if ( pOvls[ i ].flags & ignore_mask )
        {
            continue;
        }

        b = pOvls[ i ].bread;

        ae1 = pOvls[ i ].path.aepos;
        be1 = pOvls[ i ].path.bepos;

        for ( k = i + 1; k < n && pOvls[ k ].bread == b; k++ )
        {
            if ( pOvls[ k ].flags & ignore_mask ||
                 ( pOvls[ i ].flags & OVL_COMP ) != ( pOvls[ k ].flags & OVL_COMP ) )
            {
                continue;
            }

            ab2 = pOvls[ k ].path.abpos;
            bb2 = pOvls[ k ].path.bbpos;

            int deltaa = abs( ae1 - ab2 );
            int deltab = abs( be1 - bb2 );

            if ( deltaa < fuzz && deltab < fuzz )
            {
                if ( pOvls[ i ].path.abpos < beg - MIN_LR && pOvls[ k ].path.aepos > end + MIN_LR )
                {
                    t++;

                    // printf("stitch using b %d\n", b);
                }
            }
        }
    }

    return t;
}

static int ovl_intersect( Overlap* a, Overlap* b )
{
    return intersect( a->path.abpos, a->path.aepos, b->path.abpos, b->path.aepos );
}

static int handle_gaps_and_breaks( GapContext* ctx, Overlap* ovls, int novl )
{
    int stitch               = ctx->stitch;
    int stitch_min_reads     = ctx->stitch_min_reads;
    uint64_t min_spanning    = ctx->min_spanning;
    HITS_TRACK* trackExclude = ctx->trackExclude;

    bzero( ctx->rm_bins, sizeof( uint64 ) * ctx->rm_maxbins );

    int trim_b = INT_MAX;
    int trim_e = 0;

    int i;
    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl = ovls + i;

        if ( ( ovl->flags & OVL_DISCARD ) ||
             ( ovl->aread == ovl->bread ) )
        {
            continue;
        }

        int j;
        for ( j = i + 1; j < novl && ovls[ j ].bread == ovl->bread; j++ )
        {
            if ( ovl_intersect( ovl, ovls + j ) )
            {
                if ( ovl->path.aepos - ovl->path.abpos < ovls[ j ].path.aepos - ovls[ j ].path.abpos )
                {
                    ovl->flags |= OVL_TEMP;
                }
                else
                {
                    ovls[ j ].flags |= OVL_TEMP;
                }
            }
        }

        if ( ovl->flags & OVL_TEMP )
        {
            continue;
        }

        int b = ( ovl->path.abpos + MIN_LR ) / BIN_SIZE;
        int e = ( ovl->path.aepos - MIN_LR ) / BIN_SIZE;

        trim_b = MIN( ovl->path.abpos, trim_b );
        trim_e = MAX( ovl->path.aepos, trim_e );

        while ( b < e )
        {
            ctx->rm_bins[ b ]++;
            b++;
        }
    }

    if ( trim_b >= trim_e )
    {
        return 1;
    }

    // printf("trim %d..%d\n", trim_b, trim_e);

    int b = ( trim_b + MIN_LR ) / BIN_SIZE;
    int e = ( trim_e - MIN_LR ) / BIN_SIZE;

    int beg = -1;

    while ( b < e )
    {
        if ( ctx->rm_bins[ b ] >= min_spanning )
        {
            if ( beg != -1 )
            {
                int breakb = ( beg - 1 ) * BIN_SIZE;
                int breake = ( b + 1 ) * BIN_SIZE;

#ifdef DEBUG_GAPS
                printf( "READ %7d BREAK %3d..%3d ", ovls->aread, beg, b );
#endif
                // break covered by track interval

                int skip = 0;

                if ( trackExclude )
                {
                    track_anno* ta = trackExclude->anno;
                    track_data* td = trackExclude->data;

                    track_anno tab = ta[ ovls->aread ] / sizeof( track_data );
                    track_anno tae = ta[ ovls->aread + 1 ] / sizeof( track_data );

                    int masked = 0;

                    track_data maskb, maske;

                    while ( tab < tae )
                    {
                        maskb = td[ tab ];
                        maske = td[ tab + 1 ];

                        if ( breakb > maskb && breake < maske )
                        {
                            masked = 1;
                            break;
                        }

                        tab += 2;
                    }

                    if ( masked )
                    {
#ifdef DEBUG_GAPS
                        printf( " MASKED %5d..%5d\n", maskb, maske );
#endif
                        skip = 1;
                    }
                }

                if ( !skip && stitch >= stitch_min_reads )
                {
                    int nstitch = stitchable( ovls, novl, stitch, breakb, breake );

                    if ( nstitch )
                    {
#ifdef DEBUG_GAPS
                        printf( " STITCHABLE using %d\n", nstitch );
#endif
                        skip = 1;
                    }
                }

                if ( !skip )
                {
#ifdef DEBUG_GAPS
                    printf( "\n" );
#endif
                    ctx->stats_breaks += 1;
                    ctx->stats_breaks_novl += drop_break( ovls, novl, breakb, breake );
                }

                beg = -1;
            }
        }
        else
        {
            if ( beg == -1 )
            {
                beg = b;
            }
        }

        b++;
    }

    if ( beg != -1 )
    {
        int breakb = ( beg - 1 ) * BIN_SIZE;
        int breake = ( b + 1 ) * BIN_SIZE;

        ctx->stats_breaks += 1;
        ctx->stats_breaks_novl += drop_break( ovls, novl, breakb, breake );
    }

    return 1;
}

static int contained( int ab, int ae, int bb, int be )
{
    if ( ab >= bb && ae <= be )
    {
        return 1;
    }

    if ( bb >= ab && be <= ae )
    {
        return 2;
    }

    return 0;
}

static int drop_containments( GapContext* ctx, Overlap* ovl, int novl )
{
    if ( novl < 2 )
    {
        return 1;
    }

    int i;
    int ab1, ab2, ae1, ae2;
    int bb1, bb2, be1, be2;

    for ( i = 0; i < novl; i++ )
    {
        int bread = ovl[ i ].bread;

        if ( ( ovl[ i ].flags & OVL_DISCARD ) ||
             ovl[ i ].aread == bread )
        {
            continue;
        }

        int blen = DB_READ_LEN( ctx->db, bread );

        ab1 = ovl[ i ].path.abpos;
        ae1 = ovl[ i ].path.aepos;

        if ( ovl[ i ].flags & OVL_COMP )
        {
            bb1 = blen - ovl[ i ].path.bepos;
            be1 = blen - ovl[ i ].path.bbpos;
        }
        else
        {
            bb1 = ovl[ i ].path.bbpos;
            be1 = ovl[ i ].path.bepos;
        }

        int k;
        for ( k = i + 1; k < novl; k++ )
        {
            if ( ovl[ k ].flags & OVL_DISCARD )
            {
                continue;
            }

            if ( ovl[ k ].bread != bread )
            {
                break;
            }

            ab2 = ovl[ k ].path.abpos;
            ae2 = ovl[ k ].path.aepos;

            if ( ovl[ k ].flags & OVL_COMP )
            {
                bb2 = blen - ovl[ k ].path.bepos;
                be2 = blen - ovl[ k ].path.bbpos;
            }
            else
            {
                bb2 = ovl[ k ].path.bbpos;
                be2 = ovl[ k ].path.bepos;
            }

            int cont = contained( ab1, ae1, ab2, ae2 );
            if ( cont && contained( bb1, be1, bb2, be2 ) )
            {
#ifdef VERBOSE_CONTAINMENT
                printf( "CONTAINMENT %8d @ %5d..%5d -> %8d @ %5d..%5d\n"
                        "                       %5d..%5d -> %8d @ %5d..%5d\n",
                        a, ab1, ae1, ovl[ i ].bread, bb1, be1,
                        ab2, ae2, ovl[ k ].bread, bb2, be2 );
#endif

                if ( cont == 1 )
                {
                    ovl[ i ].flags |= OVL_DISCARD | OVL_CONT;
                    ctx->stats_contained++;
                    break;
                }
                else if ( cont == 2 )
                {
                    ovl[ k ].flags |= OVL_DISCARD | OVL_CONT;
                    ctx->stats_contained++;
                }
            }
        }
    }

    return 1;
}

static void gaps_pre( PassContext* pctx, GapContext* ctx )
{
#ifdef VERBOSE
    printf( ANSI_COLOR_GREEN "PASS gaps" ANSI_COLOR_RESET "\n" );
#endif

    ctx->rm_maxbins = ( DB_READ_MAXLEN( ctx->db ) + BIN_SIZE ) / BIN_SIZE;
    ctx->rm_bins    = malloc( sizeof( uint64 ) * ctx->rm_maxbins );

    ctx->rc_left    = NULL;
    ctx->rc_maxleft = 0;

    ctx->rc_right    = NULL;
    ctx->rc_maxright = 0;

    // trim

    ctx->trim = trim_init( ctx->db, pctx->twidth, ctx->trackTrim, ctx->rl );
}

static void gaps_post( GapContext* ctx )
{
#ifdef VERBOSE
    if ( ctx->trackTrim )
    {
        printf( "%13lld of %13lld overlaps trimmed\n", ctx->trim->nTrimmedOvls, ctx->trim->nOvls );
        printf( "%13lld of %13lld bases trimmed\n", ctx->trim->nTrimmedBases, ctx->trim->nOvlBases );
    }

    printf( "dropped %" PRIu64 " containments\n", ctx->stats_contained );
    printf( "dropped %" PRIu64 " overlaps in %" PRIu64 " break/gaps\n", ctx->stats_breaks_novl, ctx->stats_breaks );
#endif

    free( ctx->rm_bins );

    free( ctx->rc_left );
    free( ctx->rc_right );

    if ( ctx->trackTrim )
    {
        trim_close( ctx->trim );
    }
}

static int gaps_handler( void* _ctx, Overlap* ovl, int novl )
{
    GapContext* ctx = (GapContext*)_ctx;

    // trim
    if ( ctx->trackTrim )
    {
        int i;
        for ( i = 0; i < novl; i++ )
        {
            if ( ovl[i].flags & OVL_DISCARD )
            {
                continue ;
            }

            trim_overlap( ctx->trim, ovl + i );
        }
    }

    drop_containments( ctx, ovl, novl );

    handle_gaps_and_breaks( ctx, ovl, novl );

    return 1;
}

static void usage( FILE* fout, const char* app )
{
    fprintf( fout, "usage: %s [-Lp] [-msS n] [-et track] database input.las output.las\n\n", app );

    fprintf( fout, "Detect gaps, regions not spanned by any reads, and discards all alignments on one side of the gap.\n\n" );

    fprintf( fout, "options: -s n  don't count alignments that would be stitchable with a maximum distance of n (%d)\n", DEF_ARG_S );
    fprintf( fout, "         -S n  minimum number of reads needed for stitching (%d)\n", DEF_ARG_SS );
    fprintf( fout, "         -p  purge alignments tagged as discarded\n" );
    fprintf( fout, "         -e track  exclude gaps located in the intervals of the track\n" );
    fprintf( fout, "         -t track  trim overlaps before gap detection\n" );
    fprintf( fout, "         -L two-pass processing with read caching\n" );
    fprintf( fout, "         -m n  minimum number of reads need to span a region to not make it a gap (%d)\n", DEF_ARG_M );
}

int main( int argc, char* argv[] )
{
    HITS_DB db;
    PassContext* pctx;
    FILE* fileOvlIn;
    FILE* fileOvlOut;
    GapContext gctx;
    char* app = argv[ 0 ];

    // process arguments
    int arg_purge       = DEF_ARG_P;
    char* arg_trimTrack = NULL;
    char* excludeTrack  = NULL;
    // char* repeatTrack = NULL;

    bzero( &gctx, sizeof( GapContext ) );

    gctx.stitch           = DEF_ARG_S;
    gctx.stitch_min_reads = DEF_ARG_SS;
    gctx.min_spanning     = DEF_ARG_M;

    opterr = 0;

    int c;
    while ( ( c = getopt( argc, argv, "e:Lm:ps:S:t:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'e':
                excludeTrack = optarg;
                break;

            case 'L':
                gctx.use_read_loader = 1;
                break;

            case 'm':
                gctx.min_spanning = strtol( optarg, NULL, 10 );
                break;

            case 'p':
                arg_purge = 1;
                break;

            case 's':
                gctx.stitch = atoi( optarg );
                break;

            case 'S':
                gctx.stitch_min_reads = atoi( optarg );
                break;

            case 't':
                arg_trimTrack = optarg;
                break;

            default:
                usage( stdout, app );
                exit( 1 );
        }
    }

    if ( gctx.min_spanning < 0 )
    {
        fprintf( stderr, "-m must be positive\n" );
        exit( 1 );
    }

    if ( argc - optind < 3 )
    {
        usage( stdout, app );
        exit( 1 );
    }

    char* pcPathReadsIn     = argv[ optind++ ];
    char* pcPathOverlapsIn  = argv[ optind++ ];
    char* pcPathOverlapsOut = argv[ optind++ ];

    if ( ( fileOvlIn = fopen( pcPathOverlapsIn, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open input track '%s'\n", pcPathOverlapsIn );
        exit( 1 );
    }

    if ( ( fileOvlOut = fopen( pcPathOverlapsOut, "w" ) ) == NULL )
    {
        fprintf( stderr, "could not open output track '%s'\n", pcPathOverlapsOut );
        exit( 1 );
    }

    if ( Open_DB( pcPathReadsIn, &db ) )
    {
        fprintf( stderr, "could not open database '%s'\n", pcPathReadsIn );
        exit( 1 );
    }

    if ( excludeTrack )
    {
        gctx.trackExclude = track_load( &db, excludeTrack );

        if ( !gctx.trackExclude )
        {
            fprintf( stderr, "could not open track '%s'\n", excludeTrack );
            exit( 1 );
        }
    }

    gctx.db = &db;

    if ( arg_trimTrack != NULL )
    {
        gctx.trackTrim = track_load( gctx.db, arg_trimTrack );
        if ( !gctx.trackTrim )
        {
            fprintf( stderr, "could not open track '%s'\n", arg_trimTrack );
            exit( 1 );
        }

        if ( gctx.use_read_loader )
        {
            gctx.rl = rl_init( &db, 1 );

            pctx = pass_init( fileOvlIn, NULL );

            pctx->data       = &gctx;
            pctx->split_b    = 1;
            pctx->load_trace = 0;

            pass( pctx, loader_handler );
            rl_load_added( gctx.rl );
            pass_free( pctx );
        }
    }
    else if ( gctx.use_read_loader )
    {
        fprintf( stderr, "read loader requires a trim track\n" );
        exit( 1 );
    }

    pctx = pass_init( fileOvlIn, fileOvlOut );

    pctx->split_b         = 0;
    pctx->load_trace      = 1;
    pctx->unpack_trace    = ( gctx.trackTrim == NULL ? 0 : 1 );
    pctx->data            = &gctx;
    pctx->write_overlaps  = 1;
    pctx->purge_discarded = arg_purge;

    gaps_pre( pctx, &gctx );

    pass( pctx, gaps_handler );

    gaps_post( &gctx );

    // cleanup

    Close_DB( &db );

    if ( gctx.use_read_loader )
    {
        rl_free( gctx.rl );
    }

    pass_free( pctx );

    fclose( fileOvlIn );
    fclose( fileOvlOut );

    return 0;
}
