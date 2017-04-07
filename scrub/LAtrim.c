/*******************************************************************************************
 *
 *  applies the trim track to an overlap file and realigns the trimmed ends to establish
 *  new trace points, bbpos and bepos
 *
 *  Author  :  MARVEL Team
 *
 *  Date    :  November 2014
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
#include "lib/read_loader.h"
#include "lib/tracks.h"
#include "lib/trim.h"
#include "lib/utils.h"

#include "dalign/align.h"
#include "db/DB.h"

// arguments

#define DEF_ARG_P 0
#define DEF_ARG_T TRACK_TRIM

// switches

#define VERBOSE

// trim context

typedef struct
{
    HITS_DB* db; // database
    HITS_TRACK* trackTrim;

    Read_Loader* rl;

    TRIM* trim;

} TrimContext;

// for getopt()

extern char* optarg;
extern int optind, opterr, optopt;

static void trim_pre( PassContext* pctx, TrimContext* tctx, Read_Loader* rl )
{
#ifdef VERBOSE
    printf( ANSI_COLOR_GREEN "PASS trim\n" ANSI_COLOR_RESET );
#endif

    tctx->trim = trim_init( tctx->db, pctx->twidth, tctx->trackTrim, rl );
}

static void trim_post( TrimContext* tctx, int verbose )
{
    if ( verbose )
    {
        printf( "nOvls    : %13lld nTrimOvls : %13lld\n", tctx->trim->nOvls, tctx->trim->nTrimmedOvls );
        printf( "nOvlBases: %13lld nTrimBases: %13lld\n", tctx->trim->nOvlBases, tctx->trim->nTrimmedBases );
    }

    trim_close( tctx->trim );
}

static int trim_handler( void* _ctx, Overlap* ovl, int novl )
{
    TrimContext* ctx = (TrimContext*)_ctx;

    int i;
    for ( i = 0; i < novl; i++ )
    {
        trim_overlap( ctx->trim, ovl + i );
    }

    return 1;
}

static int loader_handler( void* _ctx, Overlap* ovl, int novl )
{
    TrimContext* ctx = (TrimContext*)_ctx;
    Read_Loader* rl  = ctx->rl;

    int i;
    for ( i = 0; i < novl; i++ )
    {
        int b = ovl[ i ].bread;

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

        int bbt = MAX( trim_b_left, ovl[ i ].path.bbpos );
        int bet = MIN( trim_b_right, ovl[ i ].path.bepos );

        if ( bbt >= bet )
        {
            continue;
        }

        if ( bbt == ovl[ i ].path.bbpos && bet == ovl[ i ].path.bepos )
        {
            continue;
        }

        bbt = MAX( trim_b_left, ovl[ i ].path.bbpos );
        bet = MIN( trim_b_right, ovl[ i ].path.bepos );

        if ( bbt < bet && ( bbt != ovl[ i ].path.bbpos || bet != ovl[ i ].path.bepos ) )
        {
            rl_add( rl, ovl[ i ].aread );
            rl_add( rl, ovl[ i ].bread );

            continue;
        }

        int bepos = ovl[ i ].path.bepos;

        if ( bepos > bet )
        {
            rl_add( rl, ovl[ i ].aread );
            rl_add( rl, ovl[ i ].bread );
        }
    }

    return 1;
}

static void usage()
{
    fprintf( stderr, "usage:  [-vpL] [-t <track>] <db> <overlaps.in> <overlaps.out>\n" );
    fprintf( stderr, "options: -v ... verbose\n" );
    fprintf( stderr, "         -p ... purge discarded overlaps\n" );
    fprintf( stderr, "         -t ... trim track name (default: %s)\n", DEF_ARG_T );
    fprintf( stderr, "         -L ... two pass processing with read caching\n");
}

int main( int argc, char* argv[] )
{
    HITS_DB db;
    FILE* fileOvlIn;
    FILE* fileOvlOut;
    PassContext* pctx;
    TrimContext tctx;

    // process arguments

    tctx.db         = &db;
    char* arg_track = DEF_ARG_T;
    int arg_purge   = DEF_ARG_P;
    int arg_verbose = 0;
    int arg_rloader = 0;

    int c;

    opterr = 0;

    while ( ( c = getopt( argc, argv, "vpLt:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'p':
                arg_purge = 1;
                break;

            case 'v':
                arg_verbose = 1;
                break;

            case 'L':
                arg_rloader = 1;
                break;

            case 't':
                arg_track = optarg;
                break;

            default:
                usage();
                exit( 1 );
        }
    }

    if ( argc - optind != 3 )
    {
        usage();
        exit( 1 );
    }

    char* pcPathReadsIn     = argv[ optind++ ];
    char* pcPathOverlapsIn  = argv[ optind++ ];
    char* pcPathOverlapsOut = argv[ optind++ ];

    if ( Open_DB( pcPathReadsIn, &db ) )
    {
        fprintf( stderr, "could not open database '%s'\n", pcPathReadsIn );
        exit( 1 );
    }

    if ( ( fileOvlIn = fopen( pcPathOverlapsIn, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open '%s'\n", pcPathOverlapsIn );
        exit( 1 );
    }

    if ( ( fileOvlOut = fopen( pcPathOverlapsOut, "w" ) ) == NULL )
    {
        fprintf( stderr, "could not open '%s'\n", pcPathOverlapsOut );
        exit( 1 );
    }

    tctx.trackTrim = track_load( &db, arg_track );
    if ( tctx.trackTrim == NULL )
    {
        fprintf( stderr, "could not open trim track '%s'\n", arg_track );
        exit( 1 );
    }

    tctx.db = &db;

    if ( arg_rloader )
    {
        tctx.rl = rl_init( &db, 1 );

        pctx = pass_init( fileOvlIn, NULL );

        pctx->data       = &tctx;
        pctx->split_b    = 1;
        pctx->load_trace = 0;

        pass( pctx, loader_handler );
        rl_load_added( tctx.rl );
        pass_free( pctx );
    }

    // trim

    pctx = pass_init( fileOvlIn, fileOvlOut );

    pctx->split_b         = 0;
    pctx->load_trace      = 1;
    pctx->unpack_trace    = 1;
    pctx->data            = &tctx;
    pctx->write_overlaps  = 1;
    pctx->purge_discarded = arg_purge;

    trim_pre( pctx, &tctx, tctx.rl );

    pass( pctx, trim_handler );

    trim_post( &tctx, arg_verbose );

    pass_free( pctx );

    // cleanup

    if ( arg_rloader )
    {
        rl_free( tctx.rl );
    }

    if ( db.tracks == NULL )
        track_close( tctx.trackTrim );

    Close_DB( &db );

    fclose( fileOvlOut );

    return 0;
}
