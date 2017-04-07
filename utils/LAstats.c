/*******************************************************************************************
 *
 *  display basic stats on the overlaps contained in a .las file
 *
 *  Author :  MARVEL Team
 *
 *  Date   :  May 2015
 *
 *******************************************************************************************/

#include <assert.h>
#include <limits.h>
#include <math.h>
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

// command line defaults

#define DEF_ARG_S -1
#define DEF_ARG_F 0
#define DEF_ARG_B 1000
#define DEF_ARG_T TRACK_TRIM

// read flags

#define R_CONTAINED ( 1 << 0 )
#define R_USED_A ( 1 << 1 )
#define R_USED_B ( 1 << 2 )

#define R_CONTAINED_G ( 1 << 3 )
#define R_USED_A_G ( 1 << 4 )
#define R_USED_B_G ( 1 << 5 )

#define R_MASK_G ( R_CONTAINED_G | R_USED_A_G | R_USED_B_G )

typedef struct
{
    // stat counters for a single overlap file
    uint32_t nContainedReads;
    uint64_t nContainedBases;
    uint64_t nOverlaps;
    uint64_t nComplementOverlaps;
    uint64_t nIdentityOverlaps;
    uint64_t nStitched;

    // overall stat counters for all overlap files
    uint32_t nAllContainedReads;
    uint64_t nAllContainedBases;
    uint64_t nAllBases;
    uint64_t nAllOverlaps;
    uint64_t nAllComplementOverlaps;
    uint64_t nAllIdentityOverlaps;

    // command line switches
    int verbose;
    int fuzzing;
    int stitch;
    int dumpOutContainedReads;
	int raw;

    // overlap length binning
    int ovlBinSize;
    int nbin;
    int* hist;
    uint64_t* bsum;

    // db
    HITS_DB* db;
    HITS_TRACK* tracktrim;
} StatsContext;

// for getopt

extern char* optarg;
extern int optind, opterr, optopt;

static void stitch( StatsContext* ctx, Overlap* pOvls, int n, int sfuzz )
{
    if ( n < 2 )
    {
        return;
    }

    int i, k, b;
    int ab2, ae1, ae2;
    int bb2, be1, be2;

    const int ignore_mask = OVL_CONT | OVL_STITCH | OVL_GAP | OVL_TRIM;

    for ( i = 0; i < n; i++ )
    {
        if ( pOvls[ i ].flags & ignore_mask )
        {
            continue;
        }

        b = pOvls[ i ].bread;

        ae1 = pOvls[ i ].path.aepos;
        be1 = pOvls[ i ].path.bepos;

        for ( k = i + 1; k < n && pOvls[ k ].bread <= b; k++ )
        {
            if ( ( pOvls[ k ].flags & ignore_mask ) || ( pOvls[ i ].flags & OVL_COMP ) != ( pOvls[ k ].flags & OVL_COMP ) )
            {
                continue;
            }

            ab2 = pOvls[ k ].path.abpos;
            ae2 = pOvls[ k ].path.aepos;

            bb2 = pOvls[ k ].path.bbpos;
            be2 = pOvls[ k ].path.bepos;

            int deltaa = abs( ae1 - ab2 );
            int deltab = abs( be1 - bb2 );

            if ( deltaa < sfuzz && deltab < sfuzz && ( abs( deltaa - deltab ) < 40 ) )
            {
                pOvls[ i ].path.aepos = ae2;
                pOvls[ i ].path.bepos = be2;
                pOvls[ i ].path.diffs += pOvls[ k ].path.diffs;
                pOvls[ i ].path.tlen = 0;

                pOvls[ i ].flags &= ~( OVL_DISCARD | OVL_LOCAL ); // force a re-evaluation of the OVL_LOCAL flags
                pOvls[ i ].flags |= OVL_OPTIONAL;

                pOvls[ k ].flags |= OVL_DISCARD | OVL_STITCH | OVL_TEMP;

                ctx->nStitched++;
            }
        }
    }
}

static int contained_stats( StatsContext* ctx, Overlap* pOvls, int n )
{
    int i, a, b;
    int ab, ae, bb, be;
    int rLenA, rLenB;
    int tab, tae, tbb, tbe;

    rLenA = DB_READ_LEN( ctx->db, pOvls->aread );
    rLenB = DB_READ_LEN( ctx->db, pOvls->bread );

    a = pOvls->aread;
    b = pOvls->bread;

    if ( ctx->tracktrim )
    {
        get_trim( ctx->db, ctx->tracktrim, a, &tab, &tae );
        get_trim( ctx->db, ctx->tracktrim, b, &tbb, &tbe );
    }
    else
    {
        tab = 0;
        tae = rLenA;
        tbb = 0;
        tbe = rLenB;
    }

    for ( i = 0; i < n; i++ )
    {
        Overlap* ovl = pOvls + i;

        if ( ovl->flags & OVL_DISCARD )
        {
            continue;
        }

        ctx->nOverlaps++;

        ab = ovl->path.abpos;
        ae = ovl->path.aepos;

        if ( ovl->flags & OVL_COMP )
        {
            ctx->nComplementOverlaps++;
            bb = rLenB - ovl->path.bepos;
            be = rLenB - ovl->path.bbpos;
        }
        else
        {
            bb = ovl->path.bbpos;
            be = ovl->path.bepos;
        }

        // count identity overlaps
        if ( ovl->aread == ovl->bread )
        {
            ctx->nIdentityOverlaps++;
        }

        // check for contained reads
        if ( bb <= tbb + ctx->fuzzing && be >= tbe - ctx->fuzzing )
        {
            ctx->db->reads[ b ].flags |= R_CONTAINED | R_CONTAINED_G;
        }

        if ( ab <= tab + ctx->fuzzing && ae >= tae - ctx->fuzzing )
        {
            ctx->db->reads[ a ].flags |= R_CONTAINED | R_CONTAINED_G;
        }
    }

    return 1;
}

static void stats_pre( StatsContext* fctx )
{
    int nreads = fctx->db->nreads;

    fctx->nContainedReads     = 0;
    fctx->nContainedBases     = 0;
    fctx->nOverlaps           = 0;
    fctx->nComplementOverlaps = 0;
    fctx->nIdentityOverlaps   = 0;

    // allocate bit masks
    if ( fctx->hist == NULL )
    {
        fctx->nbin = DB_READ_MAXLEN( fctx->db ) / fctx->ovlBinSize + 1;
        fctx->hist = (int*)malloc( sizeof( int ) * fctx->nbin );
        fctx->bsum = (uint64_t*)malloc( sizeof( uint64_t ) * fctx->nbin );
    }

    int i;
    HITS_READ* reads = fctx->db->reads;
    for ( i = 0; i < nreads; i++ )
    {
        reads[ i ].flags &= R_MASK_G;
    }

    bzero( fctx->hist, sizeof( int ) * fctx->nbin );
    bzero( fctx->bsum, sizeof( uint64_t ) * fctx->nbin );
}

static void stats_post( StatsContext* ctx, int last )
{
    // update stats
    int i;
    int dbReads         = DB_NREADS( ctx->db );
    int used            = 0;
    int useda           = 0;
    int usedb           = 0;
    int contained       = 0;
    uint64_t containedb = 0;
	int raw = ctx->raw;

    for ( i = 0; i < dbReads; i++ )
    {
        HITS_READ* read = ctx->db->reads + i;
        int flags       = read->flags;

        if ( flags & R_CONTAINED )
        {
            contained += 1;
            containedb += DB_READ_LEN( ctx->db, i );
        }

        if ( flags & ( R_USED_A | R_USED_B ) )
        {
            used += 1;
        }

        if ( flags & R_USED_A )
        {
            useda += 1;
        }

        if ( flags & R_USED_B )
        {
            usedb += 1;
        }
    }

    uint64_t nbases = 0;
    for ( i = ctx->nbin - 1; i >= 0; i-- )
    {
        if ( ctx->hist[ i ] )
        {
            nbases += ctx->bsum[ i ];
        }
    }

    ctx->nAllBases += nbases;

    // update overall stats
    ctx->nAllOverlaps += ctx->nOverlaps;
    ctx->nAllComplementOverlaps += ctx->nComplementOverlaps;
    ctx->nAllIdentityOverlaps += ctx->nIdentityOverlaps;

    // output

	if (!raw)
	{
		printf( "a-reads: %7d\n", useda );
		printf( "b-reads: %7d\n", usedb );
		printf( "reads:   %7d\n", used );

		if ( contained > 0 )
		{
			printf( "#contained reads: %d (%.2f%%), contained bases: %" PRIu64 " (%.2f%%), avgLen: %" PRIu64 "\n",
					contained,
					100.0 * contained / used,
					containedb,
					100.0 * containedb / nbases,
					containedb / contained );
		}

		if ( ctx->nIdentityOverlaps > 0 )
		{
			printf( "#identity overlaps %" PRIu64 " from %d reads\n",
					ctx->nIdentityOverlaps,
					useda );
		}
	}

    if ( ctx->nOverlaps > 0 )
    {
        uint64_t nfwd = ctx->nOverlaps - ctx->nComplementOverlaps;

		if (raw)
		{
			printf( "%" PRIu64 " %" PRIu64 "\n", ctx->nOverlaps, nbases / ctx->nOverlaps );
		}
		else
		{
			printf( "#overlaps %" PRIu64 ", n: %" PRIu64 " (%.2f%%), c: %" PRIu64 " (%.2f%%), avgLen %" PRIu64 "\n",
					ctx->nOverlaps,
					nfwd,
					nfwd * 100.0 / ctx->nOverlaps,
					ctx->nComplementOverlaps,
					ctx->nComplementOverlaps * 100.0 / ctx->nOverlaps,
					nbases / ctx->nOverlaps );
		}
    }

    if ( !raw && ctx->nStitched > 0 )
    {
        printf( "#stitched overlaps %" PRIu64 "\n", ctx->nStitched );
    }

	if ( !raw )
	{
	    printf( "\n  Distribution of Overlap Lengths (Bin size = %d)\n\n        Bin:      Count  %% Overlaps  %% Bases    cumAverage    binAverage\n", ctx->ovlBinSize );
	}

    uint64_t cum  = 0;
    uint64_t btot = 0;
    for ( i = ctx->nbin - 1; i >= 0; i-- )
    {
        cum += ctx->hist[ i ];
        btot += ctx->bsum[ i ];

        if ( ( ctx->hist[ i ] > 0 ) )
        {
			if (raw)
			{
				printf( "%d %d %.1f %.1f %" PRIu64 " %" PRIu64 "\n",
						i * ctx->ovlBinSize,
						ctx->hist[ i ],
						( 100. * cum ) / ctx->nOverlaps,
						( 100. * btot ) / nbases,
						btot / cum,
						ctx->bsum[ i ] / ctx->hist[ i ] );
			}
			else
			{
				printf( "%11d:   %8d       %5.1f    %5.1f     %9" PRIu64 "     %9" PRIu64 "\n",
						i * ctx->ovlBinSize,
						ctx->hist[ i ],
						( 100. * cum ) / ctx->nOverlaps,
						( 100. * btot ) / nbases,
						btot / cum,
						ctx->bsum[ i ] / ctx->hist[ i ] );
			}
        }
    }

    if ( !raw && last )
    {
        int used_g            = 0;
        int useda_g           = 0;
        int usedb_g           = 0;
        int contained_g       = 0;
        uint64_t containedb_g = 0;
        uint64_t nfwd_g       = ctx->nAllOverlaps - ctx->nAllComplementOverlaps;

        for ( i = 0; i < dbReads; i++ )
        {
            HITS_READ* read = ctx->db->reads + i;
            int flags       = read->flags;

            if ( flags & R_CONTAINED_G )
            {
                contained_g += 1;
                containedb_g += DB_READ_LEN( ctx->db, i );
            }

            if ( flags & ( R_USED_A_G | R_USED_B_G ) )
            {
                used_g += 1;
            }

            if ( flags & R_USED_A_G )
            {
                useda_g += 1;
            }

            if ( flags & R_USED_B_G )
            {
                usedb_g += 1;
            }
        }

        if ( used_g != used )
        {
            printf( "#overall Areads: %d\n", useda_g );
            printf( "#overall Breads: %d\n", usedb_g );
            printf( "#overall reads: %d\n", used_g );

            if ( contained_g > 0 )
            {
                printf( "#overall contained reads: %d  (%.2f%%), contained bases: %" PRIu64 " (%.2f%%), avgLen: %" PRIu64 "\n",
                        contained_g, 100.0 * contained_g / used_g, containedb_g, 100.0 * containedb_g / ctx->nAllBases, containedb_g / contained_g );
            }

            if ( ctx->nAllIdentityOverlaps > 0 )
            {
                printf( "#overall identity overlaps %" PRIu64 " (%.2f%%) from %d reads\n",
                        ctx->nAllIdentityOverlaps, 100.0 * ctx->nAllIdentityOverlaps / ctx->nAllOverlaps, useda_g );
            }

            if ( ctx->nAllOverlaps > 0 )
            {
                printf( "#overall overlaps %" PRIu64 ", n: %" PRIu64 ", (%.2f%%), c: %" PRIu64 " (%.2f%%) avgLen %" PRIu64 "\n",
                        ctx->nAllOverlaps, nfwd_g, nfwd_g * 100.0 / ctx->nAllOverlaps,
                        ctx->nAllComplementOverlaps, ctx->nAllComplementOverlaps * 100.0 / ctx->nAllOverlaps, ctx->nAllBases / ctx->nAllOverlaps );
            }
        }

        if ( ctx->dumpOutContainedReads && last )
        {
            for ( i = 0; i < dbReads; i++ )
            {
                if ( ctx->db->reads[ i ].flags & R_CONTAINED_G )
                    printf( "%d\n", i );
            }
        }
    }
}

static int stats_handler( void* _ctx, Overlap* ovls, int novl )
{
    StatsContext* ctx = (StatsContext*)_ctx;
    int j;

    ctx->db->reads[ ovls->aread ].flags |= R_USED_A | R_USED_A_G;

    // stitch
    if ( ctx->stitch >= 0 )
    {
        int k;
        j = k = 0;
        while ( j < novl )
        {
            while ( k < novl - 1 && ovls[ j ].bread == ovls[ k + 1 ].bread )
                k++;

            stitch( ctx, ovls + j, k - j + 1, ctx->stitch );

            j = k + 1;
        }
    }

    // contained stats
    {
        int k;
        j = k = 0;
        while ( j < novl )
        {
            while ( k < novl - 1 && ovls[ j ].bread == ovls[ k + 1 ].bread )
                k++;

            contained_stats( ctx, ovls + j, k - j + 1 );

            ctx->db->reads[ ovls[ j ].bread ].flags |= R_USED_B | R_USED_B_G;

            j = k + 1;
        }
    }

    // overlap length histogram
    for ( j = 0; j < novl; j++ )
    {
        Overlap* ovl = ovls + j;

        if ( ovl->flags & OVL_DISCARD )
            continue;

        int l = ovl->path.aepos - ovl->path.abpos;
        int b = l / ctx->ovlBinSize;

        if ( b >= ctx->nbin )
            b = ctx->nbin - 1;

        ctx->hist[ b ] += 1;
        ctx->bsum[ b ] += l;
    }

    return 1;
}

static void usage()
{
    fprintf( stderr, "[-vdr] [-bfs <int>] <db> <overlaps_in> ... | <overlaps.#.las> \n" );
    fprintf( stderr, "options: -v ... verbose\n" );
    fprintf( stderr, "         -f ... containment fuzzing, i.e. #fuzzing bases, that are ignored from begin or end of overlap (default: %d)\n", DEF_ARG_F );
    fprintf( stderr, "         -s ... stitch (%d)\n", DEF_ARG_S );
    fprintf( stderr, "         -d ... dump out contained reads\n" );
    fprintf( stderr, "         -b ... bucket size of histogram length (%d)\n", DEF_ARG_B );
    fprintf( stderr, "         -t ... trim track (%s)\n", DEF_ARG_T );
	fprintf( stderr, "         -r ... raw output\n");
}

int main( int argc, char* argv[] )
{
    HITS_DB db;
    StatsContext sctx;
    PassContext* pctx;
    FILE* fileOvlIn;

    bzero( &sctx, sizeof( StatsContext ) );

    sctx.db                    = &db;
    sctx.fuzzing               = DEF_ARG_F;
    sctx.stitch                = DEF_ARG_S;
    sctx.dumpOutContainedReads = 0;
    sctx.ovlBinSize            = DEF_ARG_B;
	sctx.raw 				   = 0;

    // args

    int c;
    char* trimname = DEF_ARG_T;
    opterr         = 0;

    while ( ( c = getopt( argc, argv, "rvdf:s:t:" ) ) != -1 )
    {
        switch ( c )
        {
			case 'r':
				sctx.raw = 1;
				break;

            case 't':
                trimname = optarg;
                break;

            case 'v':
                sctx.verbose++;
                break;

            case 'd':
                sctx.dumpOutContainedReads = 1;
                break;

            case 'f':
                sctx.fuzzing = atoi( optarg );
                if ( sctx.fuzzing < 0 )
                {
                    fprintf( stderr,
                             "[ERROR] - LAstats: -f argument must be positive!\n" );
                    usage();
                    exit( 1 );
                }
                break;

            case 's':
                sctx.stitch = atoi( optarg );
                break;

            default:
                usage();
                exit( 1 );
        }
    }

    if ( argc - optind < 2 )
    {
        usage();
        exit( 1 );
    }

    char* pcPathReadsIn = argv[ optind++ ];

    if ( sctx.verbose )
        printf( "Open database: %s\n", pcPathReadsIn );

    if ( Open_DB( pcPathReadsIn, &db ) )
    {
        fprintf( stderr, "could not open %s\n", pcPathReadsIn );
        exit( 1 );
    }

    if ( !( sctx.tracktrim = track_load( &db, trimname ) ) )
    {
        fprintf( stderr, "could not open %s\n", trimname );
    }

    char* pcPathOverlapsIn = argv[ optind ];
    char* hashPos          = strchr( pcPathOverlapsIn, '#' );
    int blocks             = 1;

    char* pathLas = malloc( strlen( pcPathOverlapsIn ) + 100 );
    char *prefix, *suffix;

    if ( hashPos != NULL )
    {
        blocks = DB_Blocks( pcPathReadsIn );

        prefix   = pcPathOverlapsIn;
        suffix   = hashPos + 1;
        *hashPos = '\0';
    }
    else
    {
        blocks = argc - optind;
    }

    int b;
    for ( b = 1; b <= blocks; b++ )
    {
        if ( hashPos != NULL )
        {
            sprintf( pathLas, "%s%d%s", prefix, b, suffix );

            if ( ( fileOvlIn = fopen( pathLas, "r" ) ) == NULL )
            {
                fprintf( stderr, "could not open %s\n", pathLas );
                exit( 1 );
            }

            if ( sctx.verbose )
                printf( ANSI_COLOR_GREEN "PASS stats %s\n" ANSI_COLOR_RESET,
                        pathLas );
        }
        else
        {
            if ( ( fileOvlIn = fopen( argv[ optind ], "r" ) ) == NULL )
            {
                fprintf( stderr, "could not open %s\n", argv[ optind ] );
                exit( 1 );
            }
            if ( sctx.verbose )
                printf( ANSI_COLOR_GREEN "PASS stats %s\n" ANSI_COLOR_RESET,
                        argv[ optind ] );
            optind++;
        }

        // passes

        pctx = pass_init( fileOvlIn, NULL );

        pctx->split_b         = 0;
        pctx->load_trace      = 0;
        pctx->data            = &sctx;
        pctx->write_overlaps  = 0;
        pctx->purge_discarded = 0;

        stats_pre( &sctx );

        pass( pctx, stats_handler );

        int last = 0;
        if ( b == blocks )
        {
            last = 1;
        }

        stats_post( &sctx, last );

        // cleanup

        pass_free( pctx );
        fclose( fileOvlIn );
    }

    if ( hashPos != NULL )
    {
        *hashPos = '#';
    }

    Close_DB( &db );
    free( pathLas );

    return 0;
}
