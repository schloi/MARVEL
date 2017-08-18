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

#define DEF_ARG_F 0
#define DEF_ARG_B 1000
#define DEF_ARG_T NULL

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

    // overall stat counters for all overlap files
    uint32_t nAllContainedReads;
    uint64_t nAllContainedBases;
    uint64_t nAllBases;
    uint64_t nAllOverlaps;
    uint64_t nAllComplementOverlaps;
    uint64_t nAllIdentityOverlaps;

    // command line switches
    int fuzzing;
    int dumpOutContainedReads;
    int raw;

    // overlap length binning
    int ovlBinSize;
    int nbin;
    uint64_t* hist;
    uint64_t* bsum;
    uint64_t* hist_local;

    // db
    HITS_DB* db;
    HITS_TRACK* tracktrim;
} StatsContext;

// for getopt

extern char* optarg;
extern int optind, opterr, optopt;

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
        fctx->nbin       = DB_READ_MAXLEN( fctx->db ) / fctx->ovlBinSize + 1;
        fctx->hist       = malloc( sizeof( uint64_t ) * fctx->nbin );
        fctx->hist_local = malloc( sizeof( uint64_t ) * fctx->nbin );
        fctx->bsum       = malloc( sizeof( uint64_t ) * fctx->nbin );
    }

    int i;
    HITS_READ* reads = fctx->db->reads;
    for ( i = 0; i < nreads; i++ )
    {
        reads[ i ].flags &= R_MASK_G;
    }

    bzero( fctx->hist, sizeof( uint64_t ) * fctx->nbin );
    bzero( fctx->hist_local, sizeof( uint64_t ) * fctx->nbin );
    bzero( fctx->bsum, sizeof( uint64_t ) * fctx->nbin );
}

static void stats_post( StatsContext* ctx )
{
    // update stats
    int i;
    int dbReads         = DB_NREADS( ctx->db );
    int used            = 0;
    int useda           = 0;
    int usedb           = 0;
    int contained       = 0;
    uint64_t containedb = 0;
    int raw             = ctx->raw;

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

    if ( !raw )
    {
        printf( "distinct A-reads: %'7d\n", useda );
        printf( "distinct B-reads: %'7d\n", usedb );
        printf( "distinct reads:   %'7d\n", used );

        if ( contained > 0 )
        {
            printf( "contained reads:  %'7d (%.2f%%)\n", contained, 100.0 * contained / used );
        }
    }

    if ( ctx->nOverlaps > 0 )
    {
        if ( raw )
        {
            printf( "# alignments avg.length\n" );
            printf( "%" PRIu64 " %" PRIu64 "\n", ctx->nOverlaps, nbases / ctx->nOverlaps );
        }
        else
        {
            printf( "alignments:       %'7" PRIu64 " average length %" PRIu64 "\n",
                    ctx->nOverlaps, nbases / ctx->nOverlaps );
        }
    }

    if ( !raw )
    {
        printf( "\nDistribution of Alignment Lengths (Bin size = %dbp)\n\n    Bin      Count    Local      %%  %% Bases  cum.avgerage\n", ctx->ovlBinSize );
    }
    else
    {
        printf( "# bin count.alignemnts local.alignments %%.alignments %%.bases cum.average\n" );
    }

    uint64_t cum  = 0;
    uint64_t btot = 0;
    for ( i = ctx->nbin - 1; i >= 0; i-- )
    {
        cum += ctx->hist[ i ];
        btot += ctx->bsum[ i ];

        if ( ( ctx->hist[ i ] > 0 ) )
        {
            if ( raw )
            {
                printf( "%d %" PRIu64 " %" PRIu64 " %.1f %.1f %" PRIu64 "\n",
                        i * ctx->ovlBinSize,
                        ctx->hist[ i ],
                        ctx->hist_local[ i ],
                        ( 100. * cum ) / ctx->nOverlaps,
                        ( 100. * btot ) / nbases,
                        btot / cum );
            }
            else
            {
                printf( "%7d   %8" PRIu64 "%8" PRIu64 "  %5.1f    %5.1f     %9" PRIu64 "\n",
                        i * ctx->ovlBinSize,
                        ctx->hist[ i ],
                        ctx->hist_local[ i ],
                        ( 100. * cum ) / ctx->nOverlaps,
                        ( 100. * btot ) / nbases,
                        btot / cum );
            }
        }
    }
}

static int stats_handler( void* _ctx, Overlap* ovls, int novl )
{
    StatsContext* ctx = (StatsContext*)_ctx;
    int j;

    ctx->db->reads[ ovls->aread ].flags |= R_USED_A | R_USED_A_G;

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

    int trim_ab, trim_ae, trim_bb, trim_be;

    if ( ctx->tracktrim )
    {
        get_trim( ctx->db, ctx->tracktrim, ovls->aread, &trim_ab, &trim_ae );
    }
    else
    {
        trim_ab = 0;
        trim_ae = DB_READ_LEN( ctx->db, ovls->aread );
    }

    // overlap length histogram
    for ( j = 0; j < novl; j++ )
    {
        Overlap* ovl = ovls + j;

        if ( ovl->flags & OVL_DISCARD )
        {
            continue;
        }

        int l = ovl->path.aepos - ovl->path.abpos;
        int b = l / ctx->ovlBinSize;

        if ( b >= ctx->nbin )
        {
            b = ctx->nbin - 1;
        }

        if ( ctx->tracktrim )
        {
            get_trim( ctx->db, ctx->tracktrim, ovl->bread, &trim_bb, &trim_be );

            if ( ovl->flags & OVL_COMP )
            {
                int blen = DB_READ_LEN( ctx->db, ovl->bread );
                int t    = trim_bb;

                trim_bb  = blen - trim_be;
                trim_be  = blen - t;
            }

            if ( ( ( ovl->path.abpos - trim_ab ) > 0 && ( ovl->path.bbpos - trim_bb ) > 0 ) ||
                 ( ( trim_ae - ovl->path.aepos ) > 0 && ( trim_be - ovl->path.bepos ) > 0 ) )
            {
                ctx->hist_local[ b ] += 1;
            }
        }

        ctx->hist[ b ] += 1;
        ctx->bsum[ b ] += l;
    }

    return 1;
}

static void usage( FILE* fout, const char* app )
{
    fprintf( fout, "usage: %s [-r] [-t track] [-b n] database input.las [input2.las ...]\n\n", app );
    fprintf( fout, "Output basic statistics on the alignments found in the las file(s).\n\n" );
    fprintf( fout, "options: -b  bin size for the alignment lengths' histogram (%d)\n", DEF_ARG_B );
    fprintf( fout, "         -t  trim track for local alignment classification\n" );
    fprintf( fout, "         -r  raw (parsing friendly) output\n" );
}

int main( int argc, char* argv[] )
{
    HITS_DB db;
    StatsContext sctx;
    PassContext* pctx;
    FILE* fileOvlIn;
    char* app = argv[ 0 ];

    bzero( &sctx, sizeof( StatsContext ) );

    sctx.db         = &db;
    sctx.fuzzing    = DEF_ARG_F;
    sctx.ovlBinSize = DEF_ARG_B;
    sctx.raw        = 0;

    // args

    int c;
    char* trimname = DEF_ARG_T;
    opterr         = 0;

    while ( ( c = getopt( argc, argv, "rt:b:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'r':
                sctx.raw = 1;
                break;

            case 't':
                trimname = optarg;
                break;

            case 'b':
                sctx.ovlBinSize = atoi( optarg );
                if ( sctx.ovlBinSize <= 0 )
                {
                    fprintf( stderr, "Invalid histogram bucket size of %d\n", sctx.ovlBinSize );
                    exit( 1 );
                }
                break;

            default:
                usage( stdout, app );
                exit( 1 );
        }
    }

    if ( argc - optind < 2 )
    {
        usage( stdout, app );
        exit( 1 );
    }

    char* pcPathReadsIn = argv[ optind++ ];

    if ( Open_DB( pcPathReadsIn, &db ) )
    {
        fprintf( stderr, "could not open %s\n", pcPathReadsIn );
        exit( 1 );
    }

    if ( trimname != NULL )
    {
        if ( !( sctx.tracktrim = track_load( &db, trimname ) ) )
        {
            fprintf( stderr, "could not open %s\n", trimname );
        }
    }

    char* pcPathOverlapsIn = argv[ optind ];
    int blocks             = 1;

    char* pathLas = malloc( strlen( pcPathOverlapsIn ) + 100 );

    blocks = argc - optind;

    stats_pre( &sctx );

    int b;
    for ( b = 1; b <= blocks; b++ )
    {
        if ( ( fileOvlIn = fopen( argv[ optind ], "r" ) ) == NULL )
        {
            fprintf( stderr, "could not open %s\n", argv[ optind ] );
            exit( 1 );
        }

        optind++;

        // passes

        pctx = pass_init( fileOvlIn, NULL );

        pctx->split_b         = 0;
        pctx->load_trace      = 0;
        pctx->data            = &sctx;
        pctx->write_overlaps  = 0;
        pctx->purge_discarded = 0;

        pass( pctx, stats_handler );

        int last = 0;
        if ( b == blocks )
        {
            last = 1;
        }

        // cleanup

        pass_free( pctx );
        fclose( fileOvlIn );
    }

    stats_post( &sctx );

    Close_DB( &db );
    free( pathLas );

    return 0;
}
