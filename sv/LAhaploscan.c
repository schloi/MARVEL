
#include "dalign/align.h"
#include "db/DB.h"
#include "lib/colors.h"
#include "lib/iseparator.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/utils.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>

// defaults

#define DEF_ARG_T  "trim"
#define DEF_ARG_Q  "q"
#define DEF_ARG_QQ 25
#define DEF_ARG_D  100
#define DEF_ARG_S  2

// flags for reads

#define FLAG_READ_PROCESS 0x01
#define FLAG_READ_POS_EVT 0x02
#define FLAG_READ_NEG_EVT 0x04

#define FLAG_READ_BOTH_EVTS ( FLAG_READ_POS_EVT | FLAG_READ_NEG_EVT )

// switches

#define VERBOSE

#undef DEBUG_SHOW_HAPLOTYPE_EVENTS
#undef DEBUG_SHOW_HAPLOTYPE_ALIGNMENTS

// helper macros

#define OVL_COMP_INT( o ) ( ( o )->flags & OVL_COMP ? 1 : 0 )

// structs

typedef struct _LaHaploScanContext
{
    HITS_DB* db;
    HITS_TRACK* tracktrim;
    HITS_TRACK* trackq;
    FILE* fileHapOut;
    ovl_header_twidth twidth;

    // command line arguments
    uint64_t lbreaks_maxdistance;
    uint64_t lbreaks_minsupport;
    uint16_t maxq;

    // working structs
    int64_t* events;
    size_t maxevents;

    uint64_t* lbreaks_intervals;
    size_t lbreaks_maxintervals;

} LaHaploscanContext;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

static int cmp_events( const void* a, const void* b )
{
    const int64_t* x = (int64_t*)a;
    const int64_t* y = (int64_t*)b;

    int64_t cmp = labs( x[ 0 ] ) - labs( y[ 0 ] );

    if ( cmp == 0 )
    {
        cmp = x[ 0 ] - y[ 0 ];
    }

    if ( cmp == 0 )
    {
        cmp = x[ 1 ] - y[ 1 ];
    }

    return cmp;
}

static int64_t dist_events( const void* a, const void* b )
{
    const int64_t* x = (int64_t*)a;
    const int64_t* y = (int64_t*)b;

    return labs( x[ 0 ] ) - labs( y[ 0 ] );
}

static void haploscan_pre( PassContext* pctx, LaHaploscanContext* hctx )
{
#ifdef VERBOSE
    printf( ANSI_COLOR_GREEN "PASS haplotype detection\n" ANSI_COLOR_RESET );
#endif

    hctx->twidth = pctx->twidth;

    hctx->maxevents = 1000;
    hctx->events    = malloc( sizeof( int64_t ) * hctx->maxevents );

    hctx->lbreaks_intervals    = NULL;
    hctx->lbreaks_maxintervals = 0;

    UNUSED( pctx );
}

static void haploscan_post( LaHaploscanContext* hctx )
{
    free( hctx->events );
    free( hctx->lbreaks_intervals );
}

static int haploscan_handler( void* _ctx, Overlap* ovls, int novl )
{
    LaHaploscanContext* hctx     = (LaHaploscanContext*)_ctx;
    HITS_DB* db                  = hctx->db;
    HITS_READ* reads             = db->reads;
    HITS_TRACK* tracktrim        = hctx->tracktrim;
    HITS_TRACK* trackq           = hctx->trackq;
    FILE* fileHapOut             = hctx->fileHapOut;
    int aread                    = ovls->aread;
    int64_t* events              = hctx->events;
    size_t maxevents             = hctx->maxevents;
    size_t nevents               = 0;
    uint64_t* lbreaks_intervals  = hctx->lbreaks_intervals;
    size_t lbreaks_maxintervals  = hctx->lbreaks_maxintervals;
    uint64_t lbreaks_maxdistance = hctx->lbreaks_maxdistance;
    uint64_t lbreaks_minsupport  = hctx->lbreaks_minsupport;
    uint16_t maxq                = hctx->maxq;
    int trimab, trimae;

    if ( !( reads[ aread ].flags & FLAG_READ_PROCESS ) )
    {
        return 1;
    }

    get_trim( db, tracktrim, aread, &trimab, &trimae );

    ovl_header_twidth twidth = hctx->twidth;
    track_anno* qanno        = trackq->anno;
    track_data* qdata        = trackq->data;
    track_data* qa           = qdata + ( qanno[ aread ] / sizeof( track_data ) );

    uint64_t alen      = DB_READ_LEN( db, aread );
    uint64_t nsegments = ( alen + twidth - 1 ) / twidth;

    int ob = qanno[ aread ] / sizeof( track_data );
    int oe = qanno[ aread + 1 ] / sizeof( track_data );

    if ( oe - ob != nsegments )
    {
        fprintf( stderr, "read %d expected %" PRIu64 " Q track entries, found %d\n", aread, nsegments, oe - ob );
        exit( 1 );
    }

    uint32_t i;
    for ( i = 0; i < (uint32_t)novl; i++ )
    {
        Overlap* ovl = ovls + i;

        // TODO: OVL_REPEAT exclusion ???

        if ( !( ovl->flags & OVL_LOCAL ) || ( ovl->flags & ( OVL_CONT | OVL_REPEAT | OVL_GAP ) ) )
        {
            continue;
        }

        // look for local alignments that can continue only on one end
        // collect events (pos, idx_ovl) for these ends. negative pos for left open
        // local alignments, positive for right open local alignments

        int bread = ovl->bread;
        int trimbb, trimbe;

        get_trim( db, tracktrim, bread, &trimbb, &trimbe );

        int abpos = ovl->path.abpos;
        int aepos = ovl->path.aepos;

        if ( qa[ abpos / twidth ] > maxq || qa[ aepos / twidth ] > maxq )
        {
            continue;
        }

        // TODO: ignore alignments ending in a low q region

        if ( ovl->flags & OVL_COMP )
        {
            int blen = DB_READ_LEN( db, bread );
            int bbpos    = blen - ovl->path.bepos;
            int bepos    = blen - ovl->path.bbpos;

            if ( ( abpos == trimab || bepos == trimbe ) && bbpos > trimbb )
            {
                events[ nevents ]     = aepos;
                events[ nevents + 1 ] = i;
                nevents += 2;
            }
            else if ( ( aepos == trimae || bbpos == trimbb ) && bepos < trimbe )
            {
                events[ nevents ]     = ( -1 ) * abpos;
                events[ nevents + 1 ] = i;
                nevents += 2;
            }
        }
        else
        {
            int bbpos = ovl->path.bbpos;
            int bepos = ovl->path.bepos;

            if ( ( abpos == trimab || bbpos == trimbb ) && bepos < trimbe )
            {
                events[ nevents ]     = aepos;
                events[ nevents + 1 ] = i;
                nevents += 2;
            }
            else if ( ( aepos == trimae || bepos == trimbe ) && bbpos > trimbb )
            {
                events[ nevents ]     = ( -1 ) * abpos;
                events[ nevents + 1 ] = i;
                nevents += 2;
            }
        }

        if ( nevents + 2 >= maxevents )
        {
            maxevents = maxevents * 1.2 + 100;
            events    = realloc( events, sizeof( int64_t ) * maxevents );
        }
    }

    if ( nevents == 0 )
    {
        return 1;
    }

    hctx->events    = events;
    hctx->maxevents = maxevents;

    qsort( events, nevents / 2, sizeof( int64_t ) * 2, cmp_events );

#ifdef DEBUG_SHOW_HAPLOTYPE_EVENTS
    for ( i = 0; i < nevents; i += 2 )
    {
        printf( "%6" PRIi64 " %6" PRIi64 "\n", events[ i ], events[ i + 1 ] );
    }
#endif // DEBUG_SHOW_HAPLOTYPE_EVENTS

    // separate events into clusters with a max in-cluster distance
    // iseparator expects the events to be sorted

    size_t nintervals;
    nintervals = iseparator( events, nevents / 2, sizeof( int64_t ) * 2, lbreaks_maxdistance, &lbreaks_intervals,
                             &lbreaks_maxintervals, dist_events );

    hctx->lbreaks_intervals    = lbreaks_intervals;
    hctx->lbreaks_maxintervals = lbreaks_maxintervals;

    uint32_t breakid = 0;

    for ( i = 0; i < nintervals; i += 2 )
    {
        uint64_t lbreaks_from = lbreaks_intervals[ i ] * 2;
        uint64_t lbreaks_to   = lbreaks_intervals[ i + 1 ] * 2;

        // we need at least twice the min support to get min support-many distinct reads
        // distinct reads are not guaranteed at this point
        if ( lbreaks_to - lbreaks_from < lbreaks_minsupport * 2 )
        {
            continue;
        }

#ifdef DEBUG_SHOW_HAPLOTYPE_EVENTS
        printf( "%3" PRIu64 " %3" PRIu64 " | ", lbreaks_intervals[ i ], lbreaks_intervals[ i + 1 ] );
        printf( "%6" PRIi64 " %6" PRIi64 "\n", events[ lbreaks_intervals[ i ] * 2 ],
                events[ lbreaks_intervals[ i + 1 ] * 2 ] );
#endif // DEBUG_SHOW_HAPLOTYPE_EVENTS

        uint64_t j;
        uint64_t distinctreads = 0;
        for ( j = lbreaks_from; j <= lbreaks_to; j += 2 )
        {
            int64_t evnt = events[ j ];
            Overlap* ovl = ovls + events[ j + 1 ];

            if ( evnt < 0 && !( reads[ ovl->bread ].flags & FLAG_READ_NEG_EVT ) )
            {
                reads[ ovl->bread ].flags |= FLAG_READ_NEG_EVT;

                if ( ( reads[ ovl->bread ].flags & FLAG_READ_BOTH_EVTS ) == FLAG_READ_BOTH_EVTS )
                {
                    distinctreads += 1;
                }
            }
            else if ( evnt > 0 && !( reads[ ovl->bread ].flags & FLAG_READ_POS_EVT ) )
            {
                reads[ ovl->bread ].flags |= FLAG_READ_POS_EVT;

                if ( ( reads[ ovl->bread ].flags & FLAG_READ_BOTH_EVTS ) == FLAG_READ_BOTH_EVTS )
                {
                    distinctreads += 1;
                }
            }
        }

#ifdef DEBUG_SHOW_HAPLOTYPE_ALIGNMENTS
        for ( j = lbreaks_from; j <= lbreaks_to; j += 2 )
        {
            Overlap* ovl = ovls + events[ j + 1 ];

            if ( ( reads[ ovl->bread ].flags & FLAG_READ_BOTH_EVTS ) == FLAG_READ_BOTH_EVTS )
            {
                printf( "%d x %d %d..%d %c %d..%d\n", ovl->aread, ovl->bread, ovl->path.abpos, ovl->path.aepos,
                        ovl->flags & OVL_COMP ? 'c' : 'n', ovl->path.bbpos, ovl->path.bepos );
            }
        }
#endif // DEBUG_SHOW_HAPLOTYPE_ALIGNMENTS

        // write as: (svid aread aposb apose) (b bposb bpose comp)+

        if ( distinctreads >= lbreaks_minsupport )
        {
            breakid += 1;

            uint64_t abpos = 0;
            uint64_t aepos = 0;

            for ( j = lbreaks_intervals[ i ] * 2; j <= lbreaks_intervals[ i + 1 ] * 2; j += 2 )
            {
                Overlap* ovl = ovls + events[ j + 1 ];

                if ( ( reads[ ovl->bread ].flags & FLAG_READ_BOTH_EVTS ) == FLAG_READ_BOTH_EVTS )
                {
                    if ( abpos == 0 )
                    {
                        abpos = labs( events[ j ] );
                    }

                    aepos = labs( events[ j ] );
                }
            }

            fprintf( fileHapOut, "%" PRIu32 " %d %" PRIu64 " %" PRIu64, breakid, aread, abpos, aepos );

            // alignments supporting the break

            for ( j = lbreaks_from; j <= lbreaks_to; j += 2 )
            {
                Overlap* ovl      = ovls + events[ j + 1 ];
                Overlap* ovl_pair = NULL;

                if ( ovl->flags & OVL_TEMP )
                {
                    continue;
                }

                uint64_t k;
                for ( k = j + 2; k <= lbreaks_to; k += 2 )
                {
                    Overlap* ovl2 = ovls + events[ k + 1 ];

                    if ( ( ovl2->flags & OVL_TEMP ) || ( ovl->flags & OVL_COMP ) != ( ovl2->flags & OVL_COMP ) )
                    {
                        continue;
                    }

                    if ( ovl->bread == ovl2->bread )
                    {
                        ovl_pair = ovl2;
                        break;
                    }
                }

                if ( ( reads[ ovl->bread ].flags & FLAG_READ_BOTH_EVTS ) == FLAG_READ_BOTH_EVTS )
                {
                    if ( !ovl_pair )
                    {
                        // printf("UNPAIRED %d %d %d\n", ovl->bread, ovl->path.bbpos, ovl->path.bepos);
                        continue;
                    }

                    ovl->flags |= OVL_TEMP;
                    ovl_pair->flags |= OVL_TEMP;

                    fprintf( fileHapOut, " %d %d %d %d %d %d", ovl->bread, ovl->path.bbpos, ovl->path.bepos,
                             ovl_pair->path.bbpos, ovl_pair->path.bepos, OVL_COMP_INT( ovl_pair ) );
                }
            }

            // alignments crossing the break

            for ( j = 0 ; j < novl ; j++ )
            {
                Overlap* o = ovls + j;

                // only proper alignments

                if ( o->flags & OVL_DISCARD )
                {
                    continue;
                }

                // crosses break
                // TODO: require additional padding ?

                if ( o->path.abpos < abpos && o->path.aepos > aepos )
                {
                    fprintf( fileHapOut, " %d", (-1) * o->bread );
                }
            }

            fprintf( fileHapOut, "\n" );
        }

        // clear flags

        for ( j = lbreaks_from; j <= lbreaks_to; j += 2 )
        {
            Overlap* ovl = ovls + events[ j + 1 ];
            reads[ ovl->bread ].flags &= ~FLAG_READ_BOTH_EVTS;
        }
    }

    return 1;
}

static void process_rids( const char* pathReadIds, HITS_DB* db )
{
    FILE* fileIn = fopen( pathReadIds, "r" );

    if ( fileIn == NULL )
    {
        fprintf( stderr, "could not open %s\n", pathReadIds );
        exit( 1 );
    }

    int* values;
    int nvalues;

    fread_integers( fileIn, &values, &nvalues );

    int i;
    for ( i = 0; i < nvalues; i++ )
    {
        db->reads[ values[ i ] ].flags = FLAG_READ_PROCESS;
    }

    free( values );

    fclose( fileIn );
}

static void usage()
{
    printf( "LAhaploscan [-dQs <int>] [-r <file>] [-qt <track>] <db> <input.las> <output.hap>\n" );
    printf( "options: -d maximum distance between haplotype-induced split alignments (%d)\n", DEF_ARG_D );
    printf( "         -q q track (%s)\n", DEF_ARG_Q);
    printf( "         -Q ignore haplotypes in regions with an error higher than (%d)\n", DEF_ARG_QQ );
    printf( "         -r only process reads with the id contained in the file\n" );
    printf( "         -s number of split alignments needed as support for a haplotype (%d)\n", DEF_ARG_S );
    printf( "         -t trim track (%s)\n", DEF_ARG_T );
};

int main( int argc, char* argv[] )
{
    HITS_DB db;
    LaHaploscanContext hctx;
    char* tracktrimname = DEF_ARG_T;
    char* trackqname    = DEF_ARG_Q;
    char* pathReadIds   = NULL;

    bzero( &hctx, sizeof( LaHaploscanContext ) );

    hctx.lbreaks_maxdistance = DEF_ARG_D;
    hctx.lbreaks_minsupport  = DEF_ARG_S;
    hctx.maxq                = DEF_ARG_QQ;

    // process arguments

    opterr = 0;

    int c;
    while ( ( c = getopt( argc, argv, "d:q:Q:r:s:t:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'd':
                hctx.lbreaks_maxdistance = strtol( optarg, NULL, 10 );
                break;

            case 'q':
                trackqname = optarg;
                break;

            case 'Q':
                hctx.maxq = strtol( optarg, NULL, 10 );
                break;

            case 'r':
                pathReadIds = optarg;
                break;

            case 's':
                hctx.lbreaks_minsupport = strtol( optarg, NULL, 10 );
                break;

            case 't':
                tracktrimname = optarg;
                break;

            default:
                usage();
                exit( 1 );
        }
    }

    if ( argc - optind < 3 )
    {
        usage();
        exit( 1 );
    }

    char* pcPathDbIn       = argv[ optind++ ];
    char* pcPathOverlapsIn = argv[ optind++ ];
    char* pcPathHapOut     = argv[ optind++ ];
    FILE* fileOvlIn        = NULL;

    if ( Open_DB( pcPathDbIn, &db ) )
    {
        fprintf( stderr, "could not open '%s'\n", pcPathDbIn );
        exit( 1 );
    }
    hctx.db = &db;

    if ( ( hctx.tracktrim = track_load( &db, tracktrimname ) ) == NULL )
    {
        fprintf( stderr, "could not open track '%s'\n", tracktrimname );
        exit( 1 );
    }

    if ( ( hctx.trackq = track_load( &db, trackqname ) ) == NULL )
    {
        fprintf( stderr, "could not open track '%s'\n", trackqname );
        exit( 1 );
    }

    if ( ( fileOvlIn = fopen( pcPathOverlapsIn, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open %s\n", pcPathOverlapsIn );
        exit( 1 );
    }

    if ( strcmp( pcPathHapOut, "-" ) == 0 )
    {
        hctx.fileHapOut = stdout;
    }
    else if ( ( hctx.fileHapOut = fopen( pcPathHapOut, "w" ) ) == NULL )
    {
        fprintf( stderr, "could not open %s\n", pcPathHapOut );
        exit( 1 );
    }

    if ( pathReadIds )
    {
        process_rids( pathReadIds, &db );
    }
    else
    {
        int i;
        for ( i = 0; i < DB_NREADS( &db ); i++ )
        {
            db.reads[ i ].flags = FLAG_READ_PROCESS;
        }
    }

    // init
    PassContext* pctx;

    pctx = pass_init( fileOvlIn, NULL );

    pctx->split_b      = 0;
    pctx->load_trace   = 1;
    pctx->unpack_trace = 1;
    pctx->data         = &hctx;

    haploscan_pre( pctx, &hctx );
    pass( pctx, haploscan_handler );
    haploscan_post( &hctx );

    // cleanup

    pass_free( pctx );
    Close_DB( hctx.db );

    return 0;
}
