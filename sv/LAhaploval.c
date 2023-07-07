
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

#define DEF_ARG_L 200
#define DEF_ARG_B 100

// flags for reads

#define FLAG_READ_PROCESS 0x01

// switches

#define VERBOSE
#undef DEBUG_SHOW_SPAN

// structs

typedef struct _HapCand
{
    uint32_t svid;
    uint32_t aread;
    uint32_t bread;
    uint32_t comp;
    uint32_t votes;

    uint64_t bb1;
    uint64_t be1;
    uint64_t bb2;
    uint64_t be2;
} HapCand;

typedef struct _LaHaploValidateContext
{
    HITS_DB* db;
    HITS_TRACK* tracktrim;
    FILE* fileHapValOut;

    size_t maxhapcand;
    size_t nhapcand;
    HapCand* hapcand;

    uint64_t* hapcandidx;

    uint32_t span_minlr;
    uint32_t span_binsize;

    uint64_t* span_bins;
    size_t span_maxbins;

} LaHaploValidateContext;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

static int cmp_hapcand( const void* a, const void* b )
{
    const HapCand* x = (HapCand*)a;
    const HapCand* y = (HapCand*)b;

    int64_t cmp = x->bread - y->bread;

    if ( cmp == 0 )
    {
        cmp = x->bb1 - y->bb1;
    }

    if ( cmp == 0 )
    {
        cmp = x->be1 - y->be1;
    }

    return cmp;
}

static void process_hap_file( LaHaploValidateContext* hctx, FILE* fileIn )
{
    HITS_DB* db       = hctx->db;
    HapCand* hapcand  = hctx->hapcand;
    size_t maxhapcand = hctx->maxhapcand;
    size_t nhapcand   = hctx->nhapcand;
    int64_t* values;
    uint64_t* sets;
    size_t nsets;

    uint64_t* hapcandidx = hctx->hapcandidx = calloc( DB_NREADS( db ) + 1, sizeof( uint64_t ) );

    nsets = fread_integer_sets( fileIn, &values, &sets );

    // sets: (svid aread aposb apose) (b bposb bpose comp)+

    size_t i;
    for ( i = 0; i < nsets; i++ )
    {
        uint64_t beg = sets[ i ];
        uint64_t end = sets[ i + 1 ];

        int64_t svid  = values[ beg ];
        int64_t aread = values[ beg + 1 ];
        // int64_t ab    = values[ beg + 2 ];
        // int64_t ae    = values[ beg + 3 ];
        beg += 4;

        while ( beg < end )
        {
            // if the bread is positive, then we have support for a break
            // otherwise it is the id of a read crossing the break (unbroken)

            int64_t bread = values[ beg ];

            if ( bread < 0 )
            {
                beg += 1;
                continue;
            }

            int64_t bb1   = values[ beg + 1 ];
            int64_t be1   = values[ beg + 2 ];
            int64_t bb2   = values[ beg + 3 ];
            int64_t be2   = values[ beg + 4 ];
            int64_t comp  = values[ beg + 5 ];

            if ( comp )
            {
                int64_t blen = DB_READ_LEN( db, bread );

                int tmp = bb1;
                bb1     = blen - be1;
                be1     = blen - tmp;

                tmp = bb2;
                bb2 = blen - be2;
                be2 = blen - tmp;
            }

            if ( be2 < be1 )
            {
                int64_t tmp;
                tmp = bb1;
                bb1 = bb2;
                bb2 = tmp;

                tmp = be1;
                be1 = be2;
                be2 = tmp;
            }

            if ( nhapcand == maxhapcand )
            {
                size_t tmp;
                tmp = maxhapcand * 1.2 + 1000;
                hapcand    = realloc( hapcand, sizeof( HapCand ) * tmp );
                bzero(hapcand + maxhapcand, sizeof(HapCand) * (tmp - maxhapcand));
                maxhapcand =  tmp;
            }

            hapcand[ nhapcand ].svid  = svid;
            hapcand[ nhapcand ].aread = aread;
            hapcand[ nhapcand ].bread = bread;
            hapcand[ nhapcand ].comp  = comp;
            hapcand[ nhapcand ].bb1   = bb1;
            hapcand[ nhapcand ].be1   = be1;
            hapcand[ nhapcand ].bb2   = bb2;
            hapcand[ nhapcand ].be2   = be2;

            nhapcand += 1;

            db->reads[ bread ].flags |= FLAG_READ_PROCESS;

            beg += 6;
        }
    }

    qsort( hapcand, nhapcand, sizeof( HapCand ), cmp_hapcand );

    int bprev = 0;
    for ( i = 0; i < nhapcand; i += 1 )
    {
        int b = hapcand[ i ].bread;

        while ( bprev <= b )
        {
            hapcandidx[ bprev ] = i;
            bprev += 1;
        }
    }

    while ( bprev != DB_NREADS( db ) + 1 )
    {
        hapcandidx[ bprev ] = i;
        bprev += 1;
    }

#ifdef VERBOSE
    printf( "loaded %" PRIu64 " haplotype breaks for validation\n", nhapcand );
#endif

    hctx->maxhapcand = maxhapcand;
    hctx->hapcand    = hapcand;
    hctx->nhapcand   = nhapcand;

    free( values );
    free( sets );
}

static void haploval_pre( PassContext* pctx, LaHaploValidateContext* hctx )
{
    UNUSED( pctx );

#ifdef VERBOSE
    printf( ANSI_COLOR_GREEN "PASS haplotype detection\n" ANSI_COLOR_RESET );
#endif

    uint32_t span_binsize = hctx->span_binsize;

    hctx->span_maxbins = ( DB_READ_MAXLEN( hctx->db ) + span_binsize - 1 ) / span_binsize;
    hctx->span_bins    = malloc( sizeof( uint64 ) * hctx->span_maxbins );
}

static void haploval_post( LaHaploValidateContext* hctx )
{
    HITS_DB*db = hctx->db;
    HapCand* hapcand = hctx->hapcand;
    uint64_t* hapcandidx = hctx->hapcandidx;
    FILE* fileHapValOut = hctx->fileHapValOut;

    uint64_t ob = 0;
    uint64_t oe = hapcandidx[ DB_NREADS(db) ];

    while ( ob < oe )
    {
        HapCand* hc = hapcand + ob;

        if ( hc->votes > 0 )
        {
            fprintf(fileHapValOut, "%" PRIu32 " %" PRIu32 " %" PRIu32 "\n",
                    hc->svid, hc->aread, hc->votes);
        }

        ob += 1;
    }

    free( hctx->span_bins );
}

static int haploval_handler( void* _ctx, Overlap* ovls, int novl )
{
    LaHaploValidateContext* hctx = (LaHaploValidateContext*)_ctx;
    uint64_t* hapcandidx         = hctx->hapcandidx;
    HapCand* hapcand             = hctx->hapcand;
    HITS_DB* db                  = hctx->db;
    uint64_t* span_bins          = hctx->span_bins;
    size_t span_maxbins          = hctx->span_maxbins;
    uint32_t span_minlr          = hctx->span_minlr;
    uint32_t span_binsize        = hctx->span_binsize;
    int aread                    = ovls->aread;

    if ( !( db->reads[ aread ].flags & FLAG_READ_PROCESS ) )
    {
        return 1;
    }

    printf( "processing %d\n", aread );

    bzero( span_bins, sizeof( uint64 ) * span_maxbins );

    // TODO: use trim track

    int i;
    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl = ovls + i;

        if ( ( ovl->flags & OVL_DISCARD ) || ( ovl->aread == ovl->bread ) )
        {
            continue;
        }

        int b = ( ovl->path.abpos + span_minlr ) / span_binsize;
        int e = ( ovl->path.aepos - span_minlr ) / span_binsize;

        while ( b < e )
        {
            span_bins[ b ] += 1;
            b += 1;
        }
    }

    uint64_t ob = hapcandidx[ aread ];
    uint64_t oe = hapcandidx[ aread + 1 ];

    while ( ob < oe )
    {
        // (svid aread bread bb1 be1 bb2 be2 comp)
        uint64_t svid  = hapcand[ ob ].svid;
        uint64_t aread = hapcand[ ob ].aread;
        uint64_t bread = hapcand[ ob ].bread;
        uint64_t bb1   = hapcand[ ob ].bb1;
        uint64_t be1   = hapcand[ ob ].be1;
        uint64_t bb2   = hapcand[ ob ].bb2;
        uint64_t be2   = hapcand[ ob ].be2;
        uint64_t comp  = hapcand[ ob ].comp;

        printf( "%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "..%" PRIu64 " %" PRIu64 "..%" PRIu64 " %" PRIu64 "\n",
                svid, aread, bread, bb1, be1, bb2, be2, comp );

#ifdef DEBUG_SHOW_SPAN
        uint64_t _be1 = be1 / span_binsize - 1;
        uint64_t _bb2 = bb2 / span_binsize + 1;

        assert( _be1 < _bb2 );

        printf( "span %" PRIu64 "..%" PRIu64, _be1, _bb2 );
        while ( _be1 <= _bb2 )
        {
            printf( " %" PRIu64 "", span_bins[ _be1 ] );
            _be1 += 1;
        }
        printf( "\n" );
#endif // DEBUG_SHOW_SPAN

        be1 = be1 / span_binsize - 1;
        bb2 = bb2 / span_binsize + 1;

        int has_gap = 0;
        while ( be1 <= bb2 )
        {
            if ( span_bins[be1] == 0 )
            {
                has_gap = 1;
                break;
            }

            be1 += 1;
        }

        if ( has_gap == 0 )
        {
            hapcand[ob].votes += 1;
        }

        ob += 1;
    }

    return 1;
}

static void usage()
{
    printf( "usage: <db> <input.las> <input.hap> <output.val.hap>\n" );
    printf( "options:\n" );
}

int main( int argc, char* argv[] )
{
    HITS_DB db;
    LaHaploValidateContext hctx;

    bzero( &hctx, sizeof( LaHaploValidateContext ) );

    // process arguments

    hctx.span_binsize = DEF_ARG_B;
    hctx.span_minlr   = DEF_ARG_L;

    opterr = 0;

    int c;
    while ( ( c = getopt( argc, argv, "b:l:" ) ) != -1 )
    {
        switch ( c )
        {
            case 'b':
                hctx.span_binsize = strtol(optarg, NULL, 10);
                break;

            case 'l':
                hctx.span_minlr = strtol(optarg, NULL, 10);
                break;

            default:
                usage();
                exit( 1 );
        }
    }

    if ( argc - optind < 4 )
    {
        usage();
        exit( 1 );
    }

    char* pcPathDbIn       = argv[ optind++ ];
    char* pcPathOverlapsIn = argv[ optind++ ];
    char* pcPathHapIn      = argv[ optind++ ];
    char* pcPathHapValOut  = argv[ optind++ ];
    FILE* fileOvlIn        = NULL;
    FILE* fileHapIn        = NULL;

    if ( Open_DB( pcPathDbIn, &db ) )
    {
        fprintf( stderr, "could not open '%s'\n", pcPathDbIn );
        exit( 1 );
    }
    hctx.db = &db;

    if ( ( fileOvlIn = fopen( pcPathOverlapsIn, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open %s\n", pcPathOverlapsIn );
        exit( 1 );
    }

    if ( ( fileHapIn = fopen( pcPathHapIn, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open %s\n", pcPathHapIn );
        exit( 1 );
    }

    if ( strcmp( pcPathHapValOut, "-" ) == 0 )
    {
        hctx.fileHapValOut = stdout;
    }
    else if ( ( hctx.fileHapValOut = fopen( pcPathHapValOut, "w" ) ) == NULL )
    {
        fprintf( stderr, "could not open %s\n", pcPathHapValOut );
        exit( 1 );
    }

    int i;
    for ( i = 0; i < DB_NREADS( &db ); i++ )
    {
        db.reads[ i ].flags = 0;
    }

    process_hap_file( &hctx, fileHapIn );
    fclose( fileHapIn );

    // init
    PassContext* pctx;

    pctx = pass_init( fileOvlIn, NULL );

    pctx->split_b      = 0;
    pctx->load_trace   = 1;
    pctx->unpack_trace = 1;
    pctx->data         = &hctx;

    haploval_pre( pctx, &hctx );
    pass( pctx, haploval_handler );
    haploval_post( &hctx );

    // cleanup

    free( hctx.hapcand ); // from process_hap_file
    free( hctx.hapcandidx );

    pass_free( pctx );
    Close_DB( hctx.db );

    return 0;
}
