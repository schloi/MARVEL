/*******************************************************************************************
 *
 * performs basic sanity checks on the overlaps
 *
 *******************************************************************************************/

#include "dalign/align.h"
#include "db/DB.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/utils.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// switches

#undef PROGRESS // show verification progress

// command line defaults

#define DEF_ARG_A 0
#define DEF_ARG_D 0
#define DEF_ARG_P 0
#define DEF_ARG_S 0

// errors

#define ERR_NOT_SORTED       ( 0x1 << 0 )
#define ERR_BAD_POS          ( 0x1 << 1 )
#define ERR_BAD_TRACE_LENGTH ( 0x1 << 2 )
#define ERR_BAD_TRACE_POINTS ( 0x1 << 3 )
#define ERR_CORRUPT          ( 0x1 << 4 )

// macros

#define CMP( a, b )      \
    cmp = ( a ) - ( b ); \
    if ( cmp != 0 )      \
        return cmp;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

// structs

typedef struct
{
    HITS_DB* db;
    ovl_header_twidth twidth;

    int error; // file didn't pass check
    int verbose;     // print a lot of stuff

    int check_ptp;   // pass through points
    int check_sort;  // sort order
    int check_dupes; // duplicates
    int check_align; // compute alignment

    int ignoreDiscardedOvls;

    ovl_header_novl novl; // Overlaps counted

    int prev_a;

    Alignment* align;      // alignment record (reused)
    Work_Data* align_work; // global alignment module work data

} CheckContext;

inline static int compare_sort( Overlap* o1, Overlap* o2 )
{
    int cmp;

    CMP( o1->aread, o2->aread );
    CMP( o1->bread, o2->bread );
    CMP( o1->flags & OVL_COMP, o2->flags & OVL_COMP );
    CMP( o1->path.abpos, o2->path.abpos );

    return cmp;
}

// used jointly with compare_sort

inline static int compare_duplicate( Overlap* o1, Overlap* o2 )
{
    int cmp;

    CMP( o1->flags, o2->flags );

    CMP( o1->path.aepos, o2->path.aepos );
    CMP( o1->path.bbpos, o2->path.bbpos );
    CMP( o1->path.bepos, o2->path.bepos );
    CMP( o1->path.tlen, o2->path.tlen );

    return cmp;
}

static void check_pre( PassContext* pctx, CheckContext* cctx )
{
    cctx->twidth = pctx->twidth;
    cctx->prev_a = 0;
}

static void check_post( PassContext* pctx, CheckContext* cctx )
{
    if ( !cctx->error && pctx->novl != cctx->novl )
    {
        fprintf( stderr, "novl of %lld doesn't match actual overlap count of %lld\n", pctx->novl, cctx->novl );
        cctx->error = 1;
    }
}

static int check_process( void* _ctx, Overlap* ovl, int novl )
{
    CheckContext* ctx = (CheckContext*)_ctx;
    HITS_DB* db       = ctx->db;
    int aread         = ovl->aread;
    int lena          = DB_READ_LEN( db, aread );
    int verbose = ctx->verbose;

    if ( ctx->check_align )
    {
        ctx->align->alen = lena;
        Load_Read( db, aread, ctx->align->aseq, 0 );
    }

    int i;
    for ( i = 0; i < novl; i++ )
    {
        Overlap* o = ovl + i;
        int bread  = o->bread;

        ctx->novl++;

        if ( verbose )
        {
            printf( "%llu %d x %d %d..%d %d..%d\n", ctx->novl, o->aread, o->bread,
                    o->path.abpos, o->path.aepos,
                    o->path.bbpos, o->path.bepos);
        }

        if ( ctx->ignoreDiscardedOvls && ( o->flags & OVL_DISCARD ) )
        {
            continue;
        }

        if ( i == 0 )
        {
            if ( ctx->check_sort && ctx->prev_a > aread )
            {
                fprintf( stderr, "overlap %lld: not sorted\n", ctx->novl );
                ctx->error |= ERR_NOT_SORTED;
            }
        }
        else
        {
            int cmp = compare_sort( ovl + ( i - 1 ), o );

            if ( cmp > 0 && ctx->check_sort )
            {
                fprintf( stderr, "overlap %lld: not sorted\n", ctx->novl );
                ctx->error |= ERR_NOT_SORTED;
            }
            else if ( cmp == 0 && ctx->check_dupes && compare_duplicate( ovl + ( i - 1 ), o ) == 0 )
            {
                fprintf( stderr, "overlap %lld: equal to previous overlap\n", ctx->novl );
                ctx->error = 1;
            }
        }

        int lenb = DB_READ_LEN( ctx->db, bread );

        if ( o->path.abpos < 0 )
        {
            fprintf( stderr, "overlap %lld: abpos < 0\n", ctx->novl );
            ctx->error |= ERR_BAD_POS;
        }

        if ( o->path.bbpos < 0 )
        {
            fprintf( stderr, "overlap %lld: bbpos < 0\n", ctx->novl );
            ctx->error |= ERR_BAD_POS;
        }

        if ( o->path.aepos > lena )
        {
            fprintf( stderr, "overlap %lld: aepos > lena\n", ctx->novl );
            ctx->error |= ERR_BAD_POS;
        }

        if ( o->path.bepos > lenb )
        {
            fprintf( stderr, "overlap %lld: bepos > lenb\n", ctx->novl );
            ctx->error |= ERR_BAD_POS;
        }

        if ( o->path.tlen < 0 )
        {
            fprintf( stderr, "overlap %lld: invalid tlen %d\n", ctx->novl, o->path.tlen );
            ctx->error |= ERR_BAD_TRACE_LENGTH;
        }

        if ( ctx->check_ptp )
        {
            ovl_trace* trace = o->path.trace;

            int apos = o->path.abpos;
            int bpos = o->path.bbpos;

            int j;
            for ( j = 0; j < o->path.tlen; j += 2 )
            {
                apos += ( apos / ctx->twidth + 1 ) * ctx->twidth;
                bpos += trace[ j + 1 ];
            }

            if ( bpos != o->path.bepos )
            {
                fprintf( stderr, "overlap %lld (%d x %d): pass-through points inconsistent be = %d (expected %d)\n",
                         ctx->novl, aread, bread, bpos, o->path.bepos );
                ctx->error |= ERR_BAD_TRACE_POINTS;
            }
        }

        if ( ctx->check_align )
        {
            ctx->align->blen = lenb;
            Load_Read( db, bread, ctx->align->bseq, 0 );

            ctx->align->path = &( o->path );

            if ( o->flags & OVL_COMP )
            {
                Complement_Seq( ctx->align->bseq, lenb );
            }

            if ( verbose )
            {
                printf("aligning\n");
            }

            Compute_Trace_PTS( ctx->align, ctx->align_work, ctx->twidth, 0, verbose );
        }
    }

    ctx->prev_a = ovl->aread;

    return ( ctx->error == 0 );
}

static void usage()
{
    fprintf( stderr, "usage: [-adhips] database input.las\n\n" );
    fprintf( stderr, "Check the contents of a .las file for consistency.\n\n" );
    fprintf( stderr, "options: -a  compute alignment based on trace points\n" );
    fprintf( stderr, "         -d  report duplicates\n" );
    fprintf( stderr, "         -h  only check headers\n" );
    fprintf( stderr, "         -i  skip overlaps tagged as discarded\n" );
    fprintf( stderr, "         -p  check pass-through points\n" );
    fprintf( stderr, "         -s  check sort order\n" );
    fprintf( stderr, "         -v  verbose" );
}

int main( int argc, char* argv[] )
{
    PassContext* pctx;
    CheckContext cctx;
    HITS_DB db;
    FILE* fileOvlIn;

    bzero( &cctx, sizeof( CheckContext ) );
    cctx.db = &db;

    // process arguments

    cctx.check_ptp           = DEF_ARG_P;
    cctx.check_sort          = DEF_ARG_S;
    cctx.check_dupes         = DEF_ARG_D;
    cctx.check_align         = DEF_ARG_A;
    cctx.ignoreDiscardedOvls = 0;

    int header_only = 0;

    int c;
    opterr = 0;

    while ( ( c = getopt( argc, argv, "adhipsv" ) ) != -1 )
    {
        switch ( c )
        {
            case 'a':
                cctx.check_align = 1;
                break;

            case 'h':
                header_only = 1;
                break;

            case 'p':
                cctx.check_ptp = 1;
                break;

            case 'd':
                cctx.check_dupes = 1;
                cctx.check_sort  = 1;
                break;

            case 's':
                cctx.check_sort = 1;
                break;

            case 'i':
                cctx.ignoreDiscardedOvls = 1;
                break;

            case 'v':
                cctx.verbose = 1;
                break;

            default:
                usage();
                exit( 1 );
        }
    }

    if ( opterr || argc - optind != 2 )
    {
        usage();
        exit( 1 );
    }

    char* pcPathReadsIn    = argv[ optind++ ];
    char* pcPathOverlapsIn = argv[ optind++ ];

    if ( ( fileOvlIn = fopen( pcPathOverlapsIn, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open '%s'\n", pcPathOverlapsIn );
        exit( 1 );
    }

    pctx = pass_init( fileOvlIn, NULL );

    if ( pctx == NULL )
    {
        fprintf( stderr, "corrupt las file '%s'\n", pcPathOverlapsIn );
        return ERR_CORRUPT;
    }

    if ( !header_only )
    {
        if ( Open_DB( pcPathReadsIn, &db ) )
        {
            fprintf( stderr, "could not open database '%s'\n", pcPathReadsIn );
            exit( 1 );
        }

        if ( cctx.check_align )
        {
            cctx.align = (Alignment*)malloc( sizeof( Alignment ) );

            cctx.align->aseq = New_Read_Buffer( &db );
            cctx.align->bseq = New_Read_Buffer( &db );

            cctx.align_work = New_Work_Data();
        }

        pctx->load_trace   = ( cctx.check_ptp || cctx.check_align );
        pctx->unpack_trace = ( cctx.check_ptp || cctx.check_align );
        pctx->data         = &cctx;

        check_pre( pctx, &cctx );

        pass( pctx, check_process );

        check_post( pctx, &cctx );

        if ( cctx.check_align )
        {
            free( cctx.align->aseq - 1 );
            free( cctx.align->bseq - 1 );

            free( cctx.align );
            Free_Work_Data( cctx.align_work );
        }

        Close_DB( &db );
    }

    pass_free( pctx );

    fclose( fileOvlIn );

    return cctx.error;
}
