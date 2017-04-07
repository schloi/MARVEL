/*******************************************************************************************
 *
 * conversion from unccorected db to corrected db and las
 *
 * Author: MARVEL Team
 *
 * Date  : March 2017
 *
 *******************************************************************************************/

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <assert.h>

#include "dalign/align.h"
#include "db/DB.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/tracks.h"

// command line defaults

#define DEF_ARG_T TRACK_TRIM
#define DEF_ARG_TT TRACK_TRIM

// switches

#undef DEBUG

// structs

typedef struct
{
    HITS_DB* db_u;
    HITS_DB* db_c;

    // uncorrected tracks

    HITS_TRACK* trim;
    HITS_TRACK* postrace;
    HITS_TRACK* source;

    track_anno* trim_a_c;
    track_data* trim_d_c;

    int* amap;
    int* bmap;

    char* nameTrim_c;

    int tbytes;
    ovl_header_twidth twidth;

    int* riduc_to_ridc;

    FILE* fileLas_c;
} ConvertContext;

static void trace_to_posmap( int32_t* trace, int tlen, int alen, int* posmap )
{
    int t, a, ac, p;
    a = ac = 0;

    for ( t = 0; t < tlen; t++ )
    {
        p = trace[ t ];

        if ( p < 0 )
        {
            p = -p - 1;

            while ( a < p )
            {
                posmap[ a ] = ac;

                a += 1;
                ac += 1;
            }

            ac += 1;
        }
        else
        {
            p--;

            while ( ac < p )
            {
                posmap[ a ] = ac;

                a += 1;
                ac += 1;
            }

            posmap[ a ] = -1;
            a += 1;
        }
    }

    p = alen;
    while ( a < p )
    {
        posmap[ a ] = ac;

        a += 1;
        ac += 1;
    }
}

static void convert_pre( PassContext* pctx, ConvertContext* cctx )
{
    ovl_header_write( cctx->fileLas_c, pctx->novl, pctx->twidth );

    cctx->tbytes = pctx->tbytes;
    cctx->amap = malloc(sizeof(int) * cctx->db_u->maxlen);
    cctx->bmap = malloc(sizeof(int) * cctx->db_u->maxlen);

    // mapping of uncorrected read ids to corrected rids

    cctx->riduc_to_ridc = malloc(sizeof(int) * cctx->db_u->nreads);

    int i;
    for (i = 0; i < cctx->db_u->nreads; i++)
    {
        cctx->riduc_to_ridc[i] = -1;
    }

    track_data* data = cctx->source->data;
    track_anno* anno = cctx->source->anno;

    for (i = 0; i < cctx->db_c->nreads; i++)
    {
        track_anno b = anno[i] / sizeof(track_data);
        track_anno e = anno[i+1] / sizeof(track_data);

        assert( b < e );

        cctx->riduc_to_ridc[ data[b] ] = i;
    }

    // convert trim track

    data = cctx->trim->data;
    anno = cctx->trim->anno;

    track_anno* anno_c = cctx->trim_a_c = malloc(sizeof(track_anno) * (cctx->db_c->nreads + 1));
    track_data* data_c = cctx->trim_d_c = malloc(sizeof(track_data) * (cctx->db_c->nreads * 2));

    track_anno* anno_pos_c = cctx->postrace->anno;
    track_data* data_pos_c = cctx->postrace->data;

    int* amap = cctx->amap;

    track_anno dcur = anno_c[0] = 0;

    for ( i = 0; i < cctx->db_u->nreads; i++)
    {
        int rc = cctx->riduc_to_ridc[i];

        if (rc == -1)
        {
            continue;
        }

        track_anno b = anno[i] / sizeof(track_data);
        track_anno e = anno[i + 1] / sizeof(track_data);

        assert( b + 2 == e );

        track_data tb = data[b];
        track_data te = data[b+1];
        track_data tb_c = -1;
        track_data te_c = -1;

        if (tb == te)
        {
            tb_c = 0;
            te_c = 0;
        }
        else
        {
            // create pos mapping

            track_anno ab = anno_pos_c[ rc ] / sizeof(track_data);
            track_anno ae = anno_pos_c[ rc + 1] / sizeof(track_data);

            trace_to_posmap((int32_t*)(data_pos_c + ab), ae - ab, DB_READ_LEN(cctx->db_u, i), amap);

            while ( (tb_c = amap[tb]) == -1 )
            {
                tb++;
            }

            te--;

            while ( (te_c = amap[te]) == -1 )
            {
                te--;
            }

            te_c++;

        }

        // printf("mapping %5d..%5d -> %5d..%5d\n", data[b], data[b+1], tb_c, te_c);

        data_c[dcur++] = tb_c;
        data_c[dcur++] = te_c;

        anno_c[rc + 1] = sizeof(track_data) * dcur;
    }
}

static void convert_post( PassContext* pctx, ConvertContext* cctx )
{
    int nreads = cctx->db_c->nreads;

    track_write(cctx->db_c, cctx->nameTrim_c, 0, cctx->trim_a_c, cctx->trim_d_c, cctx->trim_a_c[nreads] / sizeof(track_data));

    free(cctx->amap);
    free(cctx->bmap);

    free(cctx->riduc_to_ridc);

    free(cctx->trim_a_c);
    free(cctx->trim_d_c);
}

static int convert_process( void* _ctx, Overlap* ovl, int novl )
{
    ConvertContext* ctx = (ConvertContext*)_ctx;
    FILE* fileLas_c     = ctx->fileLas_c;
    int tbytes          = ctx->tbytes;
    int a_u = ovl->aread;
    int a_c = ctx->riduc_to_ridc[a_u];

    track_anno* anno = ctx->postrace->anno;
    track_data* data = ctx->postrace->data;

    track_anno ab = anno[ a_c ] / sizeof(track_data);
    track_anno ae = anno[ a_c + 1] / sizeof(track_data);

    int* amap = ctx->amap;
    int* bmap = ctx->bmap;

    trace_to_posmap((int32_t*)(data + ab), ae - ab, DB_READ_LEN(ctx->db_u, a_u), amap);

    int i;
    for ( i = 0; i < novl; i++ )
    {
        Overlap* o = ovl + i;

        if (o->flags & OVL_DISCARD)
        {
            continue;
        }

#ifdef DEBUG
        printf("%5d x %5d | %5d..%5d %5d -> %5d..%5d %5d\n",
                a_u, o->bread,
                o->path.abpos, o->path.aepos, DB_READ_LEN(ctx->db_u, a_u),
                o->path.bbpos, o->path.bepos, DB_READ_LEN(ctx->db_u, o->bread));
#endif

        o->path.tlen = 0;

        int b_u = o->bread;
        int b_c = ctx->riduc_to_ridc[ b_u ];

        track_anno tbb = anno[ b_c ] / sizeof(track_data);
        track_anno tbe = anno[ b_c + 1] / sizeof(track_data);
        trace_to_posmap((int32_t*)(data + tbb), tbe - tbb, DB_READ_LEN(ctx->db_u, b_u), bmap);

        int ab_u = o->path.abpos;
        int ae_u = o->path.aepos - 1;
        int ab_c = -1;
        int ae_c = -1;

        while ( (ab_c = amap[ ab_u ]) == -1 )
        {
            ab_u++;
        }

        while ( (ae_c = amap[ ae_u ]) == -1 )
        {
            ae_u--;
        }

        ae_c++;

        o->path.abpos = ab_c;
        o->path.aepos = ae_c;

#ifdef DEBUG
        printf("%5d         | %5d..%5d %5d\n",
                a_c,
                ab_c, ae_c, DB_READ_LEN(ctx->db_c, a_c));
#endif

        assert( 0 <= ab_c && ab_c < ae_c && ae_c <= DB_READ_LEN(ctx->db_c, a_c) );

        int bb_u;
        int be_u;
        int bb_c = -1;
        int be_c = -1;

        if ( o->flags & OVL_COMP )
        {
            int blen_u = DB_READ_LEN(ctx->db_u, b_u);
            bb_u = blen_u - o->path.bepos;
            be_u = blen_u - o->path.bbpos - 1;
        }
        else
        {
            bb_u = o->path.bbpos;
            be_u = o->path.bepos - 1;
        }

        while ( (bb_c = bmap[ bb_u ]) == -1 )
        {
            bb_u++;
        }

        while ( (be_c = bmap[ be_u ]) == -1 )
        {
            be_u--;
        }

        if ( o->flags & OVL_COMP )
        {
            int blen_c = DB_READ_LEN(ctx->db_c, b_c);
            o->path.bbpos = blen_c - be_c - 1;
            o->path.bepos = blen_c - bb_c;

#ifdef DEBUG
            printf("%5d         <                       %5d..%5d (%5d..%5d) %5d\n",
                    b_c,
                    o->path.bbpos, o->path.bepos, bb_c, be_c, DB_READ_LEN(ctx->db_c, b_c));
#endif
        }
        else
        {
            o->path.bbpos = bb_c;
            o->path.bepos = be_c + 1;

#ifdef DEBUG
            printf("%5d         >                       %5d..%5d %5d\n",
                    b_c,
                    o->path.bbpos, o->path.bepos, DB_READ_LEN(ctx->db_c, b_c));
#endif
        }

        o->aread = a_c;
        o->bread = b_c;

        assert( 0 <= o->path.bbpos && o->path.bbpos < o->path.bepos && o->path.bepos <= DB_READ_LEN(ctx->db_c, b_c) );

        Write_Overlap( fileLas_c, o, tbytes );
    }

    return 1;
}

static void usage()
{
    fprintf( stderr, "usage  : [-tT <track>] <uncorrected.db> <uncorrected.las> <corrected.db> <corrected.las>\n" );
    fprintf( stderr, "options: -t ... uncorrected db trim track (%s)\n", DEF_ARG_T );
    fprintf( stderr, "         -T ... corrected db trim track (%s)\n", DEF_ARG_TT );
}

int main( int argc, char* argv[] )
{
    PassContext* pctx;
    ConvertContext cctx;
    HITS_DB db_u, db_c;
    FILE* fileLas_u;

    bzero( &cctx, sizeof( ConvertContext ) );
    cctx.db_u = &db_u;
    cctx.db_c = &db_c;

    // process arguments

    char* trim_u    = DEF_ARG_T;
    cctx.nameTrim_c = DEF_ARG_TT;

    int c;
    opterr = 0;

    while ( ( c = getopt( argc, argv, "t:T:" ) ) != -1 )
    {
        switch ( c )
        {
            case 't':
                trim_u = optarg;
                break;

            case 'T':
                cctx.nameTrim_c = optarg;
                break;

            default:
                usage();
                exit( 1 );
        }
    }

    if ( opterr || argc - optind != 4 )
    {
        usage();
        exit( 1 );
    }

    char* pathdb_u  = argv[ optind++ ];
    char* pathlas_u = argv[ optind++ ];
    char* pathdb_c  = argv[ optind++ ];
    char* pathlas_c = argv[ optind++ ];

    if ( ( fileLas_u = fopen( pathlas_u, "r" ) ) == NULL )
    {
        fprintf( stderr, "could not open '%s'\n", pathlas_u );
        exit( 1 );
    }

    if ( ( cctx.fileLas_c = fopen( pathlas_c, "w" ) ) == NULL )
    {
        fprintf( stderr, "could not open '%s'\n", pathlas_c );
        exit( 1 );
    }

    if ( Open_DB( pathdb_u, &db_u ) )
    {
        fprintf( stderr, "could not open database '%s'\n", pathdb_u );
        exit( 1 );
    }

    if ( Open_DB( pathdb_c, &db_c ) )
    {
        fprintf( stderr, "could not open database '%s'\n", pathdb_c );
        exit( 1 );
    }

    if ( !( cctx.trim = track_load( &db_u, trim_u ) ) )
    {
        fprintf( stderr, "failed load track %s", trim_u );
    }

    if ( !( cctx.postrace = track_load( &db_c, "postrace" ) ) )
    {
        fprintf( stderr, "failed load track %s", "postrace" );
    }

    if ( !( cctx.source = track_load( &db_c, "source" ) ) )
    {
        fprintf( stderr, "failed load track %s", "source" );
    }

    pctx             = pass_init( fileLas_u, NULL );
    pctx->split_b    = 0;
    pctx->load_trace = 0;
    pctx->data       = &cctx;

    convert_pre( pctx, &cctx );

    pass( pctx, convert_process );

    convert_post( pctx, &cctx );

    pass_free( pctx );

    Close_DB( &db_u );
    Close_DB( &db_c );

    fclose( fileLas_u );
    fclose( cctx.fileLas_c );

    return 0;
}
