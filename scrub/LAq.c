/*******************************************************************************************
 *
 *  calculate the quality and trim track
 *
 *  Date   : October 2015
 *
 *  Author : MARVEL Team
 *
 *******************************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/param.h>
#include <unistd.h>

#include "lib/oflags.h"
#include "lib/colors.h"
#include "lib/stats.h"
#include "lib/tracks.h"
#include "lib/pass.h"
#include "lib/utils.h"

#include "db/DB.h"
#include "dalign/align.h"

// toggles

#define VERBOSE
#undef DEBUG_Q

#define TRIM_WINDOW        5

// command line defaults

#define DEF_ARG_B          0
#define DEF_ARG_U          0
#define DEF_ARG_O       1000
#define DEF_ARG_D         25

#define DEF_ARG_S          1
#define DEF_ARG_SS        20

#define DEF_ARG_T   	TRACK_TRIM
#define DEF_ARG_Q   	TRACK_Q


// structs

typedef struct
{
    HITS_DB* db;

    int tblock;             // db block
    int min_trimmed_len;    // min length of read after trimming

    int twidth;             // trace point spacing

    unsigned int segmin;    // min number of segments for q estimate
    unsigned int segmax;

    uint32* q_histo;
    size_t q_histo_len;

    track_anno* q_anno;
    track_data* q_data;
    int q_dprev, q_dcur, q_dmax;

    // re-annotate only
    HITS_TRACK* q_track;        // q track
    HITS_TRACK* trim_track;     // trim track
    int trim_q;                 // trimming quality cutoff

    char *track_trim_in;
    char *track_trim_out;

    char *track_q_in;
    char *track_q_out;

    track_anno* trim_anno;
    track_data* trim_data;
    track_anno tcur;
} AnnotateContext;

// for getopt()

extern char* optarg;
extern int optind, opterr, optopt;

static int trim_q_offsets(AnnotateContext* actx, int rid, int rlen, track_data* dataq, track_anno* annoq, int* trim_b, int* trim_e)
{
    int min_trimmed_len = actx->min_trimmed_len;
    int twidth = actx->twidth;
    int left = (*trim_b) ? (*trim_b) / twidth : 0;
    int right = (*trim_e) ? (*trim_e) / twidth : ( rlen + twidth - 1 ) / twidth;

    track_anno ob = annoq[rid] / sizeof(track_data);
    track_anno oe = annoq[rid + 1] / sizeof(track_data);

    int trim_q = actx->trim_q;

    if (ob >= oe)
    {
        *trim_b = *trim_e = 0;
        return 0;
    }

#ifdef DEBUG_Q
    {
        track_anno k;
        printf("Q");
        for (k = ob; k < oe; k++)
        {
            if (dataq[k] == 0 || dataq[k] > trim_q)
            {
                printf(ANSI_COLOR_RED " %2d" ANSI_COLOR_RESET, dataq[k]);
            }
            else
            {
                printf(" %2d", dataq[k]);
            }
        }
        printf("\n");
    }
#endif

    ob += left;
    oe -= ( rlen + twidth - 1 ) / twidth - right;

    track_anno wnd_left;
    int sum = 0;

    for (wnd_left = ob ; wnd_left - ob <= TRIM_WINDOW && ob < oe ; wnd_left++)
    {
        track_data q = dataq[wnd_left];

        if (q >= trim_q || q == 0)
        {
            // printf("reset left\n");

            ob = wnd_left + 1;
            sum = 0;
            continue;
        }

        if (wnd_left - ob == TRIM_WINDOW && sum / TRIM_WINDOW >= trim_q)
        {
            // printf("move left %d\n", sum / TRIM_WINDOW);

            sum -= dataq[ob];
            ob++;
        }

        sum += q;
    }

    track_anno wnd_right;
    sum = 0;

    for (wnd_right = oe ; oe - wnd_right <= TRIM_WINDOW && ob < oe ; wnd_right--)
    {
        track_data q = dataq[wnd_right - 1];

        if (q >= trim_q || q == 0)
        {
            // printf("reset right q %d oe %llu wnd_right %llu\n", q, oe, wnd_right);

            oe = wnd_right - 1;
            sum = 0;
            continue;
        }

        if (oe - wnd_right == TRIM_WINDOW && sum / TRIM_WINDOW >= trim_q)
        {
            // printf("move right %d\n", sum / TRIM_WINDOW);

            sum -= dataq[oe];
            oe--;
        }

        sum += q;
    }

    int tb = MIN( rlen, ((int) (ob - annoq[rid] / sizeof(track_data))) * twidth );
    int te = MIN( rlen, ((int) (oe - annoq[rid] / sizeof(track_data))) * twidth );

    if (te - tb < min_trimmed_len)
    {
        tb = te = 0;
    }

    *trim_b = tb;
    *trim_e = te;

    return 1;
}

static void calculate_trim(AnnotateContext* actx)
{
    int dcur = 0;
    int dmax = 1000;
    track_data* data = (track_data*)malloc( sizeof(track_data) * dmax );

    track_anno* anno = (track_anno*)malloc( sizeof(track_anno) * (DB_NREADS(actx->db) + 1) );
    bzero(anno, sizeof(track_anno) * (DB_NREADS(actx->db) + 1));

    track_data* dataq = actx->q_data;
    track_anno* annoq = actx->q_anno;

    int a;
    for (a = 0; a < actx->db->nreads; a++)
    {
        int alen = DB_READ_LEN(actx->db, a);

        int tb = 0;
        int te = 0;

        if (trim_q_offsets(actx, a, alen, dataq, annoq, &tb, &te) )
        {
            // assert(dbg_tb == data[dcur-2] && dbg_te == data[dcur-1]);

            if (dcur + 2 >= dmax)
            {
                dmax = dmax * 1.2 + 1000;
                data = (track_data*)realloc(data, sizeof(track_data) * dmax);
            }

            data[dcur++] = tb;
            data[dcur++] = te;

            anno[ a ] += 2 * sizeof(track_data);

#ifdef DEBUG_Q
            printf("trimming %6d (%5d) to %5d..%5d\n",
                        a, alen,
                        tb, te);
#endif
       }
    }

    track_anno off = 0;
    track_anno coff;

    int j;
    for (j = 0; j <= DB_NREADS(actx->db); j++)
    {
        coff = anno[j];
        anno[j] = off;
        off += coff;
    }

    track_write(actx->db, actx->track_trim_out, actx->tblock, anno, data, dcur);

    free(data);
    free(anno);
}


static void pre_annotate(PassContext* pctx, AnnotateContext* ctx)
{
#ifdef VERBOSE
    printf(ANSI_COLOR_GREEN "PASS quality estimate and trimming" ANSI_COLOR_RESET "\n");
#endif

    ctx->twidth = pctx->twidth;

    ctx->q_anno = (track_anno*)malloc(sizeof(track_anno)*(DB_NREADS(ctx->db)+1));
    bzero(ctx->q_anno, sizeof(track_anno)*(DB_NREADS(ctx->db)+1));

    ctx->q_dmax = DB_NREADS(ctx->db);
    ctx->q_dcur = ctx->q_dprev = 0;
    ctx->q_data = (track_data*)malloc(sizeof(track_data)*ctx->q_dmax);

    int maxtiles = (DB_READ_MAXLEN(ctx->db) + ctx->twidth - 1) / ctx->twidth;

    ctx->q_histo_len = ctx->twidth * 2 * maxtiles;
    ctx->q_histo = malloc( sizeof(uint32) * ctx->q_histo_len );
}

static void post_annotate(AnnotateContext* ctx)
{
    int j;
    track_anno qoff, coff;

    qoff = 0;

    for (j = 0; j <= DB_NREADS(ctx->db); j++)
    {
        coff = ctx->q_anno[j];
        ctx->q_anno[j] = qoff;
        qoff += coff;
    }

    assert( qoff / sizeof(track_data) == (uint64_t)ctx->q_dcur );

    calculate_trim(ctx);

    track_write(ctx->db, ctx->track_q_out, ctx->tblock, ctx->q_anno, ctx->q_data, ctx->q_dcur);

    free(ctx->q_anno);
    free(ctx->q_data);

    free(ctx->q_histo);
}

static int handler_annotate(void* _ctx, Overlap* ovls, int novl)
{
    AnnotateContext* ctx = (AnnotateContext*)_ctx;

    unsigned int segmin = ctx->segmin;
    unsigned int segmax = ctx->segmax;

    int a = ovls->aread;
    int alen = DB_READ_LEN(ctx->db, a);
    int ntiles = (alen + ctx->twidth - 1) / ctx->twidth;

    bzero(ctx->q_histo, ctx->q_histo_len * sizeof(uint32));

    int i;
    for (i = 0; i < novl; i++)
    {
        Overlap* ovl = ovls + i;

        if ( ovl->aread == ovl->bread )
        {
            continue;
        }

        int comp = (ovl->flags & OVL_COMP) ? 1 : 0;

        int tile = ovl->path.abpos / ctx->twidth;
        ovl_trace* trace = ovl->path.trace;

        if ( (ovl->path.abpos % ctx->twidth) == 0 )
        {
            int q = trace[0];
            ctx->q_histo[ ctx->twidth * 2 * tile + 2 * q + comp] += 1;
        }

        tile++;

        int t;
        for (t = 2; t < ovls[i].path.tlen - 2; t+=2)
        {
            int q = trace[t];
            ctx->q_histo[ ctx->twidth * 2 * tile + 2 * q + comp] += 1;

            tile += 1;
        }

        if ( (ovl->path.aepos % ctx->twidth) == 0 || ovl->path.aepos == alen )
        {
            int q = trace[t];
            ctx->q_histo[ ctx->twidth * 2 * tile + 2 * q + comp] += 1;
        }
    }

    // estimate mean q

    if (ctx->q_dcur + ntiles >= ctx->q_dmax)
    {
        ctx->q_dmax = ctx->q_dmax * 1.2 + ntiles;
        ctx->q_data = realloc(ctx->q_data, sizeof(track_data) * ctx->q_dmax);
    }

    for (i = 0; i < ntiles; i++)
    {
        uint32* tile_qhisto = ctx->q_histo + 2 * ctx->twidth * i;
        uint32 sum = 0;
        uint32 count = 0;

        int q;

        for ( q = 0 ; q < ctx->twidth && count != segmax ; q++ )
        {
            uint32 has = MIN(tile_qhisto[2 * q] + tile_qhisto[2 * q + 1], segmax - count);
            count += has;
            sum += has * q;
        }

        if (count < segmin)
        {
            q = 0;
        }
        else
        {
            if (sum == 0)
            {
                sum = count;
            }

            q = (int)( ((float)sum)/count + 0.5 );
        }

        ctx->q_data[ ctx->q_dcur++ ] = q;
        ctx->q_anno[ a ] += 1 * sizeof(track_data);
    }

    return 1;
}

static void pre_update_anno(PassContext* pctx, AnnotateContext* actx)
{
#ifdef VERBOSE
    printf(ANSI_COLOR_GREEN "PASS update quality estimate and trimming" ANSI_COLOR_RESET "\n");
#endif

    actx->q_track = track_load(actx->db, actx->track_q_in);

    if (!actx->q_track)
    {
        fprintf(stderr, "could not open %s track\n", actx->track_q_in);
        exit(1);
    }

    actx->trim_track = track_load(actx->db, actx->track_trim_in);

    if (!actx->trim_track)
    {
        fprintf(stderr, "could not open %s track\n", actx->track_trim_in);
        exit(1);
    }

    int nreads = DB_NREADS(actx->db);

    actx->twidth = pctx->twidth;

    actx->trim_anno = (track_anno*)malloc(sizeof(track_anno) * (nreads + 1));
    actx->trim_data = (track_data*)malloc( ((track_anno*)actx->trim_track->anno)[ nreads ] );
    actx->tcur = 0;

    bzero(actx->trim_anno, sizeof(track_anno) * (nreads + 1));
}

static void post_update_anno(AnnotateContext* actx)
{
    int j;
    track_anno qoff, coff;

    qoff = 0;

    for (j = 0; j <= DB_NREADS(actx->db); j++)
    {
        coff = actx->trim_anno[j];
        actx->trim_anno[j] = qoff;
        qoff += coff;
    }

    track_write(actx->db, actx->track_trim_out, actx->tblock, actx->trim_anno, actx->trim_data, actx->tcur);

    free(actx->trim_anno);
    free(actx->trim_data);
}

static int handler_update_anno(void* _ctx, Overlap* ovls, int novl)
{
    AnnotateContext* actx = _ctx;
    track_anno* trim_anno = actx->trim_anno;
    track_data* trim_data = actx->trim_data;

    int a = ovls->aread;

    track_anno* annoq = actx->q_track->anno;
    track_data* dataq = actx->q_track->data;

    int ab_min = INT_MAX;
    int ae_max = 0;

    int i;
    for (i = 0; i < novl; i++)
    {
        if (ovls[i].flags & OVL_DISCARD)
        {
            continue;
        }

        if (ovls[i].aread == ovls[i].bread)
        {
            continue;
        }

        ab_min = MIN(ab_min, ovls[i].path.abpos);
        ae_max = MAX(ae_max, ovls[i].path.aepos);
    }

    track_anno ob = ((track_anno*)actx->trim_track->anno)[a] / sizeof(track_data);
    track_anno oe = ((track_anno*)actx->trim_track->anno)[a + 1] / sizeof(track_data);

    assert( ob + 2 == oe );

    int tb, te, tb_new, te_new;

    tb_new = tb = ((track_data*)actx->trim_track->data)[ob];
    te_new = te = ((track_data*)actx->trim_track->data)[ob + 1];

    // tighten trim

    if (tb < ab_min || te > ae_max)
    {
        int alen = DB_READ_LEN(actx->db, ovls->aread);

#ifdef DEBUG_Q
        printf("READ %6d (%5d) ... CURRENT %5d..%5d ... NEW %5d..%5d\n", a, alen, tb, te, ab_min, ae_max);
#endif

        if (ab_min == INT_MAX)
        {
            tb_new = te_new = 0;
        }
        else
        {
            tb_new = ab_min + actx->twidth - 1;
            te_new = ae_max; //  - actx->twidth + 1;

            trim_q_offsets(actx, a, alen, dataq, annoq, &tb_new, &te_new);

            /*
            if (ae_max == alen)
            {
                te_new = alen;
            }
            */
        }

#ifdef DEBUG_Q
        printf(" ... UPDATED %5d..%5d\n\n", tb_new, te_new);
#endif
    }

    trim_data[ actx->tcur++ ] = tb_new;
    trim_data[ actx->tcur++ ] = te_new;
    trim_anno[ a ] += 2 * sizeof(track_data);

    return 1;
}

static void usage()
{
    fprintf( stderr, "usage: [-u] [-b n] [-d n] [-s n] [-S n] [-t track] [-T track] [-q track] [-Q track] database input.las\n\n" );

    fprintf( stderr, "Creates an annotation track containing the reads' qualities and computes trim information.\n\n" );

    fprintf( stderr, "options: -b n      block number\n" );
    fprintf( stderr, "         -d n      trim reads based on this quality cutoff (default %d)\n", DEF_ARG_D );

    fprintf( stderr, "         -s n      minimum number of segments for quality estimate (default %d)\n", DEF_ARG_S );
    fprintf( stderr, "         -S n      maximum number of segments for quality estimate (default %d)\n", DEF_ARG_SS );

    fprintf( stderr, "         -o n      minimum overlap length after trim (default %d)\n", DEF_ARG_O );

    fprintf( stderr, "         -u        update existing trim track\n" );

    fprintf( stderr, "         -t track  input trim track in -u mode (default %s)\n", DEF_ARG_T );
    fprintf( stderr, "         -T track  output trim track (default %s)\n", DEF_ARG_T );

    fprintf( stderr, "         -q track  input quality track in -u mode (default %s)\n", DEF_ARG_Q );
    fprintf( stderr, "         -Q track  output quality track (default %s)\n", DEF_ARG_Q );
}

int main(int argc, char* argv[])
{
    HITS_DB db;
    PassContext* pctx;
    AnnotateContext actx;
    FILE* fileOvlIn;

    bzero(&actx, sizeof(AnnotateContext));
    actx.db = &db;

    // process arguments

    int arg_u = DEF_ARG_U;
    char* qlog = NULL;

    opterr = 0;

    actx.tblock = DEF_ARG_B;
    actx.min_trimmed_len = DEF_ARG_O;
    actx.trim_q = DEF_ARG_D;
    actx.segmin = DEF_ARG_S;
    actx.segmax = DEF_ARG_SS;
    actx.track_trim_in = DEF_ARG_T;
    actx.track_trim_out = DEF_ARG_T;
    actx.track_q_in = DEF_ARG_Q;
    actx.track_q_out = DEF_ARG_Q;

    int c;
    while ((c = getopt(argc, argv, "s:S:o:ub:d:L:t:T:q:Q:")) != -1)
    {
        switch (c)
        {
            case 's':
                      actx.segmin = atoi(optarg);
                      break;

            case 'S':
                      actx.segmax = atoi(optarg);
                      break;

            case 'L':
                      qlog = optarg;
                      break;

            case 'd':
                      actx.trim_q = atoi(optarg);
                      break;

            case 'o':
                      actx.min_trimmed_len = atoi(optarg);
                      break;

            case 'u':
                      arg_u = 1;
                      break;

            case 'b':
                      actx.tblock = atoi(optarg);
                      break;

            case 't':
                      actx.track_trim_in = optarg;
                      break;

            case 'T':
                      actx.track_trim_out = optarg;
                      break;

            case 'q':
                      actx.track_q_in = optarg;
                      break;

            case 'Q':
                      actx.track_q_out = optarg;
                      break;

            default:
                      usage();
                      exit(1);
        }
    }

    if (argc - optind != 2)
    {
        usage();
        exit(1);
    }

    if (actx.trim_q == 0)
    {
        fprintf(stderr, "error: -q not specified\n");
        exit(1);
    }

    if (actx.segmin < 1)
    {
        fprintf(stderr, "error: invalid -s\n");
        exit(1);
    }

    if (actx.segmin > actx.segmax)
    {
        fprintf(stderr, "error: invalid -s -S combination\n");
        exit(1);
    }

    if (actx.track_q_in != NULL && arg_u == 0)
    {
        fprintf( stderr, "error: -q specified without -u\n" );
        exit( 1 );
    }

    char* pcPathReadsIn = argv[optind++];
    char* pcPathOverlaps = argv[optind++];

    if ( (fileOvlIn = fopen(pcPathOverlaps, "r")) == NULL )
    {
        fprintf(stderr, "could not open '%s'\n", pcPathOverlaps);
        exit(1);
    }

    // init

    if (Open_DB(pcPathReadsIn, &db))
    {
        fprintf(stderr, "failed to open %s\n", pcPathReadsIn);
        exit(1);
    }

    pctx = pass_init(fileOvlIn, NULL);

    pctx->split_b = 0;
    pctx->data = &actx;

    pctx->load_trace = 1;
    pctx->unpack_trace = 1;

    // passes

    // update existing trim track
    if (arg_u)
    {
        pre_update_anno(pctx, &actx);
        pass(pctx, handler_update_anno);
        post_update_anno(&actx);
    }
    else
    {
        pre_annotate(pctx, &actx);
        pass(pctx, handler_annotate);
        post_annotate(&actx);
    }

    if (qlog != NULL)
    {
        FILE* fileq = fopen(qlog, "w");

        if (fileq)
        {
            fprintf(fileq, "%d\n", actx.trim_q);
            fclose(fileq);
        }
        else
        {
            fprintf(stderr, "error: failed to open %s\n", qlog);
        }

    }

    // cleanup

    pass_free(pctx);

    fclose(fileOvlIn);

    Close_DB(&db);

    return 0;
}
