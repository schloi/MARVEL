/*******************************************************************************************
 *
 *  creates a repeat annotation track (named -t) based on local alignments
 *
 *  Date   :  February 2016
 *
 *  Author :  MARVEL Team
 *
 *******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/param.h>
#include <assert.h>
#include <unistd.h>

#include "lib/tracks.h"
#include "lib/colors.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/utils.h"

#include "db/DB.h"
#include "dalign/align.h"


// constants

#define DEF_ARG_H 5
#define DEF_ARG_L 3

#define DEF_ARG_T "lrepeats"

// toggles

#define VERBOSE
#undef DEBUG_LOCAL

// macros

typedef struct
{
    HITS_DB* db;
    HITS_TRACK* track_trim;

    // arguments

    int cnt_enter;
    int cnt_leave;
    char* rp_track;
    int rp_block;

    int merge_distance;

    // repeat pass

    int rp_emax;
    int rp_dmax;
    int rp_dcur;

    track_data* rp_data;
    track_anno* rp_anno;

    int* rp_events;

    uint64_t stats_bases;
    uint64_t stats_repeat_bases;
    uint64_t stats_merged;

} RepeatContext;

extern char* optarg;
extern int optind, opterr, optopt;

static int cmp_repeats_events(const void* x, const void* y)
{
    int* e1 = (int*)x;
    int* e2 = (int*)y;

    int cmp = abs(*e1) - abs(*e2);

    if (cmp == 0)
    {
        cmp = (*e1) - (*e2);
    }

    return cmp;
}

static void pre_repeats(RepeatContext* ctx)
{
#ifdef VERBOSE
    printf(ANSI_COLOR_GREEN "PASS repeats\n" ANSI_COLOR_RESET);
#endif

    ctx->track_trim = track_load(ctx->db, TRACK_TRIM);
    if (!ctx->track_trim)
    {
        fprintf(stderr, "error: failed to load track %s\n", TRACK_TRIM);
        exit(1);
    }

    ctx->rp_emax = 100;
    ctx->rp_events = (int*)malloc(sizeof(int) * ctx->rp_emax);
    ctx->rp_anno = (track_anno*)malloc(sizeof(track_anno) * ( DB_NREADS(ctx->db) + 1));
    bzero(ctx->rp_anno, sizeof(track_anno)*( DB_NREADS(ctx->db) + 1));

    ctx->rp_dcur = 0;
    ctx->rp_dmax = 100;
    ctx->rp_data = (track_data*)malloc(sizeof(track_data)*ctx->rp_dmax);
}

static void post_repeats(RepeatContext* ctx)
{
    int j;

    track_anno coff, off;
    off = 0;

    for (j = 0; j <= DB_NREADS(ctx->db); j++)
    {
        coff = ctx->rp_anno[j];
        ctx->rp_anno[j] = off;
        off += coff;
    }

    track_write(ctx->db, ctx->rp_track, ctx->rp_block, ctx->rp_anno, ctx->rp_data, ctx->rp_dcur);

    free(ctx->rp_anno);
    free(ctx->rp_data);
    free(ctx->rp_events);

#ifdef VERBOSE
    printf("BASES_TOTAL %" PRIu64 "\n", ctx->stats_bases);
    printf("BASES_REPEAT %" PRIu64 "\n", ctx->stats_repeat_bases);
    printf("BASES_REPEAT_PERCENT %d%%\n", (int)(ctx->stats_repeat_bases*100.0/ctx->stats_bases));
#endif
}

static int handler_repeats(void* _ctx, Overlap* ovls, int novl)
{
    RepeatContext* ctx = (RepeatContext*)_ctx;

    int a = ovls->aread;
    int alen = DB_READ_LEN(ctx->db, a);
    int trim_ab, trim_ae;

    get_trim(ctx->db, ctx->track_trim, a, &trim_ab, &trim_ae);


    if (trim_ab >= trim_ae)
    {
        return 1;
    }

    ctx->stats_bases += alen;

    if (2 * novl > ctx->rp_emax)
    {
        ctx->rp_emax = 1.2 * ctx->rp_emax + 2 * novl;
        ctx->rp_events = (int*)realloc(ctx->rp_events, sizeof(int) * ctx->rp_emax);
    }

    int i;
    int j = 0;
    for (i = 0; i < novl; i++)
    {
        Overlap* ovl = ovls + i;

        if (ovl->flags & OVL_CONT)
        {
            continue;
        }

        if(ovl->aread == ovl->bread)
        {
            continue;
        }

        int trim_bb, trim_be;
        get_trim(ctx->db, ctx->track_trim, ovl->bread, &trim_bb, &trim_be);

        if (trim_bb >= trim_be)
        {
            continue;
        }

        if ( ovl->flags & OVL_COMP )
        {
            int blen = DB_READ_LEN(ctx->db, ovl->bread);

            int t = trim_bb;
            trim_bb = blen - trim_be;
            trim_be = blen - t;
        }

        if ( ((ovl->path.abpos - trim_ab) > 0 && (ovl->path.bbpos - trim_bb) > 0) ||
             ((trim_ae - ovl->path.aepos) > 0 && (trim_be - ovl->path.bepos) > 0) )
        {
#ifdef DEBUG_LOCAL
            printf("%7d ALN %5d..%5d x %5d..%5d TRIM %5d..%5d %5d..%5d\n", a,
                    ovl->path.abpos, ovl->path.aepos, ovl->path.bbpos, ovl->path.bepos,
                    trim_ab, trim_ae, trim_bb, trim_be);
#endif
            ctx->rp_events[j++] = ovl->path.abpos;
            ctx->rp_events[j++] = -(ovl->path.aepos-1);
        }
    }

    novl = j/2;

    qsort(ctx->rp_events, 2*novl, sizeof(int), cmp_repeats_events);

    int span = 0;
    int span_leave = ctx->cnt_leave;
    int span_enter = ctx->cnt_enter;

    int span_max = 0;

    int in_repeat = 0;
    int rp_dcur = ctx->rp_dcur;

    for (i = 0; i < 2*novl; i++)
    {
        if (ctx->rp_events[i] < 0) span--;
        else span++;

        if (span > span_max)
        {
            span_max = span;
        }

        if (in_repeat)
        {
            if (span < span_leave)
            {
                ctx->rp_anno[a] += 1 * sizeof(track_data);

                ctx->rp_data[ctx->rp_dcur++] = -(ctx->rp_events[i]);

                ctx->stats_repeat_bases += ctx->rp_data[ ctx->rp_dcur - 1 ] - ctx->rp_data[ ctx->rp_dcur - 2 ];

#ifdef DEBUG_LOCAL
                printf("%7d L %3d %5d\n", a, span, -(ctx->rp_events[i]));
#endif

                in_repeat = 0;
            }
        }
        else
        {
            if (span > span_enter)
            {
                if (ctx->rp_dcur + 3 >= ctx->rp_dmax)
                {
                    ctx->rp_dmax = 1.2 * ctx->rp_dmax + 20;
                    ctx->rp_data = (int*)realloc(ctx->rp_data, sizeof(int)*ctx->rp_dmax);
                }

                ctx->rp_anno[a] += 1 * sizeof(track_data);
                ctx->rp_data[ctx->rp_dcur++] = ctx->rp_events[i];

#ifdef DEBUG_LOCAL
                printf("%7d E %3d %5d\n", a, span, ctx->rp_events[i]);
#endif

                in_repeat = 1;
                span_max = 0;
            }
        }
    }


    if (ctx->rp_dcur - rp_dcur > 2)
    {
        i = rp_dcur;
        j = i + 2;
        while ( j < ctx->rp_dcur )
        {
            int e = ctx->rp_data[ i + 1 ];
            int b_next = ctx->rp_data[ j ];

            if ( b_next - e < ctx->merge_distance )
            {
                ctx->stats_merged++;
                ctx->rp_data[ i + 1 ] = ctx->rp_data[ j + 1 ];
            }
            else
            {
                i += 2;
            }

            j += 2;
        }

        ctx->rp_dcur = i + 2;
        ctx->rp_anno[a] = (ctx->rp_dcur - rp_dcur) * sizeof(track_data);
    }

    return 1;
}

static void usage()
{
    printf("usage:   [-h <float>] [-l <float>] [-t <string>] [-b <int>] <db> <overlaps>\n");
    printf("options: -h ... enter count (%d)\n", DEF_ARG_H);
    printf("         -l ... leave count (%d)\n", DEF_ARG_L);

    printf("         -t ... track name (%s)\n", DEF_ARG_T);
    printf("         -b ... track block\n");
}

int main(int argc, char* argv[])
{
    HITS_DB db;
    PassContext* pctx;
    RepeatContext rctx;
    FILE* fileOvlIn;

    bzero(&rctx, sizeof(RepeatContext));
    rctx.db = &db;

    // process arguments

    rctx.cnt_enter = DEF_ARG_H;
    rctx.cnt_leave = DEF_ARG_L;
    rctx.merge_distance = 0;

    rctx.rp_track = DEF_ARG_T;
    rctx.rp_block = 0;

    int c;

    opterr = 0;

    while ((c = getopt(argc, argv, "h:l:t:b:")) != -1)
    {
        switch (c)
        {
            case 'b':
                      rctx.rp_block = atoi(optarg);
                      break;

            case 'h':
                      rctx.cnt_enter = atof(optarg);
                      break;

            case 'l':
                      rctx.cnt_leave = atof(optarg);
                      break;

            case 't':
                      rctx.rp_track = optarg;
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

    char* pcPathReadsIn = argv[optind++];
    char* pcPathOverlaps = argv[optind++];

    if (rctx.cnt_enter < rctx.cnt_leave)
    {
        fprintf(stderr, "invalid arguments: low %d > high %d\n", rctx.cnt_leave, rctx.cnt_enter);
        exit(1);
    }

    if ( (fileOvlIn = fopen(pcPathOverlaps, "r")) == NULL )
    {
        fprintf(stderr, "could not open '%s'\n", pcPathOverlaps);
        exit(1);
    }


    // init

    pctx = pass_init(fileOvlIn, NULL);

    pctx->split_b = 0;
    pctx->load_trace = 0;
    pctx->data = &rctx;

    Open_DB(pcPathReadsIn, &db);

    // passes

    pre_repeats(&rctx);

    pass(pctx, handler_repeats);

    post_repeats(&rctx);

    // cleanup

    pass_free(pctx);

    fclose(fileOvlIn);

    Close_DB(&db);

    return 0;
}
