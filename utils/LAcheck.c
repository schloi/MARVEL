/*******************************************************************************************
 *
 * performs basic sanity checks on the overlaps
 *
 * Author: MARVEL Team
 *
 * Date  : December 2014
 *
 *******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "db/DB.h"
#include "dalign/align.h"
#include "lib/oflags.h"
#include "lib/pass.h"

// switches

#undef PROGRESS            // show verification progress

// command line defaults

#define DEF_ARG_P       0
#define DEF_ARG_S       0
#define DEF_ARG_D       0

// macros

#define CMP(a, b) cmp = (a) - (b); if (cmp != 0) return cmp;

// structs

typedef struct
{
    HITS_DB* db;
    ovl_header_twidth twidth;

    int error;          // file didn't pass check

    int check_ptp;      // pass through points
    int check_sort;     // sort order
    int check_dupes;    // duplicates

    int ignoreDiscardedOvls;

    ovl_header_novl novl;         // Overlaps counted

    int prev_a;

} CheckContext;

static void usage()
{
    fprintf(stderr, "usage  : [-p] [-s] [-d] [-i] <db> <las>\n");
    fprintf(stderr, "options: -p ... check pass-through points (%d)\n", DEF_ARG_P);
    fprintf(stderr, "         -s ... check sort order (%d)\n", DEF_ARG_S);
    fprintf(stderr, "         -d ... report duplicates. implies -s (%d)\n", DEF_ARG_D);
    fprintf(stderr, "         -i ... ignore overlaps that are discarded (default: 0)\n");
}

inline static int compare_sort(Overlap* o1, Overlap* o2)
{
    int cmp;

    CMP(o1->aread,      o2->aread);
    CMP(o1->bread,      o2->bread);
    CMP(o1->flags & OVL_COMP, o2->flags & OVL_COMP);
    CMP(o1->path.abpos, o2->path.abpos);

    return cmp;
}

// used jointly with compare_sort

inline static int compare_duplicate(Overlap* o1, Overlap* o2)
{
    int cmp;

    CMP(o1->flags,      o2->flags);

    CMP(o1->path.aepos, o2->path.aepos);
    CMP(o1->path.bbpos, o2->path.bbpos);
    CMP(o1->path.bepos, o2->path.bepos);
    CMP(o1->path.tlen,  o2->path.tlen);

    return cmp;
}

static void check_pre(PassContext* pctx, CheckContext* cctx)
{
    cctx->twidth = pctx->twidth;
    cctx->prev_a = 0;
}

static void check_post(PassContext* pctx, CheckContext* cctx)
{
    if (!cctx->error && pctx->novl != cctx->novl)
    {
        fprintf(stderr, "novl of %lld doesn't match actual overlap count of %lld\n", pctx->novl, cctx->novl);
        cctx->error = 1;
    }
}

static int check_process(void* _ctx, Overlap* ovl, int novl)
{
    CheckContext* ctx = (CheckContext*)_ctx;

    int i, lena, lenb;

    for (i = 0; i < novl; i++)
    {
        ctx->novl++;

        if(ctx->ignoreDiscardedOvls && (ovl[i].flags & OVL_DISCARD))
          continue;

        if (i == 0)
        {
            if (ctx->check_sort && ctx->prev_a > ovl[i].aread)
            {
                fprintf(stderr, "overlap %lld: not sorted\n", ctx->novl);
                ctx->error = 1;
            }
        }
        else
        {
            int cmp  = compare_sort(ovl + (i - 1), ovl + i);

            if (cmp > 0 && ctx->check_sort)
            {
                printf("%d %d\n", ovl[i-1].aread, ovl[i-1].bread);

                fprintf(stderr, "overlap %lld: not sorted\n", ctx->novl);
                ctx->error = 1;
            }
            else if (cmp == 0 && ctx->check_dupes && compare_duplicate(ovl + (i - 1), ovl + i) == 0)
            {
                printf("%d %d\n", ovl[i].aread, ovl[i].bread);

                fprintf(stderr, "overlap %lld: equal to previous overlap\n", ctx->novl);
                ctx->error = 1;
            }
        }

        lena = DB_READ_LEN(ctx->db, ovl[i].aread);
        lenb = DB_READ_LEN(ctx->db, ovl[i].bread);

        if (ovl[i].path.abpos < 0)
        {
            fprintf(stderr, "overlap %lld: abpos < 0\n", ctx->novl);
            ctx->error = 1;
        }

        if (ovl[i].path.bbpos < 0)
        {
            fprintf(stderr, "overlap %lld: bbpos < 0\n", ctx->novl);
            ctx->error = 1;
        }

        if (ovl[i].path.aepos > lena)
        {
            fprintf(stderr, "overlap %lld: aepos > lena\n", ctx->novl);
            ctx->error = 1;
        }

        if (ovl[i].path.bepos > lenb)
        {
            fprintf(stderr, "overlap %lld: bepos > lenb\n", ctx->novl);
            ctx->error = 1;
        }

        if (ovl[i].path.tlen < 0)
        {
            fprintf(stderr, "overlap %lld: invalid tlen %d\n", ctx->novl, ovl[i].path.tlen);
            ctx->error = 1;
        }

        if (ctx->check_ptp)
        {
            ovl_trace* trace = ovl[i].path.trace;

            int apos = ovl[i].path.abpos;
            int bpos = ovl[i].path.bbpos;

            int j;
            for (j = 0; j < ovl[i].path.tlen; j += 2)
            {
                apos += (apos/ctx->twidth + 1) * ctx->twidth;
                bpos += trace[j+1];
            }

            if (bpos != ovl[i].path.bepos)
            {
                fprintf(stderr, "overlap %lld (%d x %d): pass-through points inconsistent be = %d (expected %d)\n",
                            ctx->novl, ovl[i].aread, ovl[i].bread, bpos, ovl[i].path.bepos);
                ctx->error = 1;
            }
        }
    }

    ctx->prev_a = ovl->aread;

    return !ctx->error;
}

int main(int argc, char* argv[])
{
    PassContext* pctx;
    CheckContext cctx;
    HITS_DB db;
    FILE* fileOvlIn;

    bzero(&cctx, sizeof(CheckContext));
    cctx.db = &db;


    // process arguments

    cctx.check_ptp = DEF_ARG_P;
    cctx.check_sort = DEF_ARG_S;
    cctx.check_dupes = DEF_ARG_D;
    cctx.ignoreDiscardedOvls = 0;

    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "psdi")) != -1)
    {
        switch (c)
        {
            case 'p':
                      cctx.check_ptp = 1;
                      break;

            case 'd':
                      cctx.check_dupes = 1;
                      cctx.check_sort = 1;
                      break;

            case 's':
                      cctx.check_sort = 1;
                      break;

            case 'i':
                      cctx.ignoreDiscardedOvls = 1;
                      break;

            default:
                      usage();
                      exit(1);
        }
    }

    if (opterr || argc - optind != 2)
    {
        usage();
        exit(1);
    }

    char* pcPathReadsIn = argv[optind++];
    char* pcPathOverlapsIn = argv[optind++];

    if ( (fileOvlIn = fopen(pcPathOverlapsIn, "r")) == NULL )
    {
        fprintf(stderr, "could not open '%s'\n", pcPathOverlapsIn);
        exit(1);
    }

    if ( Open_DB(pcPathReadsIn, &db) )
    {
        fprintf(stderr, "could not open database '%s'\n", pcPathReadsIn);
        exit(1);
    }

    pctx = pass_init(fileOvlIn, NULL);
    pctx->split_b = 0;
    pctx->load_trace = cctx.check_ptp;
    pctx->unpack_trace = cctx.check_ptp;
    pctx->data = &cctx;

    check_pre(pctx, &cctx);

    pass(pctx, check_process);

    check_post(pctx, &cctx);

    pass_free(pctx);

    Close_DB(&db);

    fclose(fileOvlIn);

    return cctx.error;
}
