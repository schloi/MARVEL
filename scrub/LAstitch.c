/*******************************************************************************************
 *
 *  A->B overlaps can be split into multiple A->B records due to bad regions in either one
 *  of the reads, causing the overlapper to stop aligning. Here we look for those records
 *  (with a maximum gap of -f), join the two overlaps into a single record, discard the
 *  superfluous one, and (re-)compute pass-through points & diffs for the segments surrounding
 *  the break point.
 *
 *  Author  :  MARVEL Team
 *
 *  Date    :  November 2014
 *
 *******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/param.h>

#include "lib/read_loader.h"
#include "lib/pass.h"
#include "lib/oflags.h"
#include "lib/colors.h"
#include "lib/utils.h"

#include "db/DB.h"
#include "dalign/align.h"

// command line defaults

#define DEF_ARG_F   40
#define DEF_ARG_P    0

// switches

#define VERBOSE
#undef DEBUG_STITCH

// macros

#define OVL_STRAND(ovl) ( ( (ovl)->flags & OVL_COMP) ? 'c' : 'n' )

// context

typedef struct
{
    HITS_DB* db;                    // database

    int64 stitched;                 // how many stitch operations have occurred
    int fuzz;                       // max gap between the A/B overlaps
    int verbose;
    int useRLoader;

    ovl_header_twidth twidth;       // trace point spacing
    size_t tbytes;                  // bytes used for trace point storage

    ovl_trace* trace;               // new trace buffer for all B reads in the stitch_handler
    int tcur;                       // current position in the buffer
    int tmax;                       // size of the buffer

    Work_Data* align_work;          // working storage for the alignment module
    Alignment align;                // alignment and path record for computing the
    Path path;                      // alignment in the gap region

    Read_Loader* rl;

} StitchContext;

// externals for getopt()

extern char *optarg;
extern int optind, opterr, optopt;

static void create_pass_through_points(StitchContext* ctx, ovl_trace* trace)
{
    int twidth = ctx->twidth;
    int a = ctx->align.path->abpos;
    int b = ctx->align.path->bbpos;
    int p, t;
    int diffs = 0;
    int matches = 0;

    int aprev = a;
    int bprev = b;

    int tcur = 0;

    for (t = 0; t < ctx->align.path->tlen; t++)
    {
        if ((p = ((int*)(ctx->align.path->trace))[t]) < 0)
        {
            p = -p - 1;
            while (a < p)
            {
                if (ctx->align.aseq[a] != ctx->align.bseq[b]) diffs++;
                else matches++;

                a += 1;
                b += 1;

                if (a % twidth == 0)
                {
                    trace[tcur++] = diffs;
                    trace[tcur++] = b - bprev;

                    aprev = a;
                    bprev = b;

                    // printf(" a(%4dx%4d %3d %3d %3d)", a, b, diffs, matches, tcur);
                    diffs = matches = 0;
                }
            }

            diffs++;
            b += 1;
        }
        else
        {
            p--;

            while (b < p)
            {
                if (ctx->align.aseq[a] != ctx->align.bseq[b]) diffs++;
                else matches++;

                a += 1;
                b += 1;

                if (a % twidth == 0)
                {
                    trace[tcur++] = diffs;
                    trace[tcur++] = b - bprev;

                    aprev = a;
                    bprev = b;

                    // printf(" b(%4dx%4d %3d %3d %3d)", a, b, diffs, matches, tcur);
                    diffs = matches = 0;
                }
            }

            diffs++;
            a += 1;

            if (a % twidth == 0)
            {
                trace[tcur++] = diffs;
                trace[tcur++] = b - bprev;

                aprev = a;
                bprev = b;

                // printf(" c(%4dx%4d %3d %3d %3d)", a, b, diffs, matches, tcur);
                diffs = matches = 0;
            }
        }
    }

    p = ctx->align.path->aepos;
    while (a < p)
    {
        if (ctx->align.aseq[a] != ctx->align.bseq[b]) diffs++;
        else matches++;

        a += 1;
        b += 1;

        if (a % twidth == 0 && a != ctx->align.path->aepos)
        {
            trace[tcur++] = diffs;
            trace[tcur++] = b - bprev;

            aprev = a;
            bprev = b;

            // printf(" d(%4dx%4d %3d %3d %3d)", a, b, diffs, matches, tcur);
            diffs = matches = 0;
        }
    }


    if (a != aprev)
    {
        trace[tcur++] = diffs;
        trace[tcur++] = b - bprev;

        // printf(" e(%4dx%4d %3d %3d %3d)", a, b, diffs, matches, tcur);
    }
    else
    {
        trace[tcur-1] += b - bprev;
    }

    // printf("\n");
}

/*
    ensures that there is enough space in StitchContext.trace for
    an additional needed values.
*/
static void ensure_trace(StitchContext* sctx, int needed)
{
    // space needed

    if (needed + sctx->tcur >= sctx->tmax)
    {
        // void* traceb_old = sctx->trace;
        // void* tracee_old = sctx->trace + sctx->tmax;

        int tmax = (needed + sctx->tcur) * 2 + 100;
        void* trace = realloc( sctx->trace, sizeof(ovl_trace) * tmax );

        /*
        // re-adjust trace pointers from the overlap records
        int i;
        for (i = 0; i < novl; i++)
        {
            if (ovl[i].path.trace >= traceb_old && ovl[i].path.trace <= tracee_old)
            {
                printf("adjust %d\n", i);

                ovl[i].path.trace = trace + (ovl[i].path.trace - traceb_old);
            }
        }
        */

        sctx->tmax = tmax;
        sctx->trace = trace;
    }
}

/*
    initialises the StitchContext
*/
static void stitch_pre(PassContext* pctx, StitchContext* sctx)
{
#ifdef VERBOSE
    printf(ANSI_COLOR_GREEN "PASS stitching" ANSI_COLOR_RESET "\n");
#endif

    sctx->tbytes = pctx->tbytes;
    sctx->twidth = pctx->twidth;
    sctx->align_work = New_Work_Data();

    sctx->tmax = 2 * ( ( DB_READ_MAXLEN( sctx->db ) + pctx->twidth ) / pctx->twidth );
    sctx->trace = (ovl_trace*)malloc( sizeof(ovl_trace) * sctx->tmax );

    sctx->align.path = &(sctx->path);
    sctx->align.aseq = New_Read_Buffer(sctx->db);
    sctx->align.bseq = New_Read_Buffer(sctx->db);
}

/*
    cleanup the StitchContext
*/
static void stitch_post(PassContext* pctx, StitchContext* sctx)
{
#ifdef VERBOSE
    printf("stitched %lld out of %lld overlaps\n", sctx->stitched, pctx->novl);
#endif

    Free_Work_Data(sctx->align_work);
    free(sctx->align.aseq - 1);
    free(sctx->align.bseq - 1);

    free(sctx->trace);
}

/*
    called for each distinct A->B overlap pair
*/
static int stitch_handler(void* _ctx, Overlap* ovl, int novl)
{
    StitchContext* ctx = (StitchContext*)_ctx;

    Read_Loader* rl = ctx->rl;

    if (novl < 2)
    {
        return 1;
    }

    if ( ovl->aread == ovl->bread )
    {
        return 1;
    }

    int fuzz = ctx->fuzz;

    int i, j, k;
    int ab2, ae1, ae2;
    int bb2, be1, be2;
    int ab1, bb1;

    ctx->tcur = 0;

    int tsum = 0;
    for (i = 0; i < novl; i++)
    {
        tsum += ovl[i].path.tlen;
    }

    ensure_trace(ctx, tsum * 4);

    for (i = 0; i < novl; i++)
    {
        if (ovl[i].flags & OVL_DISCARD) continue;

        // b = ovl[i].bread;

        ab1 = ovl[i].path.abpos;
        ae1 = ovl[i].path.aepos;

        bb1 = ovl[i].path.bbpos;
        be1 = ovl[i].path.bepos;

        for (k = i+1; k < novl; k++)
        {
            if ((ovl[k].flags & OVL_DISCARD) ||
                (ovl[i].flags & OVL_COMP) != (ovl[k].flags & OVL_COMP))
            {
                continue;
            }

            ab2 = ovl[k].path.abpos;
            ae2 = ovl[k].path.aepos;

            bb2 = ovl[k].path.bbpos;
            be2 = ovl[k].path.bepos;

            if ( abs(ae1 - ab2) < fuzz && abs(be1 - bb2) < fuzz
                 && abs( (ae1 - ab2) - (be1 - bb2) ) < fuzz )          // added 2016-02-17
            {
                int segb = (MIN(ae1, ab2) - 50) / ctx->twidth;
                int sege = (MAX(ae1, ab2) + 50) / ctx->twidth;
                ovl_trace* tracei = ovl[i].path.trace;
                ovl_trace* tracek = ovl[k].path.trace;
                int align_bb, align_be, seg, bpos;

                assert(segb <= sege);

                int tcur = ctx->tcur;
                int tcur_start = tcur;

                if (ctx->verbose)
                {
                    printf("STITCH %8d %2d @ %5d..%5d -> %8d @ %5d..%5d %c\n",
                            ovl[i].aread,
                            i,
                            ab1, ae1, ovl[i].bread, bb1, be1, OVL_STRAND(ovl+i));

                    printf("                %2d @ %5d..%5d -> %8d @ %5d..%5d %c\n",
                            k,
                            ab2, ae2, ovl[k].bread, bb2, be2, OVL_STRAND(ovl+k));
                }

#ifdef DEBUG_STITCH
                char* color1 = (ae1 > ab2) ? ANSI_COLOR_RED : "";
                char* color2 = (be1 > bb2) ? ANSI_COLOR_RED : "";

                printf("STITCH %8d %2d @ %5d..%s%5d" ANSI_COLOR_RESET " -> %8d @ %5d..%s%5d" ANSI_COLOR_RESET " %c",
                        ovl[i].aread, i,
                        ab1, color1, ae1, ovl[i].bread, bb1, color2, be1, OVL_STRAND(ovl+i));

                int apos = ovl[i].path.abpos;
                bpos = ovl[i].path.bbpos;
                for (j = 0; j < ovl[i].path.tlen; j += 2)
                {
                    if (j == ovl[i].path.tlen - 2)
                    {
                        apos = ovl[i].path.aepos;
                    }
                    else
                    {
                        apos = (apos / ctx->twidth + 1) * ctx->twidth;
                    }

                    bpos += tracei[j+1];

                    if (j >= ovl[i].path.tlen - 6) printf(" (%3d, %3d, %5d, %5d)", tracei[j+1], tracei[j], apos, bpos);
                }
                printf("\n");

                printf("                %2d @ %s%5d" ANSI_COLOR_RESET "..%5d -> %8d @ %s%5d" ANSI_COLOR_RESET "..%5d %c",
                        k,
                        color1, ab2, ae2, ovl[k].bread, color2, bb2, be2, OVL_STRAND(ovl+k));

                apos = ovl[k].path.abpos;
                bpos = ovl[k].path.bbpos;
                for (j = 0; j < ovl[k].path.tlen; j += 2)
                {
                    if (j == ovl[i].path.tlen - 2)
                    {
                        apos = ovl[i].path.aepos;
                    }
                    else
                    {
                        apos = (apos / ctx->twidth + 1) * ctx->twidth;
                    }
                    bpos += tracek[j+1];

                    if (j < 6) printf(" (%3d, %3d, %5d, %5d)", tracek[j+1], tracek[j], apos, bpos);
                }
                printf("\n");

#endif

                bpos = ovl[i].path.bbpos;
                for (seg = ovl[i].path.abpos / ctx->twidth, j = 0;
                     j < ovl[i].path.tlen && seg < segb;
                     j+=2, seg++)
                {
                    ctx->trace[ tcur++ ] = tracei[j];
                    ctx->trace[ tcur++ ] = tracei[j+1];

                    bpos += tracei[j+1];
                }

                align_bb = bpos;
                align_be = 0;

                for (j = segb ; j <= sege ; j++)
                {
                    ctx->trace[ tcur++ ] = 0;
                    ctx->trace[ tcur++ ] = 0;
                }

                bpos = ovl[k].path.bbpos;

                for (seg = ovl[k].path.abpos / ctx->twidth, j = 0;
                     j < ovl[k].path.tlen;
                     j+=2, seg++)
                {
                    if (seg == sege)
                    {
                        align_be = bpos + tracek[j+1];
                    }
                    else if (seg > sege)
                    {
                        ctx->trace[ tcur++ ] = tracek[j];
                        ctx->trace[ tcur++ ] = tracek[j+1];
                    }

                    bpos += tracek[j+1];
                }

                if ( align_bb >= align_be )
                {
                    continue;
                }

#ifdef DEBUG_STITCH
                printf("%3d..%3d %5d..%5d %5d..%5d\n",
                        segb, sege,
                        segb * ctx->twidth, (sege + 1) * ctx->twidth,
                        align_bb, align_be);
#endif

                if(ctx->useRLoader)
                {
                		rl_load_read(rl, ovl[i].aread, ctx->align.aseq, 0);
                		rl_load_read(rl, ovl[i].bread, ctx->align.bseq, 0);
                }
                else
                {
                		Load_Read(ctx->db, ovl[i].aread, ctx->align.aseq, 0);
                		Load_Read(ctx->db, ovl[i].bread, ctx->align.bseq, 0);
                }

                if ((ovl[i].flags & OVL_COMP))
                {
                    Complement_Seq(ctx->align.bseq, DB_READ_LEN(ctx->db, ovl[i].bread));
                }

                ctx->align.alen = DB_READ_LEN(ctx->db, ovl[i].aread);
                ctx->align.blen = DB_READ_LEN(ctx->db, ovl[i].bread);

                ctx->align.path->abpos = segb * ctx->twidth;
                ctx->align.path->aepos = (sege + 1) * ctx->twidth;
                ctx->align.path->bbpos = align_bb;
                ctx->align.path->bepos = align_be;

                ctx->align.path->diffs = (ctx->align.path->aepos - ctx->align.path->abpos) + (ctx->align.path->bepos - ctx->align.path->bbpos);

                Compute_Trace_ALL(&(ctx->align), ctx->align_work);

                create_pass_through_points(ctx, ctx->trace + tcur_start + 2 * (segb - ovl[i].path.abpos / ctx->twidth));

#ifdef DEBUG_STITCH
                printf("TRACE");
                for (j = 0; j < ctx->align.path->tlen; j++)
                {
                    printf(" %d", ((int*)(ctx->align.path->trace))[j]);
                }
                printf("\n");

                apos = ovl[i].path.abpos;
                bpos = ovl[i].path.bbpos;
                for (j = tcur_start; j < tcur; j+=2)
                {
                    apos = (apos / ctx->twidth) * ctx->twidth + ctx->twidth;
                    bpos += ctx->trace[j+1];

                    if ( j > 0 && j % 10 == 0 ) printf("\n");
                    printf("(%3d, %3d, %5d, %5d) ", ctx->trace[j+1], ctx->trace[j], apos, bpos);
                }
                printf("\n\n");

                assert( bpos == ovl[k].path.bepos );

#endif

                // trace points can overflow if we stitch across a big gap
                int bOverflow = 0;
                if (ctx->tbytes == sizeof(uint8))
                {
                    for (j = 0; j < tcur; j += 2)
                    {
                        if (ctx->trace[j + 1] > 255)
                        {
                            bOverflow = 1;
                            break;
                        }
                    }
                }

                if (!bOverflow)
                {
                    ctx->tcur = tcur;

                    ovl[i].path.aepos = ae1 = ae2;
                    ovl[i].path.bepos = be1 = be2;
                    ovl[i].path.diffs += ovl[k].path.diffs;

                    ovl[i].path.trace = ctx->trace + tcur_start;
                    ovl[i].path.tlen = ctx->tcur - tcur_start;

                    ovl[k].flags |= OVL_DISCARD | OVL_STITCH;
                    ctx->stitched++;
                }
            }

        }
    }

    return 1;
}

static int loader_handler(void* _ctx, Overlap* ovl, int novl)
{
    StitchContext* ctx = (StitchContext*)_ctx;
    Read_Loader* rl = ctx->rl;

    if (novl < 2)
    {
        return 1;
    }

    int fuzz = ctx->fuzz;

    int i, k; // , b;
    int ab2, ae1; // , ae2;
    int bb2, be1; // , be2;
    // int ab1, bb1;

    for (i = 0; i < novl; i++)
    {
        if (ovl[i].flags & OVL_DISCARD) continue;

        // b = ovl[i].bread;

        // ab1 = ovl[i].path.abpos;
        ae1 = ovl[i].path.aepos;

        // bb1 = ovl[i].path.bbpos;
        be1 = ovl[i].path.bepos;

        for (k = i+1; k < novl; k++)
        {
            if ((ovl[k].flags & OVL_DISCARD) ||
                (ovl[i].flags & OVL_COMP) != (ovl[k].flags & OVL_COMP))
            {
                continue;
            }

            ab2 = ovl[k].path.abpos;
            // ae2 = ovl[k].path.aepos;

            bb2 = ovl[k].path.bbpos;
            // be2 = ovl[k].path.bepos;

            if ( abs(ae1 - ab2) < fuzz && abs(be1 - bb2) < fuzz )
            {
                rl_add(rl, ovl[i].aread);
                rl_add(rl, ovl[i].bread);
            }
        }
    }

    return 1;
}

static void usage()
{
    fprintf(stderr, "usage: [-pvL] [-f <int>] <db> <ovl.in> <ovl.out>\n");
    fprintf(stderr, "options: -v ... verbose\n");
    fprintf(stderr, "         -f ... fuzzing for stitch (%d)\n", DEF_ARG_F);
    fprintf(stderr, "         -L ... two pass processing with read caching\n");
    fprintf(stderr, "         -p ... purge discarded overlaps\n");
}

int main(int argc, char* argv[])
{
    HITS_DB db;
    StitchContext sctx;
    PassContext* pctx;
    FILE* fileOvlIn;
    FILE* fileOvlOut;

    bzero(&sctx, sizeof(StitchContext));
    sctx.fuzz = DEF_ARG_F;
    sctx.verbose = 0;
    sctx.db = &db;
    sctx.useRLoader = 0;

    // process arguments

    int arg_purge = DEF_ARG_P;

    opterr = 0;

    int c;
    while ((c = getopt(argc, argv, "Lpvf:")) != -1)
    {
        switch (c)
        {
            case 'p':
                      arg_purge = 1;
                      break;

            case 'L':
                      sctx.useRLoader = 1;
                      break;

            case 'v':
                      sctx.verbose = 1;
                      break;

            case 'f':
                      sctx.fuzz = atoi(optarg);
                      break;

            default:
                      usage();
                      exit(1);
        }
    }

    if (argc - optind < 3)
    {
        usage();
        exit(1);
    }

    char* pcPathReadsIn = argv[optind++];
    char* pcPathOverlapsIn = argv[optind++];
    char* pcPathOverlapsOut = argv[optind++];

    if ( (fileOvlIn = fopen(pcPathOverlapsIn, "r")) == NULL )
    {
        fprintf(stderr, "could not open '%s'\n", pcPathOverlapsIn);
        exit(1);
    }

    if ( (fileOvlOut = fopen(pcPathOverlapsOut, "w")) == NULL )
    {
        fprintf(stderr, "could not open '%s'\n", pcPathOverlapsOut);
        exit(1);
    }

    if ( Open_DB(pcPathReadsIn, &db) )
    {
        fprintf(stderr, "could not open database '%s'\n", pcPathReadsIn);
        exit(1);
    }

    if ( sctx.fuzz < 0 )
    {
        fprintf(stderr, "invalid fuzzing value of %d\n", sctx.fuzz);
        exit(1);
    }


    if(sctx.useRLoader)
    {
		// collect read ids for loading

    	sctx.rl = rl_init(&db, 1);

		pctx = pass_init(fileOvlIn, NULL);

		pctx->data = &sctx;
		pctx->split_b = 1;
		pctx->load_trace = 0;

		pass(pctx, loader_handler);

		rl_load_added(sctx.rl);

		pass_free(pctx);
    }

    // process overlaps

    pctx = pass_init(fileOvlIn, fileOvlOut);

    pctx->data = &sctx;

    pctx->split_b = 1;
    pctx->load_trace = 1;
    pctx->unpack_trace = 1;
    pctx->write_overlaps = 1;
    pctx->purge_discarded = arg_purge;

    stitch_pre(pctx, &sctx);

    pass(pctx, stitch_handler);

    stitch_post(pctx, &sctx);

    // cleanup

    if(sctx.useRLoader)
    {
    	rl_free(sctx.rl);
    }

    Close_DB(&db);

    pass_free(pctx);

    fclose(fileOvlIn);
    fclose(fileOvlOut);

    return 0;
}

