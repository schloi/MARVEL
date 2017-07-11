/*******************************************************************************************
 *
 *  Reassign repeat intervals based on overlaps and an existing repeat annotation
 *
 *  Date   :  January 2016
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
#include "lib/borders.h"
#include "lib/pass.h"
#include "lib/utils.h"

#include "lib.ext/types.h"
#include "lib.ext/bitarr.h"

#include "db/DB.h"
#include "dalign/align.h"


// defaults

#define DEF_ARG_B   0
#define DEF_ARG_I   TRACK_REPEATS
#define DEF_ARG_II  TRACK_HREPEATS
#define DEF_ARG_T   TRACK_TRIM
#define DEF_ARG_E   -1
#define DEF_ARG_R   1


// constants

#define MAX_BUF_SIZE ( 10L * 1024 * 1024 * 1024 )
#define MAX_BUFS     ( 16 )

#define MIN_INT_LEN  ( 500 )        // min intervals length

// toggles

#define VERBOSE
#undef DEBUG_HOMOGENIZE

typedef struct
{
    HITS_DB* db;
    ovl_header_twidth twidth;

    HITS_TRACK* track_repeats;
    HITS_TRACK* track_trim;

    char* track_in;
    char* track_out;
    char* track_trim_name;

    uint64_t stats_bases_tagged;

    int block;
    int copy_existing;
    int ends;
    int res;

    bit** read_masks;   	       // repeat annotation bit masks for each read
    bit** bufs;                    // buffers for repeat annotation

} HomogenizeContext;

// getopt()

extern char* optarg;
extern int optind, opterr, optopt;



static void pre_homogenize(PassContext* pctx, HomogenizeContext* ctx)
{
#ifdef VERBOSE
    printf(ANSI_COLOR_GREEN "PASS homogenising\n" ANSI_COLOR_RESET);
#endif

    int nreads = DB_NREADS(ctx->db);
    ctx->track_repeats = track_load(ctx->db, ctx->track_in);

    if (!ctx->track_repeats)
    {
        fprintf(stderr, "could not load track %s\n", ctx->track_in);
        exit(1);
    }

    ctx->twidth = pctx->twidth;

    // allocate bit mask buffer pointers

    ctx->read_masks = malloc(sizeof(bit*) * nreads);

    assert(ctx->read_masks != NULL);

    // allocate buffers

    int i;
    uint64_t nbuf = 0;
    uint64_t total = 0;

    bit** bufs = ctx->bufs = calloc(MAX_BUFS, sizeof(bit*));
    int curbuf = 0;

    for (i = 0; i < nreads; i++)
    {
        int len = DB_READ_LEN(ctx->db, i);

        if (ctx->res > 1)
        {
            len = ( len + ctx->res - 1) / ctx->res;
        }

        nbuf += ba_bufsize(len + 1);

        if (nbuf > MAX_BUF_SIZE)
        {
            bufs[curbuf] = calloc(nbuf, sizeof(bit));

            assert( bufs[curbuf] != NULL );

            total += nbuf;

            curbuf++;
            nbuf = 0;

            assert(curbuf < MAX_BUFS);
        }

        // ctx->read_masks[i] = ba_new( len + 1 );
    }

    if (nbuf > 0)
    {
        total += nbuf;

        bufs[curbuf] = calloc(nbuf, sizeof(bit));

        assert( bufs[curbuf] != NULL );

        curbuf++;
    }

    // set bit mask pointers to buffers

    nbuf = 0;
    curbuf = 0;

    for (i = 0; i < nreads; i++)
    {
        int len = DB_READ_LEN(ctx->db, i);

        if (ctx->res > 1)
        {
            len = ( len + ctx->res - 1) / ctx->res;
        }

        ctx->read_masks[i] = bufs[curbuf] + nbuf;

        nbuf += ba_bufsize(len + 1);

        if (nbuf > MAX_BUF_SIZE)
        {
            nbuf = 0;
            curbuf++;
        }
    }

#ifdef VERBOSE
    char* buf = format_bytes(total);

    printf("allocated %s in %d buffers\n", buf, curbuf + 1);

    free(buf);
#endif

    // initialize with existing annotation

    if (ctx->copy_existing)
    {
#ifdef VERBOSE
        printf("copying existing annotation\n");
#endif

        track_anno* tanno = ctx->track_repeats->anno;
        track_data* tdata = ctx->track_repeats->data;

        int a;
        for (a = 0; a < nreads; a++)
        {
            track_anno ob = tanno[a] / sizeof(track_data);
            track_anno oe = tanno[a + 1] / sizeof(track_data);

            bit* amask = ctx->read_masks[a];

            while (ob < oe)
            {
                int beg = tdata[ob];
                int end = tdata[ob + 1];
                ob += 2;

                if (end - beg < MIN_INT_LEN)
                {
                    continue;
                }

                beg = ( beg + ctx->res - 1) / ctx->res;
                end = end / ctx->res;

                ba_assign_range(amask, beg, end, 1);
            }
        }
    }
}

static void post_homogenize(HomogenizeContext* ctx)
{
    track_anno* anno = malloc(sizeof(track_anno) * (ctx->db->ureads + 1));
    uint64_t dmax = 1000;
    uint64_t dcur = 0;
    track_data* data = malloc(sizeof(track_data) * dmax);

    assert( anno != NULL && data != NULL );

    bzero(anno, sizeof(track_anno) * (ctx->db->ureads + 1));

    int i;

    for (i = 0; i < ctx->db->nreads; i++)
    {
        bit* bitarr = ctx->read_masks[i];
        int alen = DB_READ_LEN(ctx->db, i);

        if (ctx->res > 1)
        {
            alen = ( alen + ctx->res - 1) / ctx->res;
        }

        if (ba_count(bitarr, alen) < 1)
        {
            continue;
        }

        int j = 0;
        int beg = -1;
        int end = -1;

        for (j = 0; j < alen; j++)
        {
            int val = ba_value(bitarr, j);

            if (val && beg == -1)
            {
                beg = j;
            }
            else if (!val && beg != -1)
            {
                end = j - 1;

                if (dcur + 2 >= dmax)
                {
                    dmax = 1.2 * dmax + 1000;
                    data = realloc(data, sizeof(track_data) * dmax);
                }

                beg = beg * ctx->res;
                end = MIN(end * ctx->res, DB_READ_LEN(ctx->db, i));

                data[dcur++] = beg;
                data[dcur++] = end;

                anno[i] += sizeof(track_data) * 2;

                beg = -1;
            }
        }

        // interval extends to the end of the read

        if (beg != -1)
        {
            if (dcur + 2 >= dmax)
            {
                dmax = 1.2 * dmax + 1000;
                data = realloc(data, sizeof(track_data) * dmax);
            }

            beg = beg * ctx->res;

            data[dcur++] = beg;
            data[dcur++] = DB_READ_LEN(ctx->db, i) - 1;

            anno[i] += sizeof(track_data) * 2;
        }

    }

    track_anno coff, off;
    off = 0;

    int j;
    for (j = 0; j <= DB_NREADS(ctx->db); j++)
    {
        coff = anno[j];
        anno[j] = off;
        off += coff;
    }

    uint64_t k;
    uint64_t tagged = 0;
    for (k = 0; k < dcur; k += 2)
    {
        tagged += data[k+1] - data[k];
    }
    printf("tagged %" PRIu64 " as repeat\n", tagged);

    track_write(ctx->db, ctx->track_out, ctx->block, anno, data, dcur);

    for (j = 0; j < MAX_BUFS && ctx->bufs[j] != NULL; j++)
    {
        free(ctx->bufs[j]);
    }

    free(ctx->bufs);

    free(ctx->read_masks);

    free(anno);
    free(data);
}

static int handler_homogenize(void* _ctx, Overlap* ovl, int novl)
{
    HomogenizeContext* ctx = (HomogenizeContext*)_ctx;
    ovl_header_twidth twidth = ctx->twidth;
    int ends = ctx->ends;

    track_anno* tanno = ctx->track_repeats->anno;
    track_data* tdata = ctx->track_repeats->data;
    int a = ovl->aread;

    int i;
    for (i = 0; i < novl; i++)
    {
        if ( ovl[i].aread == ovl[i].bread )
        {
            continue;
        }

        // get repeat annotation for a read

        track_anno ob = tanno[a] / sizeof(track_data);
        track_anno oe = tanno[a + 1] / sizeof(track_data);
        int b = ovl[i].bread;

        int trim_bb, trim_be;
        int blen = DB_READ_LEN(ctx->db, b);

        if (ctx->track_trim)
        {
            get_trim(ctx->db, ctx->track_trim, b, &trim_bb, &trim_be);
        }
        else
        {
            trim_bb = 0;
            trim_be = blen;
        }

#ifdef DEBUG_HOMOGENIZE
        if (a == 592526)
        {
            printf("%d (%d) -> %d (%d)  %5d..%5d -> %5d..%5d\n", a, DB_READ_LEN(ctx->db, a), b, DB_READ_LEN(ctx->db, b),
                    ovl[i].path.abpos, ovl[i].path.aepos, ovl[i].path.bbpos, ovl[i].path.bepos);
            fflush(stdout);
        }
#endif

        int ab = ovl[i].path.abpos;
        int ae = ovl[i].path.aepos;

        while (ob < oe)
        {
            int beg = tdata[ob];
            int end = tdata[ob + 1];
            ob += 2;

            if (end - beg < MIN_INT_LEN)
            {
                continue;
            }

            if (beg > ae)
            {
                break ;
            }

            // does the repeat interval intersect with overlap

            int iab = MAX(beg, ab);
            int iae = MIN(end, ae);

            if (iab >= iae)
            {
                continue ;
            }

            // establish conservative estimate of the repeat extent
            // relative to the b read given the trace points

            ovl_trace* trace = ovl[i].path.trace;
            int tlen = ovl[i].path.tlen;

            int ibb = -1;
            int ibe = -1;

            int aoff = ovl[i].path.abpos;
            int boff = ovl[i].path.bbpos;

            int j;
            for (j = 0; j < tlen; j += 2)
            {
#ifdef DEBUG_HOMOGENIZE
                if (a == 592526 && b == 5971679)
                {
                    printf("%2d %2d %5d %5d %5d %5d\n", j, tlen, aoff, boff, ibb, ibe);
                }
#endif
                if ( (aoff >= iab || j == tlen - 2) && ibb == -1 )
                {
                    ibb = MIN(boff, ovl[i].path.bepos);
                }

                aoff = ( (aoff + twidth) / twidth) * twidth;

                if ( aoff >= iae && ibe == -1 )
                {
                    if (ibb == -1)
                    {
                        ibb = MIN(boff, ovl[i].path.bepos);
                    }

                    ibe = MIN(boff + trace[j + 1], ovl[i].path.bepos);

                    break;
                }

                boff += trace[j + 1];
            }

            if (ibb == -1 || ibe == -1)
            {
                continue;
            }

            if (ovl[i].flags & OVL_COMP)
            {
                int t = ibb;

                ibb = blen - ibe;
                ibe = blen - t;
            }

            assert(ibb <= ibe);

#ifdef DEBUG_HOMOGENIZE
            if (a == 592526)
            {
                printf("%6d %c %6d :: A %5d..%5d A_R %5d..%5d B %5d..%5d -> IA %5d..%5d IB %5d..%5d\n",
                            ovl[i].aread, ovl[i].flags & OVL_COMP ? 'c' : 'n', ovl[i].bread,
                            ab, ae,
                            beg, end,
                            ovl[i].path.bbpos, ovl[i].path.bepos,
                            iab, iae,
                            ibb, ibe);
                return 0;
            }
#endif

            // repeat tag the b read

            bit* bmask = ctx->read_masks[b];

            if (ends != -1)
            {
                // printf("%5d..%5d %5d..%5d ->", trim_bb, trim_be, ibb, ibe);

                if (ibb < ends + trim_bb)
                {
                    // printf(" %5d..%5d", ibb, MIN( ends + trim_bb, ibe ));
                    int bab = ( ibb + ctx->res - 1) / ctx->res;
                    int bae = MIN( ends + trim_bb, ibe ) / ctx->res;

                    ba_assign_range(bmask, bab, bae, 1);
                }

                if (ibe > trim_be - ends)
                {
                    // printf(" %5d..%5d", MAX( trim_be - ends, ibb ), ibe);
                    int bab = ( MAX( trim_be - ends, ibb ) + ctx->res - 1) / ctx->res;
                    int bae = ibe / ctx->res;

                    ba_assign_range(bmask, bab, bae, 1);
                }

                // printf("\n");
            }
            else
            {
                ibb = ( ibb + ctx->res - 1) / ctx->res;
                ibe = ibe / ctx->res;

                ba_assign_range(bmask, ibb, ibe, 1);
            }
        }
    }

    return 1;
}

static void usage( FILE* fout, const char* app )
{
    fprintf( fout, "usage: %s [-m] [-ber n] [-iIt track] database input.las\n\n", app );

    fprintf( fout, "Creates a new annotation track by transfering the annotation of the A read to the B read aligning to it.\n\n" );

    fprintf( fout, "options: -b n  track block\n" );
    fprintf( fout, "         -m  add input intervals to the output track\n" );
    fprintf( fout, "         -i track  input interval track (%s)\n", DEF_ARG_I );
    fprintf( fout, "         -I track  output interval track (%s)\n", DEF_ARG_II );
    fprintf( fout, "         -t track  trim annotation track to be used (%s)\n", DEF_ARG_T );
    fprintf( fout, "         -e n  only annotate n bases at the ends of the reads (%d), requires -t \n", DEF_ARG_E );
    fprintf( fout, "         -r n  base pair resolution (default %d)\n", DEF_ARG_R );
    fprintf( fout, "               scales memory usage by n and improves runtime\n" );
}

int main(int argc, char* argv[])
{
    HITS_DB db;
    PassContext* pctx;
    HomogenizeContext hctx;
    FILE* fileOvlIn;
    char* app = argv[ 0 ];

    bzero(&hctx, sizeof(HomogenizeContext));
    hctx.db = &db;
    hctx.block = DEF_ARG_B;
    hctx.track_in = DEF_ARG_I;
    hctx.track_out = DEF_ARG_II;
    hctx.ends = DEF_ARG_E;
    hctx.track_trim_name  = DEF_ARG_T;
    hctx.res = DEF_ARG_R;

    // process arguments

    int c;

    opterr = 0;

    while ((c = getopt(argc, argv, "mr:e:b:i:I:t:")) != -1)
    {
        switch (c)
        {
            case 'm':
                      hctx.copy_existing = 1;
                      break;

            case 'r':
                      hctx.res = atoi(optarg);
                      break;

            case 'b':
                      hctx.block = atoi(optarg);
                      break;

            case 'i':
                      hctx.track_in = optarg;
                      break;

            case 'I':
                      hctx.track_out = optarg;
                      break;

            case 't':
                      hctx.track_trim_name = optarg;
                      break;

            case 'e':
                      hctx.ends = atoi(optarg);
                      break;

            default:
                      usage(stdout, app);
                      exit(1);
        }
    }

    if (argc - optind != 2)
    {
        usage(stdout, app);
        exit(1);
    }

    if (hctx.ends < -1 || hctx.ends == 0)
    {
        fprintf(stderr, "invalid -e argument %d\n", hctx.ends);
        exit(1);
    }

    char* pcPathReadsIn = argv[optind++];
    char* pcPathOverlaps = argv[optind++];

    if ( (fileOvlIn = fopen(pcPathOverlaps, "r")) == NULL )
    {
        fprintf(stderr, "could not open '%s'\n", pcPathOverlaps);
        exit(1);
    }

    if ( Open_DB(pcPathReadsIn, &db) )
    {
        fprintf(stderr, "failed top open database %s\n", pcPathReadsIn);
        exit(1);
    }

    if ( hctx.track_out == NULL )
    {
        fprintf(stderr, "output track not set\n");
        exit(1);
    }

    if (hctx.ends != -1)
    {
        hctx.track_trim = track_load(&db, hctx.track_trim_name);
        if (hctx.track_trim == NULL)
        {
            fprintf(stderr, "-e requires a trim track\n");
            exit(1);
        }
    }

    // init

    ba_init();

    pctx = pass_init(fileOvlIn, NULL);

    pctx->split_b = 0;
    pctx->load_trace = 1;
    pctx->unpack_trace = 1;
    pctx->data = &hctx;

    // passes

    pre_homogenize(pctx, &hctx);

    pass(pctx, handler_homogenize);

    post_homogenize(&hctx);

    // cleanup

    pass_free(pctx);

    fclose(fileOvlIn);

    Close_DB(&db);

    return 0;
}
