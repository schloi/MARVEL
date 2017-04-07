/*******************************************************************************************
 *
 *  Displays the contents of a .las file
 *
 *  Date    : May 2015
 *
 *  Author  : MARVEL Team
 *
 *******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <assert.h>

#include <sys/param.h>

#include "lib/colors.h"
#include "lib/oflags.h"
#include "lib/pass.h"
#include "lib/tracks.h"
#include "lib/utils.h"

#include "db/DB.h"
#include "dalign/align.h"

// defaults

#define DEF_ARG_F           0

// constants

#define SORT_NATIVE         0
#define SORT_ID             1
#define SORT_AB             2
#define SORT_LENGTH         3
#define SORT_AE             4

#define SHOW_TRACE_BREAK_AFTER  5

typedef struct
{
    HITS_DB* db_a;
    HITS_DB* db_b;

    HITS_TRACK* trimtrack_a;
    HITS_TRACK* trimtrack_b;

    int trace;              // display trace
    int color;              // colorise output
    int sort;               // sort order (SORT_xxx)
    int revsort;            // reverse sort order
    int min_rlen;            // min read length
    int min_olen;            // min overlap length
    float min_identity;     // min overlap identity
    int show_aln;
    int flags;              // show flags
    int raw;

    int self_matches;       // only show A -> A overlaps

    // for displaying rangers of a.read ids
    int  ranges_in;
    int* ranges;
    int  ranges_npt;
    int  ranges_idx;

    int twidth;             // trace point spacing

    Alignment* align;       // alignment record (reused)
    Work_Data* align_work;  // global alignment module work data
} ShowContext;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

// oflags.c

extern OverlapFlag2Label oflag2label[];

// comparison functions for sorting overlaps

static int cmp_range(const void* l, const void* r)
{
    int x = *((int32*) l);
    int y = *((int32*) r);
    return (x - y);
}

static int cmp_ovls_abpos(const void* a, const void* b)
{
    Overlap* o1 = (Overlap*)a;
    Overlap* o2 = (Overlap*)b;

    int cmp = o1->path.abpos - o2->path.abpos;

    if (!cmp)
    {
        cmp = (o1->path.aepos - o1->path.abpos) - (o2->path.aepos - o2->path.abpos);
    }

    return cmp;
}

static int cmp_ovls_aepos(const void* a, const void* b)
{
    Overlap* o1 = (Overlap*)a;
    Overlap* o2 = (Overlap*)b;

    int cmp = o1->path.aepos - o2->path.aepos;

    if (!cmp)
    {
        cmp = (o1->path.aepos - o1->path.abpos) - (o2->path.aepos - o2->path.abpos);
    }

    return cmp;
}

static int cmp_ovls_id(const void* a, const void* b)
{
    Overlap* o1 = (Overlap*)a;
    Overlap* o2 = (Overlap*)b;

    return o1->bread - o2->bread;
}

static int cmp_ovls_length(const void* a, const void* b)
{
    Overlap* o1 = (Overlap*)a;
    Overlap* o2 = (Overlap*)b;

    return (o1->path.aepos - o1->path.abpos) - (o2->path.aepos - o2->path.abpos);
}


// displays a set of overlaps

static void show(ShowContext* ctx, Overlap* ovls, int novl)
{
    int i, end, incr;
    int rev = ctx->revsort;
    int show_trace = ctx->trace;
    int show_flags = ctx->flags;
    int show_aln = ctx->show_aln;
    HITS_TRACK* trim_a = ctx->trimtrack_a;
    HITS_TRACK* trim_b = ctx->trimtrack_b;
    HITS_DB* db_a = ctx->db_a;
    HITS_DB* db_b = ctx->db_b;

    char* color_a = ( ctx->color ? ANSI_COLOR_GREEN : "" );
    char* color_ovh = ( ctx->color ? ANSI_COLOR_RED : "" );
    char* color_flags = ( ctx->color ? ANSI_COLOR_BLUE : "" );
    char* color_reset = ( ctx->color ? ANSI_COLOR_RESET : "" );

    // load sequence in case we have to show the alignment

    int ovlALen = DB_READ_LEN(db_a, ovls->aread);

    if (show_aln)
    {
        ctx->align->alen = ovlALen;
        Load_Read(db_a, ovls->aread, ctx->align->aseq, 0);
    }

    // sort overlaps

    switch (ctx->sort)
    {
        case SORT_ID:
            qsort(ovls, novl, sizeof(Overlap), cmp_ovls_id);
            break;

        case SORT_LENGTH:
            qsort(ovls, novl, sizeof(Overlap), cmp_ovls_length);
            break;

        case SORT_AB:
            qsort(ovls, novl, sizeof(Overlap), cmp_ovls_abpos);
            break;

        case SORT_AE:
            qsort(ovls, novl, sizeof(Overlap), cmp_ovls_aepos);
            break;

        case SORT_NATIVE:
        default:
            break;
    }

    // reverse sort (or not)

    if (rev)
    {
        incr = -1;
        end = -1;
        i = novl - 1;
    }
    else
    {
        incr = 1;
        end = novl;
        i = 0;
    }

    int trim_ab, trim_ae;

    // display trim track data

    if (trim_a)
    {
        get_trim(db_a, trim_a, ovls->aread, &trim_ab, &trim_ae);
    }
    else
    {
        trim_ab = 0;
        trim_ae = ovlALen;
    }

    for (; i != end; i += incr)
    {
        Overlap* ovl = ovls + i;

        if (ctx->self_matches && ovl->aread != ovl->bread)
        {
            continue;
        }

        // length filter
        int ovlBLen = DB_READ_LEN(db_b, ovl->bread);
        if (ovlBLen < ctx->min_rlen)
        {
            continue;
        }

        // length filter
        int ovlLen = MAX(ovl->path.aepos - ovl->path.abpos, abs(ovl->path.bepos - ovl->path.bbpos));
        if (ovlLen < ctx->min_olen)
        {
            continue;
        }

        int len = ovl->path.aepos - ovl->path.abpos;
        float diff = 100.0 * ovl->path.diffs / len;

        // diff filter

        if (100 - diff < ctx->min_identity)
        {
            continue;
        }

        // read ids and overlap coordinates

        if (ctx->raw)
        {
            printf("%d %d %d %d %d %d %d %d\n",
                    ovl->aread,
                    ovl->bread,

                    ovl->flags & OVL_COMP ? 1 : 0,

                    ovl->path.abpos, ovl->path.aepos,
                    ovl->path.bbpos, ovl->path.bepos,

                    ovl->path.diffs);

            if (show_aln)
            {
                ctx->align->blen = ovlBLen;
                Load_Read(db_b, ovls[i].bread, ctx->align->bseq, 0);

                ctx->align->path = &(ovls[i].path);

                if (ovls[i].flags & OVL_COMP)
                {
                    Complement_Seq(ctx->align->bseq, ovlBLen);
                }

                Compute_Trace_PTS(ctx->align, ctx->align_work, ctx->twidth, 0);

                int width = db_a->reads[ ovls[i].aread ].rlen + db_b->reads[ ovls[i].bread ].rlen;

                Print_Reference(stdout, ctx->align, ctx->align_work, 0, width, 0, 0, 0);
            }

            continue;
        }

        printf("%s%7d%s %s%c%s %7d  %s%5d..%5d%s x %5d..%5d",
                color_a,
                ovls[i].aread,
                color_reset,

                color_flags,
                ovls[i].flags & OVL_COMP ? 'c' : 'n',
                color_reset,

                ovls[i].bread,
                color_a,
                ovls[i].path.abpos, ovls[i].path.aepos,
                color_reset,
                ovls[i].path.bbpos, ovls[i].path.bepos);

        int trim_bb, trim_be;

        // read lengths and trim track data

        if (trim_b)
        {
            get_trim(db_b, trim_b, ovls[i].bread, &trim_bb, &trim_be);

            printf("  %s%5d [%5d..%5d]%s x %5d [%5d..%5d]",
                    color_a,
                    ovlALen,
                    trim_ab, trim_ae,
                    color_reset,
                    ovlBLen,
                    trim_bb, trim_be);
        }
        else
        {
            trim_bb = 0;
            trim_be = ovlBLen;

            printf("  %s%5d%s x %5d",
                    color_a,
                    ovlALen,
                    color_reset,
                    ovlBLen);
        }

        printf("  %4.1f", diff);

        // indicate true overlaps (not containments or local alignments)

        printf(" %s%c%c%s",
                color_ovh,
                ovls[i].path.abpos == trim_ab ? '<' : ' ',
                ovls[i].path.aepos == trim_ae ? '>' : ' ',
                color_reset);

        // show overlap record flags

        if (show_flags && (ovls[i].flags & ~OVL_COMP))
        {
            char flags[OVL_FLAGS + 1];
            flags2str(flags, ovls[i].flags);

            // ignore OVL_COMP
            printf(" %s%s%s\n", color_flags, flags + 1, color_reset);
        }
        else
        {
            printf("\n");
        }

        // show trace points

        if (show_trace)
        {
            int j;
            ovl_trace* trace = ovls[i].path.trace;

            assert( (ovls[i].path.tlen % 2) == 0 );

            for (j = 0; j < ovls[i].path.tlen; j += 2)
            {
                if ( j > 0 && (j % (SHOW_TRACE_BREAK_AFTER * 2)) == 0) printf("\n");

                printf(" (%3d, %2d)", trace[j+1], trace[j]);
            }
            printf("\n");
        }

        // show full alignment

        if (show_aln)
        {
            ctx->align->blen = ovlBLen;
            Load_Read(db_b, ovls[i].bread, ctx->align->bseq, 0);

            ctx->align->path = &(ovls[i].path);

            if (ovls[i].flags & OVL_COMP)
            {
                Complement_Seq(ctx->align->bseq, ovlBLen);
            }

            Compute_Trace_PTS(ctx->align, ctx->align_work, ctx->twidth, 0);

            Print_Reference(stdout, ctx->align, ctx->align_work, 0, 100, 0, 0, 5);

            printf("\n");
        }
    }
}

static int parse_ranges(int argc, char* argv[], int* _reps, int** _pts)
{
    int *pts = (int*)malloc(sizeof(int) * 2 * (2 + argc));
    int reps = 0;

    if (argc > 0)
    {
        int   c, b, e;
        char* eptr, *fptr;

        for (c = 0; c < argc; c++)
        {
            if (argv[c][0] == '#')
            {
                fprintf(stderr, "# is not allowed as range start, '%s'\n", argv[c]);
                return 0;
            }
            else
            {
                b = strtol(argv[c], &eptr, 10);

                if (b < 0)
                {
                    fprintf(stderr, "Non-positive index?, '%d'\n", b);
                    return 0;
                }
            }

            if (eptr > argv[c])
            {
                if (*eptr == '\0')
                {
                    pts[reps++] = b;
                    pts[reps++] = b;

                    continue;
                }
                else if (*eptr == '-')
                {
                    if (eptr[1] == '#')
                    {
                        e = INT32_MAX;
                        fptr = eptr + 2;
                    }
                    else
                    {
                        e = strtol(eptr + 1, &fptr, 10);
                    }

                    if (fptr > eptr + 1 && *fptr == 0 && eptr[1] != '-')
                    {
                        pts[reps++] = b;
                        pts[reps++] = e;

                        if (b > e)
                        {
                            fprintf(stderr, "Empty range '%s'\n", argv[c]);
                            return 0;
                        }

                        continue;
                    }
                }
            }

            fprintf(stderr, "argument '%s' is not an integer range\n", argv[c]);
            return 0;
        }

        qsort(pts, reps / 2, sizeof(int64), cmp_range);

        b = 0;

        for (c = 0; c < reps; c += 2)
        {
            if (b > 0 && pts[b - 1] >= pts[c] - 1)
            {
                if (pts[c + 1] > pts[b - 1])
                {
                    pts[b - 1] = pts[c + 1];
                }
            }
            else
            {
                pts[b++] = pts[c];
                pts[b++] = pts[c + 1];
            }
        }

        pts[b++] = INT32_MAX;
        reps = b;
    }
    else
    {
        pts[reps++] = 0;
        pts[reps++] = INT32_MAX;
    }

    *_reps = reps;
    *_pts = pts;

    return 1;
}

static void pre_show(PassContext* pctx, ShowContext* sctx)
{
    sctx->twidth = pctx->twidth;

    sctx->ranges_idx = 1;
    sctx->ranges_npt = sctx->ranges[0];
}

static void post_show()
{
}

static int handler_show(void* _ctx, Overlap* ovls, int novl)
{
    ShowContext* ctx = (ShowContext*)_ctx;

    if (DB_READ_LEN(ctx->db_a, ovls->aread) < ctx->min_rlen)
    {
        return 1;
    }

    int a = ovls->aread;

    if (ctx->ranges_in)
    {
        while (a > ctx->ranges_npt)
        {
            ctx->ranges_npt = ctx->ranges[ctx->ranges_idx++];

            if (a < ctx->ranges_npt)
            {
                ctx->ranges_in = 0;
                break;
            }

            ctx->ranges_npt = ctx->ranges[ctx->ranges_idx++];
        }
    }
    else
    {
        if (ctx->ranges_npt == INT32_MAX)
        {
            return 0;
        }

        while (a >= ctx->ranges_npt)
        {
            ctx->ranges_npt = ctx->ranges[ctx->ranges_idx++];

            if (a <= ctx->ranges_npt)
            {
                ctx->ranges_in = 1;
                break;
            }

            ctx->ranges_npt = ctx->ranges[ctx->ranges_idx++];
        }
    }

    if (ctx->ranges_in)
    {
        show(ctx, ovls, novl);
    }

    return 1;
}

static void usage()
{
    fprintf(stderr, "[-tfrcm] [-s <liLI>] [-xo <int>] [-T <track>] [-i <float>] <db> [db.2] <overlaps> [ <reads:range> ... ]\n");
    fprintf(stderr, "options: -t         ... show trace\n");
    fprintf(stderr, "         -c         ... colorise output\n");
    fprintf(stderr, "         -f         ... show flags\n");
    fprintf(stderr, "         -a         ... show alignment\n");
    fprintf(stderr, "         -s <liLI>  ... sort by length or id (ASCENDING or descending)\n");
    fprintf(stderr, "         -x <int>   ... minimum read length\n");
    fprintf(stderr, "         -o <int>   ... minimum overlap length\n");
    fprintf(stderr, "         -i <float> ... minimum identity\n");
    fprintf(stderr, "         -T <track> ... use trim track (if [db.2] is present then two -T possible)\n");
    fprintf(stderr, "         -r         ... unformatted output\n");
    fprintf(stderr, "         -m         ... mixed db mode\n");
};

int main(int argc, char* argv[])
{
    HITS_DB* db_a = malloc( sizeof(HITS_DB) );
    HITS_DB* db_b = NULL;

    FILE* fileOvlIn;

    PassContext* pctx;
    ShowContext sctx;

    bzero(&sctx, sizeof(ShowContext));

    // process arguments

    int c;

    int mixed = 0;
    char* trim_a = NULL;
    char* trim_b = NULL;

    opterr = 0;

    sctx.raw = 0;
    sctx.min_rlen = 0;
    sctx.min_olen = 0;
    sctx.min_identity = -1;
    sctx.sort = SORT_NATIVE;
    sctx.flags = DEF_ARG_F;

    while ((c = getopt(argc, argv, "mrIfactx:i:o:s:T:")) != -1)
    {
        switch (c)
        {
            case 'm':
                      mixed = 1;
                      break;

            case 'r':
                      sctx.raw = 1;
                      break;

            case 'I':
                      sctx.self_matches = 1;
                      break;

            case 'f':
                      sctx.flags = 1;
                      break;

            case 'T':
            {
                      if(trim_a == NULL)
                    	  trim_a = optarg;
                      else
                    	  trim_b = optarg;
            }
                      break;

            case 'a':
                      sctx.show_aln = 1;
                      break;

            case 'c':
                      sctx.color = 1;
                      break;

            case 't':
                      sctx.trace = 1;
                      break;

            case 'x':
                      sctx.min_rlen = atoi(optarg);
                      break;

            case 'o':
                      sctx.min_olen = atoi(optarg);
                      break;

            case 's':
                      if (islower(optarg[0]))
                      {
                        sctx.revsort = 1;
                      }

                      switch( tolower(optarg[0]) )
                      {
                        case 'i': sctx.sort = SORT_ID;
                                  break;

                        case 'l': sctx.sort = SORT_LENGTH;
                                  break;

                        case 'e': sctx.sort = SORT_AE;
                                  break;

                        case 'b':
                        default : sctx.sort = SORT_AB;
                                  break;
                      }

                      break;

            case 'i':
                      sctx.min_identity = atof(optarg);
                      break;

            default:
                      printf("Unknown option: %s\n", argv[optind-1]);
                      usage();
                      exit(1);
        }
    }

    if (argc - optind < (2 + mixed))
    {
        usage();
        exit(1);
    }

    char* pcPathReadsIn_a = argv[optind++];
    char* pcPathReadsIn_b = NULL;

    if (mixed)
    {
        pcPathReadsIn_b = argv[optind++];
    }

    char* pcPathOverlaps = argv[optind++];

    if ( (fileOvlIn = fopen(pcPathOverlaps, "r")) == NULL )
    {
        fprintf(stderr, "could not open '%s'\n", pcPathOverlaps);
        exit(1);
    }

    if (Open_DB(pcPathReadsIn_a, db_a))
    {
        printf("could not open '%s'\n", pcPathReadsIn_a);
    }

    if (mixed)
    {
        db_b = malloc( sizeof(HITS_DB) );

        if (Open_DB(pcPathReadsIn_b, db_b))
        {
            printf("could not open '%s'\n", pcPathReadsIn_b);
        }
    }
    else
    {
        db_b = db_a;
    }

    int reps;
    int* pts = NULL;

    parse_ranges(argc - optind, argv + optind, &reps, &pts);

    // init

    if (trim_a != NULL)
    {
        sctx.trimtrack_a = track_load(db_a, trim_a);

        if (!sctx.trimtrack_a)
        {
            fprintf(stderr, "failed to load track %s\n", trim_a);
            exit(1);
        }

        if(trim_b == NULL)
        	trim_b = trim_a;

        sctx.trimtrack_b = track_load(db_b, trim_b);

        if (!sctx.trimtrack_b)
        {
            fprintf(stderr, "failed to load track %s\n", trim_b);
            exit(1);
        }
    }

    if (sctx.show_aln)
    {
        sctx.align = (Alignment*)malloc(sizeof(Alignment));

        sctx.align->aseq = New_Read_Buffer(db_a);
        sctx.align->bseq = New_Read_Buffer(db_b);

        sctx.align_work = New_Work_Data();
    }

    sctx.ranges = pts;
    sctx.db_a = db_a;
    sctx.db_b = db_b;

    pctx = pass_init(fileOvlIn, NULL);

    pctx->split_b = 0;
    pctx->load_trace = (sctx.show_aln || sctx.trace);
    pctx->unpack_trace = (sctx.show_aln || sctx.trace);
    pctx->data = &sctx;

    // pass

    pre_show(pctx, &sctx);

    pass(pctx, handler_show);

    post_show();

    // cleanup

    if (sctx.show_aln)
    {
        free(sctx.align->aseq - 1);
        free(sctx.align->bseq - 1);

        free(sctx.align);
        Free_Work_Data(sctx.align_work);
    }

    Close_DB(db_a);

    pass_free(pctx);

    fclose(fileOvlIn);

    free(db_a);

    if (mixed)
    {
        Close_DB(db_b);

        free(db_b);
    }

    return 0;
}
