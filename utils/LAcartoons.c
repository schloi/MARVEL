/*******************************************************************************************
 *
 *  Prints an ASCII representation of the overlaps. Uses ANSI color codes.
 *  Best used jointly with 'less -SR'
 *
 *  Date    : October 2014
 *
 *  Author  : MARVEL Team
 *
 *******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <sys/param.h>

#include "lib/colors.h"
#include "lib/oflags.h"
#include "lib/utils.h"
#include "lib/pass.h"
#include "lib/tracks.h"

#include "db/DB.h"
#include "dalign/align.h"

#define OVL_HIDE OVL_TEMP       // use the temp flag to hide overlaps
                                // not matching the criteria

#define SORT_ID 1
#define SORT_AB 2
#define SORT_LENGTH 3

#define SCALE(x, tw)     ( (int) ((x)/(float)(tw)) )
#define REVSCALE(x, tw)  ( (int) ((x)*(float)(tw)) )

#define REPEAT(c,n) { char strrep[1024]; memset(strrep, c, n); strrep[(n)] = '\0'; printf("%s", strrep); }

typedef struct
{
    HITS_DB* db;
    HITS_TRACK* qtrack;

    ovl_header_twidth twidth;

    int q;
    int trim;
    int ruler;
    int discarded;
    int coverage;
    int show_overlaps;
    int flags;

    int sort;
    int revsort;

    float q_scale;

    int histo_max;
    int64* histo_cov;

    int min_len;
    float min_identity;

    int  ranges_in;
    int* ranges;
    int  ranges_npt;
    int  ranges_idx;
} CartoonsContext;

extern char *optarg;
extern int optind, opterr, optopt;

static int ORDER(const void* l, const void* r)
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

static char* match_string(CartoonsContext* cctx, Overlap* ovl, ovl_header_twidth twidth)
{
    static char match[64 * 1024];

    int i, q;
    char* color = "";
    char* cur = match;
    char* pcolor = NULL;
    int width;
    char c;
    ovl_trace* trace = ovl->path.trace;

    int bp = ovl->path.bbpos;

    if (ovl->path.tlen == 0)
    {
        int len = (ovl->path.aepos - ovl->path.abpos) / twidth;
        memset(match, '*', len);
        match[len] = '\0';

        return match;
    }

    for (i = 0; i < ovl->path.tlen; i += 2)
    {
        if (i == 0)
        {
            width = twidth - (ovl->path.abpos % twidth) + trace[i + 1];
            bp += trace[i + 1];
        }
        else if (i == ovl->path.tlen - 1)
        {
            width = (ovl->path.aepos % twidth) + (ovl->path.bepos - bp);
            bp = ovl->path.bepos;
        }
        else
        {
            width = twidth + trace[i + 1];
            bp += trace[i + 1];
        }

        q = 200. * trace[i] / width * cctx->q_scale;

        if (q < 5)
        {
            color = ANSI_COLOR_WHITE;
            c = '0';
        }
        else if (q < 10)
        {
            color = ANSI_COLOR_GREEN;
            c = '1';
        }
        else if (q < 20)
        {
            color = ""; // ANSI_COLOR_YELLOW;
            c = '2';
        }
        else if (q < 30)
        {
            color = ANSI_COLOR_YELLOW;
            c = '3';
        }
        else
        {
            color = ANSI_COLOR_RED;
            c = '.';
        }

        if (pcolor != color)
        {
            sprintf(cur, "%s", ANSI_COLOR_RESET);
            cur += strlen(ANSI_COLOR_RESET);

            sprintf(cur, "%s", color);
            pcolor = color;

            cur += strlen(color);
        }

        *cur = c;
        cur++;
    }

    sprintf(cur, "%s", ANSI_COLOR_RESET);
    cur += strlen(ANSI_COLOR_RESET);

    *cur = '\0';

    return match;
}

static int round_up(int n, int f)
{
    return (n + f - 1) - ((n - 1) % f);
}

static int round_down(int n, int f)
{
    return n - n % f;
}

static char* read_string(int len, int a, HITS_TRACK* track_q)
{
    static char string[128*1024];
    char* cur = string;
    char* color = "";
    char* pcolor = NULL;

    if (track_q != NULL)
    {
        track_anno* q_anno = track_q->anno;
        track_data* q_data = track_q->data;

        track_anno aob = q_anno[a] / sizeof(track_data);
        track_anno aoe = q_anno[a+1] / sizeof(track_data);

        track_anno aoc = 0;
        int q;
        char c;

        assert((track_anno)len == aoe - aob);

        for (; aoc < (aoe-aob); aoc++)
        {
            q = q_data[aob + aoc];

            if (q == 0)
            {
                color = "";
                c = '*';
            }
            else if (q < 5)
            {
                color = ANSI_COLOR_WHITE;
                c = '0';
            }
            else if (q < 10)
            {
                color = ANSI_COLOR_GREEN;
                c = '1';
            }
            else if (q < 20)
            {
                color = ""; // ANSI_COLOR_GREEN;
                c = '2';
            }
            else if (q < 30)
            {
                color = ANSI_COLOR_YELLOW;
                c = '3';
            }
            else
            {
                color = ANSI_COLOR_RED;
                c = '.';
            }

            if (pcolor != color)
            {
                sprintf(cur, "%s", ANSI_COLOR_RESET);
                cur += strlen(ANSI_COLOR_RESET);

                sprintf(cur, "%s", color);
                pcolor = color;

                cur += strlen(color);
            }

            *cur = c;
            cur++;
        }

        *cur = '\0';
    }
    else
    {
        int i;
        for (i = 0; i < len; i++)
        {
            string[i] = '*';
        }

        string[len] = '\0';
    }

    return string;
}

static char* ruler_string(int len, int twidth)
{
    static char str[1024];
    char num[10];

    int i = 0;

    while (i < len)
    {
        if (i % 10)
        {
            str[i++] = ' ';
        }
        else
        {
            str[i] = '|';

            sprintf(num, "%d", (i * twidth) / 1000);

            i++;

            char* c = num;

            while (*c)
            {
                str[i++] = *c++;
            }
        }
    }

    str[i] = '\0';

    return str;
}

static void draw(CartoonsContext* ctx, Overlap* pOvls, int novls)
{
    HITS_DB* db = ctx->db;
    HITS_TRACK* qtrack = ctx->qtrack;
    char flags[OVL_FLAGS+1];

    ovl_header_twidth twidth = ctx->twidth;
    int ovhtrim = ctx->trim;
    int ruler = ctx->ruler;
    int rev = ctx->revsort;
    int coverage = ctx->coverage;
    int show_flags = ctx->flags;

    int flags_width = show_flags * (OVL_FLAGS + 2);

    int nMaxLeftOvh, left;
    int i;

    int ovlALen = DB_READ_LEN(db, pOvls[0].aread);
    int alen = round_up( ovlALen, twidth );

    if (ovhtrim)
    {
        nMaxLeftOvh = REVSCALE(10, twidth);
    }
    else
    {
        nMaxLeftOvh = 0;

        for (i = 0; i < novls; i++)
        {
            left = round_down(pOvls[i].path.bbpos, twidth) - round_down(pOvls[i].path.abpos, twidth);
            nMaxLeftOvh = MAX(left, nMaxLeftOvh);
        }

        nMaxLeftOvh += REVSCALE(1, twidth);
    }

    if (ruler)
    {
        REPEAT(' ', SCALE(nMaxLeftOvh, twidth) + 25 + flags_width);
        printf("%s\n", ruler_string( SCALE(ovlALen, twidth), twidth ));
    }

    printf(ANSI_COLOR_CYAN "% 10d#[%5d..%5d]", pOvls[0].aread + 1, 0, ovlALen );
    REPEAT(' ', SCALE(nMaxLeftOvh, twidth) + flags_width);
    printf("%s" ANSI_COLOR_RESET, read_string( SCALE(alen, twidth), pOvls->aread, qtrack ) );

    printf("\n");

    if (coverage)
    {
        // UNDECIDED: count partial segments or not

        int alen_seg = alen / ctx->twidth;

        bzero(ctx->histo_cov, sizeof(int64) * ctx->histo_max);

        for (i = 0; i < novls; i++)
        {
            int j;
            int last = pOvls[i].path.aepos / ctx->twidth;


            for (j = pOvls[i].path.abpos / ctx->twidth; j <= last; j++)
            {
                ctx->histo_cov[j]++;
            }
        }

        int cov_max = 0;

        for (i = 0; i <= alen_seg; i++)
        {
            if (ctx->histo_cov[i] == 0)
            {
                continue;
            }

            ctx->histo_cov[i] = log10( ctx->histo_cov[i] );

            if (ctx->histo_cov[i] > cov_max)
            {
                cov_max = ctx->histo_cov[i];
            }
        }

        for (i = cov_max; i != 0; i--)
        {
            REPEAT(' ', SCALE(nMaxLeftOvh, twidth) + 25 + flags_width);

            int j;
            for (j = 0; j < alen_seg; j++)
            {
                if (ctx->histo_cov[j] >= i)
                {
                    printf("*");
                }
                else
                {
                    printf(" ");
                }
            }
            printf(" %5dx\n", (int)pow(10, i));
        }
    }

    if (ctx->show_overlaps)
    {
        int pre, post, indent, pre_match, incr, end;
        char orient;
        char* color;

        if (rev)
        {
            incr = -1;
            end = -1;
            i = novls - 1;
        }
        else
        {
            incr = 1;
            end = novls;
            i = 0;
        }

        int ovlBLen;
        for (; i != end; i += incr)
        {
            if ( (!ctx->discarded && (pOvls[i].flags & OVL_DISCARD)) || pOvls[i].flags & OVL_HIDE )
            {
                continue;
            }
            ovlBLen = DB_READ_LEN(db,pOvls[i].bread);
            pre = round_down(pOvls[i].path.bbpos, twidth);
            post = round_up(ovlBLen, twidth) - round_up(pOvls[i].path.bepos, twidth);
            indent = nMaxLeftOvh + round_down(pOvls[i].path.abpos, twidth) - pre;

            orient = pOvls[i].flags & OVL_COMP  ? '<' : '>';

            if (pOvls[i].flags & OVL_DISCARD)
            {
                color = ANSI_COLOR_RED;
            }
            else
            {
                color = pOvls[i].flags & OVL_COMP ? ANSI_COLOR_BLUE : "";
            }

            pre_match = SCALE(indent + pre, twidth);
            indent = SCALE(indent, twidth);
            pre = pre_match - indent;
            post = SCALE(post, twidth);

            printf("%s%10d%c[%5d..%5d]" ANSI_COLOR_RESET, color, pOvls[i].bread + 1, orient, pOvls[i].path.abpos, pOvls[i].path.aepos);

            if (show_flags)
            {
                flags2str(flags, pOvls[i].flags);
                printf("[%s]", flags);
            }

            if (ovhtrim)
            {
                indent = SCALE(round_down(pOvls[i].path.abpos, twidth), twidth);

                if (pre < 10)
                {
                    REPEAT(' ', 10 - pre + indent);
                    REPEAT('-', pre);
                }
                else
                {
                    REPEAT(' ', indent);
                    printf(" %5d]", pOvls[i].path.bbpos);
                    REPEAT('-', 3);
                }
            }
            else
            {
                REPEAT(' ', indent);
                REPEAT('-', pre);
            }

            printf("%s", match_string(ctx, pOvls + i, twidth) );

            if (ovhtrim && post > 9)
            {
                REPEAT('-', 3);
                printf("[%d", ovlBLen - pOvls[i].path.bepos);
            }
            else
            {
                REPEAT('-', post);
            }

            printf("\n");
        }
    }

    printf("\n");
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

                if (b < 1)
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

        qsort(pts, reps / 2, sizeof(int64), ORDER);

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
        pts[reps++] = 1;
        pts[reps++] = INT32_MAX;
    }

    *_reps = reps;
    *_pts = pts;

    return 1;
}

static void pre_cartoons(PassContext* pctx, CartoonsContext* cctx)
{
    cctx->twidth = pctx->twidth;

    cctx->ranges_idx = 1;
    cctx->ranges_npt = cctx->ranges[0];

    cctx->histo_max = (DB_READ_MAXLEN(cctx->db) + pctx->twidth - 1) / pctx->twidth;
    cctx->histo_cov = (int64*)malloc(sizeof(int64) * cctx->histo_max);

    if (cctx->q)
    {
        cctx->qtrack = track_load(cctx->db, TRACK_Q);

        if (!cctx->qtrack)
        {
            fprintf(stderr, "could not load track %s\n", TRACK_Q);
            exit(1);
        }
    }
}

static void post_cartoons(CartoonsContext* cctx)
{
    free(cctx->histo_cov);

    Close_Track(cctx->db, TRACK_Q);
}

static int handler_cartoons(void* _ctx, Overlap* ovls, int novl)
{
    CartoonsContext* ctx = (CartoonsContext*)_ctx;

    // filter

    int i;
    int kept = 0;
    int len;

    if (DB_READ_LEN(ctx->db, ovls->aread) < ctx->min_len)
    {
        return 1;
    }

    for (i = 0; i < novl; i++)
    {
        if (DB_READ_LEN(ctx->db, ovls[i].bread) < ctx->min_len)
        {
            ovls[i].flags |= OVL_HIDE;
            continue;
        }

        len = ovls[i].path.aepos - ovls[i].path.abpos;

        if (100 - (100.0 * ovls[i].path.diffs / len) < ctx->min_identity)
        {
            ovls[i].flags |= OVL_HIDE;
            continue;
        }

        kept++;
    }

    if (kept == 0)
    {
        return 1;
    }

    int a = ovls->aread + 1;

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
        switch (ctx->sort)
        {
            case SORT_ID:
                qsort(ovls, novl, sizeof(Overlap), cmp_ovls_id);
                break;

            case SORT_LENGTH:
                qsort(ovls, novl, sizeof(Overlap), cmp_ovls_length);
                break;

            case SORT_AB:
            default:
                qsort(ovls, novl, sizeof(Overlap), cmp_ovls_abpos);
                break;
        }

        draw(ctx, ovls, novl);
    }


    return 1;
}


static void usage()
{
    printf("[-rtqdF] [-s <l|i>] [-x <int>] [-i <double>] [-F <double>] <reads:db> <overlaps:ovl> [ <reads:range> ... ]\n");

    printf("options: -r          ... show ruler\n");
    printf("         -t          ... trim overhangs\n");
    printf("         -q          ... show a read quality (requires a quality track)\n");
    printf("         -d          ... show discarded overlaps\n");
    printf("         -c          ... show coverage\n");
    printf("         -C          ... only coverage (hide overlaps)\n");
    printf("         -f          ... show flags\n");

    printf("         -s [li]     ... oder by length or read id\n");
    printf("         -x <int>    ... set minimum read length to <int>\n");
    printf("         -i <float>  ... set minimum identity to <float>\n");
    printf("         -F <float>  ... scale quality scores by <float>\n");
};

int main(int argc, char* argv[])
{
    HITS_DB db;
    FILE* fileOvlIn;

    PassContext* pctx;
    CartoonsContext cctx;

    bzero(&cctx, sizeof(CartoonsContext));

    // process arguments

    int c;

    opterr = 0;

    cctx.q_scale = 1.0;
    cctx.min_len = 0;
    cctx.min_identity = -1;
    cctx.show_overlaps = 1;

    while ((c = getopt(argc, argv, "cCdqotrx:i:s:F:f")) != -1)
    {
        switch (c)
        {
            case 'f':
                      cctx.flags = 1;
                      break;

            case 'C':
                      cctx.show_overlaps = 0;
                      cctx.coverage = 1;
                      break;

            case 'c':
                      cctx.coverage = 1;
                      break;

            case 'd':
                      cctx.discarded = 1;
                      break;

            case 'F':
                      cctx.q_scale = atof(optarg);
                      break;

            case 'q':
                      cctx.q = 1;
                      break;

            case 't':
                      cctx.trim = 1;
                      break;

            case 'r':
                      cctx.ruler = 1;
                      break;

            case 'x':
                      cctx.min_len = atoi(optarg);
                      break;

            case 's':
                      if (islower(optarg[0]))
                      {
                        cctx.revsort = 1;
                      }

                      switch( tolower(optarg[0]) )
                      {
                        case 'i': cctx.sort = SORT_ID;
                                  break;
                        case 'l': cctx.sort = SORT_LENGTH;
                                  break;
                        case 'a':
                        default : cctx.sort = SORT_AB;
                                  break;
                      }

                      break;

            case 'i':
                      cctx.min_identity = atof(optarg);
                      break;

            default:
                      usage();
                      exit(1);
        }
    }

    if (argc - optind < 2)
    {
        usage();
        exit(1);
    }

    char* pcPathReadsIn = argv[optind++];
    char* pcPathOverlaps = argv[optind++];

    if ( (fileOvlIn = fopen(pcPathOverlaps, "r")) == NULL )
    {
        fprintf(stderr, "could not open '%s'\n", pcPathOverlaps);
        exit(1);
    }

    if (Open_DB(pcPathReadsIn, &db))
    {
        printf("could not open '%s'\n", pcPathReadsIn);
    }

    int reps;
    int* pts = NULL;

    parse_ranges(argc - optind, argv + optind, &reps, &pts);

    // init

    cctx.ranges = pts;
    cctx.db = &db;

    pctx = pass_init(fileOvlIn, NULL);

    pctx->split_b = 0;
    pctx->load_trace = 1;
    pctx->unpack_trace = 1;
    pctx->data = &cctx;

    // pass

    pre_cartoons(pctx, &cctx);

//    Trim_DB(&db);

    pass(pctx, handler_cartoons);

    post_cartoons(&cctx);

    // cleanup

    Close_DB(&db);

    fclose(fileOvlIn);

    return 0;
}
