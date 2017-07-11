/*******************************************************************************************
 *
 *  Builds the overlap graph
 *  (C implementation of build_og.py)
 *
 *  Date    : January 2016
 *
 *  Author  : MARVEL Team
 *
 *******************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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

// read status

#define STATUS_CONTAINED ( 1 << 0 )
#define STATUS_WIDOW     ( 1 << 1 )
#define STATUS_PROPER    ( 1 << 2 )
#define STATUS_USED      ( 1 << 3 )
#define STATUS_OPTIONAL  ( 1 << 4 )

// graph format

typedef enum { FORMAT_GML, FORMAT_GRAPHML, FORMAT_TGF } GraphFormat;

// contained edges sorting

typedef enum { EDGE_SORT_OVL, EDGE_SORT_OVH } OgEdgeSort;

// defaults

#define DEF_ARG_C        0
#define DEF_ARG_F        "graphml"
#define DEF_ARG_P        "ovl"
#define DEF_ARG_T        TRACK_TRIM

// switches

#undef DEBUG_INSPECT

// used to store overlaps, basically a copy of everything we need from an Overlap struct

typedef struct
{
    int a, b;
    int ab, ae, bb, be;
    int flags;
    int ovh;

    unsigned short diffs;
} OgEdge;

// maintains the state of the app

typedef struct
{
    HITS_DB* db;
    HITS_TRACK* trimtrack;
    char *trimName;

    // stats counters

    uint64 stats_edges;
    uint64 stats_redges;
    uint64 stats_symdiscard;
    uint64 stats_pedges_dropped;

    // command line args

    char* path_graph;
    int contained;
    int split;
    GraphFormat gformat;

    OgEdgeSort edgeprio;

    // read -> component id

    int* comp;
    int ncomp;

    // discard during second pass

    int* discard;
    int ndiscard;
    int maxdiscard;

    // read -> status

    unsigned char* status;

    uint64* nleft;
    uint64* nright;

    OgEdge* left;
    OgEdge* right;

    int* curleft;
    int* curright;

} OgBuildContext;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

// graph output

static void print_graph_graphml_edge(FILE* f, OgEdge* e, char side)
{
    int div = 100.0 * 2 * e->diffs / ( (e->ae - e->ab) + (e->be - e->bb) );

    fprintf(f, "    <edge source=\"%d\" target=\"%d\">\n", e->a, e->b);
    fprintf(f, "      <data key=\"length\">%d</data>\n", e->ovh);
    fprintf(f, "      <data key=\"flags\">%d</data>\n", e->flags);
    fprintf(f, "      <data key=\"divergence\">%d</data>\n", div);
    fprintf(f, "      <data key=\"end\">%c</data>\n", side);
    fprintf(f, "    </edge>\n");
}

static void print_graph_gml_edge(FILE* f, OgEdge* e, char side)
{
    int div = 100.0 * 2 * e->diffs / ( (e->ae - e->ab) + (e->be - e->bb) );

    fprintf(f, "  edge [\n");
    fprintf(f, "    source %d\n", e->a);
    fprintf(f, "    target %d\n", e->b);
    fprintf(f, "    length %d\n", e->ovh);
    fprintf(f, "    flags %d\n", e->flags);
    fprintf(f, "    divergence %d\n", div);
    fprintf(f, "    end \"%c\"\n", side);
    fprintf(f, "  ]\n");
}

//
// TGF export
//
//      we are using the label as storage for various additional values
//      the standard format
//
//      id_i label_i
//      #
//      id_j id_k label_edge_i
//
//      turns into
//
//      id_i optional_i left_edges_i right_edges_i
//      #
//      id_j id_k overhang_i flags_i divergence_i side_i
//

static void print_graph_tgf_edge(FILE* f, OgEdge* e, char side)
{
    int div = 100.0 * 2 * e->diffs / ( (e->ae - e->ab) + (e->be - e->bb) );

    fprintf(f, "%d %d %d %d %d %c\n", e->a, e->b, e->ovh, e->flags, div, side);
}

static void print_graph_tgf(OgBuildContext* octx, FILE* f, int component)
{
    uint64* nleft = octx->nleft;
    uint64* nright = octx->nright;
    OgEdge* left = octx->left;
    OgEdge* right = octx->right;
    unsigned char* status = octx->status;

    int* niedges = malloc(sizeof(int) * octx->db->nreads);
    int* noedges = malloc(sizeof(int) * octx->db->nreads);
    bzero(niedges, sizeof(int) * octx->db->nreads);
    bzero(noedges, sizeof(int) * octx->db->nreads);

    int aread;
    for ( aread = 0; aread < octx->db->nreads; aread++ )
    {
        if ( !(status[aread] & STATUS_PROPER) )
        {
            continue;
        }

        if ( component != -1 && octx->comp[aread] != component )
        {
            continue;
        }

        int used = 0;
        uint64 b = nleft[aread];
        uint64 e = nleft[aread + 1];

        while (b < e)
        {
            OgEdge* e = left + b;
            int bread = e->b;

            if ( (status[bread] & STATUS_PROPER) )
            {
                status[bread] |= STATUS_USED;
                used = 1;

                noedges[aread]++;
                niedges[bread]++;
            }

            b++;
        }

        b = nright[aread];
        e = nright[aread + 1];

        while (b < e)
        {
            OgEdge* e = right + b;
            int bread = e->b;

            if ( (status[bread] & STATUS_PROPER) )
            {
                status[bread] |= STATUS_USED;
                used = 1;

                noedges[aread]++;
                niedges[bread]++;
            }

            b++;
        }

        if (used)
        {
            status[aread] |= STATUS_USED;
        }
    }

    for ( aread = 0; aread < octx->db->nreads; aread++ )
    {
        if ( !(status[aread] & STATUS_USED) )
        {
            continue;
        }

        int optional = (status[aread] & STATUS_OPTIONAL) ? 1 : 0;

        fprintf(f, "%d %d %d %d\n", aread, optional, niedges[aread], noedges[aread]);

        status[aread] ^= STATUS_USED;
    }

    fprintf(f, "#\n");

    for ( aread = 0; aread < octx->db->nreads; aread++ )
    {
        if ( !(status[aread] & STATUS_PROPER) )
        {
            continue;
        }

        if ( component != -1 && octx->comp[aread] != component )
        {
            continue;
        }

        uint64 b = nleft[aread];
        uint64 e = nleft[aread + 1];

        while (b < e)
        {
            OgEdge* e = left + b;

            if ( (status[e->b] & STATUS_PROPER) )
            {
                print_graph_tgf_edge(f, e, 'l');
            }

            b++;
        }

        b = nright[aread];
        e = nright[aread + 1];

        while (b < e)
        {
            OgEdge* e = right + b;

            if ( (status[e->b] & STATUS_PROPER) )
            {
                print_graph_tgf_edge(f, e, 'r');
            }

            b++;
        }
    }

    free(niedges);
    free(noedges);
}


static void print_graph_gml(OgBuildContext* octx, FILE* f, const char* title, char** comments, int ncomments, int component)
{
    fprintf(f, "graph [\n");

    int i;
    for ( i = 0; i < ncomments; i++ )
    {
        fprintf(f, "  comment \"%s\"\n", comments[i]);
    }

    if (title)
    {
        fprintf(f, "  label \"%s\"\n", title);
    }

    fprintf(f, "  directed 1\n");

    uint64* nleft = octx->nleft;
    uint64* nright = octx->nright;
    OgEdge* left = octx->left;
    OgEdge* right = octx->right;

    unsigned char* status = octx->status;
    int aread;
    for ( aread = 0; aread < octx->db->nreads; aread++ )
    {
        if ( !(status[aread] & STATUS_PROPER) )
        {
            continue;
        }

        if ( component != -1 && octx->comp[aread] != component )
        {
            continue;
        }

        int used = 0;
        uint64 b = nleft[aread];
        uint64 e = nleft[aread + 1];

        while (b < e)
        {
            OgEdge* e = left + b;

            if ( (status[e->b] & STATUS_PROPER) )
            {
                status[ e->b ] |= STATUS_USED;
                print_graph_gml_edge(f, e, 'l');
                used = 1;
            }

            b++;
        }

        b = nright[aread];
        e = nright[aread + 1];

        while (b < e)
        {
            OgEdge* e = right + b;

            if ( (status[e->b] & STATUS_PROPER) )
            {
                status[ e->b ] |= STATUS_USED;
                print_graph_gml_edge(f, e, 'r');
                used = 1;
            }

            b++;
        }

        if (used)
        {
            status[aread] |= STATUS_USED;
        }
    }

    for ( aread = 0; aread < octx->db->nreads; aread++ )
    {
        if ( !(status[aread] & STATUS_USED) )
        {
            continue;
        }

        int optional = (status[aread] & STATUS_OPTIONAL) ? 1 : 0;

        fprintf(f, "  node [\n");
        fprintf(f, "    id %d\n", aread);
        fprintf(f, "    read %d\n", aread);
        fprintf(f, "    optional %d\n", optional);
        fprintf(f, "  ]\n");

        status[aread] ^= STATUS_USED;
    }

    fprintf(f, "]\n");
}


static void print_graph_graphml(OgBuildContext* octx, FILE* f, const char* title, char** comments, int ncomments, int component)
{
    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n");

    if (ncomments)
    {
        fprintf(f, "<!--\n");

        int i;
        for ( i = 0; i < ncomments; i++ )
        {
            fprintf(f, "  %s\n", comments[i]);
        }

        fprintf(f, "-->\n");
    }

    fprintf(f, "<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\"\n");
    fprintf(f, "         xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");
    fprintf(f, "         xsi:schemaLocation=\"http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd\">\n");

    fprintf(f, "  <key attr.name=\"length\"     attr.type=\"int\"    for=\"edge\" id=\"length\" />\n");
    fprintf(f, "  <key attr.name=\"flags\"      attr.type=\"int\"    for=\"edge\" id=\"flags\" />\n");
    fprintf(f, "  <key attr.name=\"end\"        attr.type=\"string\" for=\"edge\" id=\"end\" />\n");
    fprintf(f, "  <key attr.name=\"divergence\" attr.type=\"int\"    for=\"edge\" id=\"divergence\" />\n");

    fprintf(f, "  <key attr.name=\"read\"       attr.type=\"int\"    for=\"node\" id=\"read\" />\n");
    fprintf(f, "  <key attr.name=\"optional\"   attr.type=\"int\"    for=\"node\" id=\"optional\" />\n");

    fprintf(f, "  <graph id=\"%s\" edgedefault=\"directed\">\n", title);

    uint64* nleft = octx->nleft;
    uint64* nright = octx->nright;
    OgEdge* left = octx->left;
    OgEdge* right = octx->right;

    unsigned char* status = octx->status;
    int aread;
    for ( aread = 0; aread < octx->db->nreads; aread++ )
    {
        if ( !(status[aread] & STATUS_PROPER) )
        {
            continue;
        }

        if ( component != -1 && octx->comp[aread] != component )
        {
            continue;
        }

        int used = 0;
        uint64 b = nleft[aread];
        uint64 e = nleft[aread + 1];

        while (b < e)
        {
            OgEdge* e = left + b;

            if ( (status[e->b] & STATUS_PROPER) )
            {
                status[ e->b ] |= STATUS_USED;
                print_graph_graphml_edge(f, e, 'l');
                used = 1;
            }

            b++;
        }

        b = nright[aread];
        e = nright[aread + 1];

        while (b < e)
        {
            OgEdge* e = right + b;

            if ( (status[e->b] & STATUS_PROPER) )
            {
                status[ e->b ] |= STATUS_USED;
                print_graph_graphml_edge(f, e, 'r');
                used = 1;
            }

            b++;
        }

        if (used)
        {
            status[aread] |= STATUS_USED;
        }
    }

    for ( aread = 0; aread < octx->db->nreads; aread++ )
    {
        if ( !(status[aread] & STATUS_USED) )
        {
            continue;
        }

        int optional = (status[aread] & STATUS_OPTIONAL) ? 1 : 0;

        fprintf(f, "    <node id=\"%d\">\n", aread);
        fprintf(f, "      <data key=\"read\">%d</data>\n", aread);
        fprintf(f, "      <data key=\"optional\">%d</data>\n", optional);
        fprintf(f, "    </node>\n");

        status[aread] ^= STATUS_USED;
    }

    fprintf(f, "  </graph>\n");
    fprintf(f, "</graphml>\n");
}

// assign reads to components

static void assign_component(OgBuildContext* octx)
{
    int nreads = octx->db->nreads;
    int* comp = octx->comp;
    unsigned char* status = octx->status;

    int i;
    for (i = 0; i < nreads; i++)
    {
        comp[i] = -1;
    }

    int maxstack = 1000;

    int curstack = 0;
    int curstack_new = 0;

    int* stack = malloc( sizeof(int) * maxstack );
    int* stack_new = malloc( sizeof(int) * maxstack );

    int curcomp = 0;
    OgEdge* left = octx->left;
    OgEdge* right = octx->right;
    uint64* nleft = octx->nleft;
    uint64* nright = octx->nright;

    // for all reads

    int j;
    for (j = 0; j < nreads; j++)
    {
        if ( comp[j] != -1 )
        {
            continue;
        }

        if ( !(status[j] & STATUS_PROPER) )
        {
            continue;
        }

        // initialize stack with read

        stack[0] = j;
        curstack = 1;
        comp[j] = curcomp;

        int empty = 1;

        // keep going until stack exhausted

        while (curstack)
        {
            for (i = 0; i < curstack; i++)
            {
                // get element from stack

                int rid = stack[i];

                uint64 b = nleft[ rid ];
                uint64 e = nleft[ rid + 1 ];

                // push reads to the left onto the new stack

                while (b < e)
                {
                    OgEdge* edge = left + b;

                    if ( comp[ edge->b ] == -1 && (status[edge->b] & STATUS_PROPER) )
                    {
                        comp[ edge->b ] = curcomp;

                        stack_new[curstack_new] = edge->b;
                        curstack_new++;

                        if (curstack_new == maxstack)
                        {
                            maxstack += 1000;
                            stack = realloc(stack, sizeof(int) * maxstack);
                            stack_new = realloc(stack_new, sizeof(int) * maxstack);
                        }
                    }

                    b++;
                }

                b = nright[ rid ];
                e = nright[ rid + 1 ];

                // push reads to the right onto the new stack

                while (b < e)
                {
                    OgEdge* edge = right + b;

                    if ( comp[ edge->b ] == -1 && (status[edge->b] & STATUS_PROPER) )
                    {
                        comp[ edge->b ] = curcomp;

                        stack_new[curstack_new] = edge->b;
                        curstack_new++;

                        if (curstack_new == maxstack)
                        {
                            maxstack += 1000;
                            stack = realloc(stack, sizeof(int) * maxstack);
                            stack_new = realloc(stack_new, sizeof(int) * maxstack);
                        }
                    }

                    b++;
                }
            }

            // were we able to go somewhere (widowed nodes)

            if (curstack_new > 0)
            {
                empty = 0;
            }

            // swap stacks

            curstack = curstack_new;

            curstack_new = 0;
            int* temp = stack;
            stack = stack_new;
            stack_new = temp;
        }

        // dead node

        if (empty)
        {
            comp[j] = -1;
        }
        else
        {
            curcomp++;
        }
    }

    printf("  %d components\n", curcomp);

    octx->ncomp = curcomp;

    free(stack);
    free(stack_new);
}

// sort OgEdge by .a and .b

static int cmp_ogedge(const void* a, const void* b)
{
    OgEdge* x = (OgEdge*)a;
    OgEdge* y = (OgEdge*)b;

    int cmp = x->a - y->a;

    if (cmp == 0)
    {
        cmp = x->b - y->b;
    }

    return cmp;
}

static int cmp_ogedge_ovh_rev(const void* a, const void* b)
{
    OgEdge* x = (OgEdge*)a;
    OgEdge* y = (OgEdge*)b;

    return y->ovh - x->ovh;
}

static int cmp_ogedge_ovl_rev(const void* a, const void* b)
{
    OgEdge* x = (OgEdge*)a;
    OgEdge* y = (OgEdge*)b;

    int xlen = x->ae - x->ab;
    int ylen = y->ae - y->ab;

    return ylen - xlen;
}

static int cmp_int(const void* a, const void* b)
{
    int* x = (int*)a;
    int* y = (int*)b;

    return (*x) - (*y);
}

static int cmp_2int(const void* a, const void* b)
{
    int* x = (int*)a;
    int* y = (int*)b;

    if ( x[0] == y[0] )
    {
        return x[1] - y[1];
    }

    return x[0] - y[0];
}

static int remove_dupes(OgEdge* e, int n)
{
    int dropped = 0;
    OgEdge* prev = e;

    int i;
    for ( i = 1; i < n; i++ )
    {
        OgEdge* cur = e + i;

        if (cur->b == prev->b)
        {
            prev->a = -1;
            prev->b = -1;

            dropped++;
        }

        prev = cur;
    }

    return dropped;
}

static int remove_lr_dupes(OgEdge* l, int nl, OgEdge* r, int nr)
{
    int dropped = 0;

    int i;
    for ( i = 0 ; i < nl ; i++)
    {
        OgEdge* le = l + i;

        int j;
        for ( j = 0 ; j < nr ; j++)
        {
            OgEdge* re = r + j;

            if (le->b == re->b)
            {
                re->a = -1;
                re->b = -1;

                dropped += 1;
            }
        }
    }

    return dropped;
}

static void remove_parallel_edges(OgBuildContext* octx)
{
    int64 dropped = 0;
    int nreads = DB_NREADS(octx->db);

    printf("parallel edges\n");

    int rid;
    for ( rid = 0; rid < nreads; rid++)
    {
        uint64 lb = octx->nleft[rid];
        uint64 le = octx->nleft[rid + 1];

        if ( lb < le )
        {
            dropped += remove_dupes(octx->left + lb, le - lb);
        }

        uint64 rb = octx->nright[rid];
        uint64 re = octx->nright[rid + 1];

        if ( rb < re )
        {
            dropped += remove_dupes(octx->right + rb, re - rb);
        }

        if ( lb < le && rb < re )
        {
            dropped += remove_lr_dupes(octx->left + lb, le - lb, octx->right + rb, re - rb);
        }
    }

    printf("  %'lld parallel edges\n", dropped);
}

static int compress_graph_side(OgBuildContext* octx, OgEdge* edges, uint64* n)
{
    int adj = 0;
    int nreads = octx->db->nreads;

    int rid;
    for ( rid = 0 ; rid < nreads ; rid++ )
    {
        uint64 b = n[rid];
        uint64 e = n[rid + 1];

        n[rid] -= adj;

        uint64 i;
        for ( i = b ; i < e ; i++ )
        {
            OgEdge* edge = edges + i;

            if (edge->a == -1 || edge->b == -1)
            {
                adj++;
            }
            else
            {
                edges[i - adj] = edges[i];
            }
        }
    }

    n[rid] -= adj;

    return adj;
}

static void compress_graph(OgBuildContext* octx)
{
    int dropped = 0;
    int nreads = octx->db->nreads;

    dropped += compress_graph_side(octx, octx->left, octx->nleft);

    octx->left = realloc(octx->left, sizeof(OgEdge) * octx->nleft[nreads]);

    dropped += compress_graph_side(octx, octx->right, octx->nright);

    octx->right = realloc(octx->right, sizeof(OgEdge) * octx->nright[nreads]);

    if (dropped > 0)
    {
        printf("removed %d edges\n", dropped);
    }
}

static void sort_edges(OgBuildContext* octx)
{
    printf("sorting edges\n");

    int rid;
    for ( rid = 0; rid < octx->db->nreads; rid++)
    {
        uint64 b = octx->nleft[rid];
        uint64 e = octx->nleft[rid + 1];

        qsort(octx->left + b, e - b, sizeof(OgEdge), cmp_ogedge);

        b = octx->nright[rid];
        e = octx->nright[rid + 1];

        qsort(octx->right + b, e - b, sizeof(OgEdge), cmp_ogedge);
    }
}

static int proper_edges(OgBuildContext* octx, OgEdge* edge, int n)
{
    unsigned char* status = octx->status;
    int cnt = 0;

    int i;
    for ( i = 0; i < n; i++ )
    {
        OgEdge* e = edge + i;

        // DOUBLE-CHECK - OPTIONAL edges
        if (e->flags & OVL_OPTIONAL)
        {
            continue;
        }

        assert( e->b != -1 );

        if ( ( status[ e->b ] & STATUS_PROPER ) )
        {
            cnt++;
        }
    }

    return cnt;
}

static void add_contained_edges(OgBuildContext* octx, int contained)
{
    printf("adding contained edges (c = %d)\n", contained);

    if (contained == -1)
    {
        contained = INT_MAX;
    }

    unsigned char* status = octx->status;
    int nreads = octx->db->nreads;
    OgEdge* left = octx->left;
    OgEdge* right = octx->right;
    uint64* nleft = octx->nleft;
    uint64* nright = octx->nright;

    int curinspect = 0;
    int maxinspect = nreads;
    int* inspect = malloc(sizeof(int) * maxinspect);
    int* inspect_new = malloc(sizeof(int) * maxinspect);

    int i;
    for (i = 0; i < nreads; i++)
    {
        if (status[i] != STATUS_PROPER)
        {
            continue;
        }

        inspect[ curinspect ] = i;
        curinspect++;
    }

    printf("  inspecting %'d proper reads\n", curinspect);

    int round = 0;
    while (curinspect > 0)
    {

        int curinspect_new = 0;

        for ( i = 0; i < curinspect; i++)
        {
            int aread = inspect[i];

            if (status[aread] != STATUS_PROPER)
            {
                continue;
            }

            uint64 b = nleft[aread];
            uint64 e = nleft[aread + 1];

            int left_nc = proper_edges(octx, left + b, e - b);

            if (left_nc == 0)
            {
                if (octx->edgeprio == EDGE_SORT_OVH)
                {
                    qsort(left + b, e - b, sizeof(OgEdge), cmp_ogedge_ovh_rev);
                }
                else if (octx->edgeprio == EDGE_SORT_OVL)
                {
                    qsort(left + b, e - b, sizeof(OgEdge), cmp_ogedge_ovl_rev);
                }
                else
                {
                    fprintf(stderr, "unknown edge priority mode %d\n", octx->edgeprio);
                    exit(1);
                }

                uint64 j;
                int n;
                for ( j = b, n = 0 ; j < e && n < contained ; j++ )
                {
                    int bread = left[j].b;

                    // DOUBLE-CHECK - optional edges
                    if (left[j].flags & OVL_OPTIONAL)
                    {
                        continue;
                    }

#ifdef DEBUG_INSPECT
                    printf("INSPECT L %7d / %7d -> %7d OVH %7d %d\n", left[j].a, aread, bread, left[j].ovh, e - b);
#endif

                    status[bread] = STATUS_PROPER;
                    status[bread] |= STATUS_OPTIONAL;

                    inspect_new[curinspect_new] = bread;
                    curinspect_new++;

                    if (curinspect_new == maxinspect)
                    {
                        maxinspect = 1.2 * maxinspect + 1000;

                        inspect_new = realloc(inspect_new, sizeof(int) * maxinspect);
                        inspect = realloc(inspect, sizeof(int) * maxinspect);
                    }

                    n++;
                }
            }

            b = nright[aread];
            e = nright[aread + 1];

            int right_nc = proper_edges(octx, right + b, e - b);

            if (right_nc == 0)
            {
                if (octx->edgeprio == EDGE_SORT_OVH)
                {
                    qsort(right + b, e - b, sizeof(OgEdge), cmp_ogedge_ovh_rev);
                }
                else if (octx->edgeprio == EDGE_SORT_OVL)
                {
                    qsort(right + b, e - b, sizeof(OgEdge), cmp_ogedge_ovl_rev);
                }
                else
                {
                    fprintf(stderr, "unknown edge priority mode %d\n", octx->edgeprio);
                    exit(1);
                }

                uint64 j;
                int n;
                for ( j = b, n = 0 ; j < e && n < contained ; j++ )
                {
                    int bread = right[j].b;

                    // DOUBLE-CHECK - optional edges
                    if (right[j].flags & OVL_OPTIONAL)
                    {
                        continue;
                    }

#ifdef DEBUG_INSPECT
                    printf("INSPECT R %7d / %7d -> %7d OVH %7d %d\n", right[j].a, aread, bread, right[j].ovh, e - b);
#endif

                    status[bread] = STATUS_PROPER;
                    status[bread] |= STATUS_OPTIONAL;

                    inspect_new[curinspect_new] = bread;
                    curinspect_new++;

                    if (curinspect_new == maxinspect)
                    {
                        maxinspect = 1.2 * maxinspect + 1000;

                        inspect_new = realloc(inspect_new, sizeof(int) * maxinspect);
                        inspect = realloc(inspect, sizeof(int) * maxinspect);
                    }

                    n++;
                }
            }
        }

        qsort(inspect_new, curinspect_new, sizeof(int), cmp_int);

        curinspect = curinspect_new;
        int* temp = inspect;
        inspect = inspect_new;
        inspect_new = temp;

        round++;

        if (curinspect > 0)
        {
            printf("  set %d reads to proper\n", curinspect);
        }
    }

    free(inspect);
    free(inspect_new);
}

/*
static int reduce_edges(OgBuildContext* octx)
{
    unsigned char* status = octx->status;
    int nreads = octx->db->nreads;
    int dropped = 0;

    printf("left edges %llu\nright edges %llu\n", octx->nleft[nreads], octx->nright[nreads]);

    uint64 i;
    for ( i = 0; i < octx->nleft[nreads]; i++)
    {
        OgEdge* e = octx->left + i;

        assert( e->a >= 0 && e->b >= 0 );

        if ( status[e->b] == STATUS_CONTAINED )
        {
            e->a = -1;
            e->b = -1;

            dropped++;
        }
    }

    for ( i = 0; i < octx->nright[nreads]; i++)
    {
        OgEdge* e = octx->right + i;

        assert( e->a >= 0 && e->b >= 0 );

        if ( status[e->b] == STATUS_CONTAINED )
        {
            e->a = -1;
            e->b = -1;

            dropped++;
        }
    }

    return dropped;
}
*/

static void write_graph(OgBuildContext* octx, const char* path)
{
    if (octx->split)
    {
        char* pathcomp = malloc( strlen(path) + 30 );

        int i;
        for (i = 0; i < octx->ncomp; i++)
        {
            char* ext;
            if (octx->gformat == FORMAT_GML)
            {
                ext = "gml";
            }
            else if (octx->gformat == FORMAT_TGF)
            {
                ext = "tgf";
            }
            else
            {
                ext = "graphml";
            }

            sprintf(pathcomp, "%s_%05d.%s", path, i, ext);

            FILE* f = fopen(pathcomp, "w");

            if (f)
            {
                if (octx->gformat == FORMAT_GML)
                {
                    print_graph_gml(octx, f, "og", NULL, 0, i);
                }
                else if (octx->gformat == FORMAT_TGF)
                {
                    print_graph_tgf(octx, f, i);
                }
                else
                {
                    print_graph_graphml(octx, f, "og", NULL, 0, i);
                }

                fclose(f);
            }
            else
            {
                fprintf(stderr, "failed to create %s\n", pathcomp);
            }
        }

        free(pathcomp);
    }
    else
    {
        FILE* f = fopen(path, "w");

        if (f)
        {
            if (octx->gformat == FORMAT_GML)
            {
                print_graph_gml(octx, f, "og", NULL, 0, -1);
            }
            else if (octx->gformat == FORMAT_TGF)
            {
                print_graph_tgf(octx, f, -1);
            }
            else
            {
                print_graph_graphml(octx, f, "og", NULL, 0, -1);
            }

            fclose(f);
        }
        else
        {
            fprintf(stderr, "failed to create %s\n", path);
        }
    }
}

static void post_build(OgBuildContext* octx)
{
    printf("  %'llu edges\n", octx->stats_edges);
    printf("  %'llu redges\n", octx->stats_redges);
    printf("  %'llu symmetric discards\n", octx->stats_symdiscard);

    sort_edges(octx);

    remove_parallel_edges(octx);

    compress_graph(octx);

    if (octx->contained != 0)
    {
        add_contained_edges(octx, octx->contained);
    }

    compress_graph(octx);

    if (octx->split)
    {
        printf("components\n");
        assign_component(octx);
    }

    write_graph(octx, octx->path_graph);

    free(octx->nleft);
    free(octx->nright);

    free(octx->left);
    free(octx->right);

    free(octx->curleft);
    free(octx->curright);

    free(octx->status);

    free(octx->comp);
}

static void drop_parallel_edges(Overlap* ovls, int novl)
{
    int i = 0;
    while (i < novl - 1)
    {
        if ( ovls[i].flags & OVL_DISCARD )
        {
            i += 1;

            continue ;
        }

        int j = i + 1;
        if ( ovls[i].bread == ovls[j].bread )
        {
            int dropped = 0;

            for ( ; j < novl && ovls[i].bread == ovls[j].bread ; j++ )
            {
                if ( ovls[j].flags & OVL_DISCARD )
                {
                    continue;
                }

                ovls[j].flags |= OVL_DISCARD;
                dropped += 1;
            }

            if (dropped > 0)
            {
                ovls[i].flags |= OVL_DISCARD;
                dropped += 1;
            }

            // ctx->filtered_parallel_edges += dropped;
        }

        i = j;
    }
}

static void assign_edge(OgEdge* edge, Overlap* ovl)
{
    edge->a = ovl->aread;
    edge->b = ovl->bread;

    edge->flags = ovl->flags;
    edge->diffs = ovl->path.diffs;
    edge->ovh = -1;

    edge->ab = ovl->path.abpos;
    edge->ae = ovl->path.aepos;
    edge->bb = ovl->path.bbpos;
    edge->be = ovl->path.bepos;
}

static void assign_edge_reversed(OgEdge* edge, Overlap* ovl, int alen, int blen)
{
    edge->a = ovl->bread;
    edge->b = ovl->aread;

    edge->flags = ovl->flags;
    edge->diffs = ovl->path.diffs;
    edge->ovh = -1;

    if (ovl->flags & OVL_COMP)
    {
        edge->ab = blen - ovl->path.bepos;
        edge->ae = blen - ovl->path.bbpos;

        edge->bb = alen - ovl->path.aepos;
        edge->be = alen - ovl->path.abpos;
    }
    else
    {
        edge->ab = ovl->path.bbpos;
        edge->ae = ovl->path.bepos;

        edge->bb = ovl->path.abpos;
        edge->be = ovl->path.aepos;
    }
}

static int handler_build(void* _ctx, Overlap* ovls, int novl)
{
    OgBuildContext* octx = (OgBuildContext*)_ctx;
    unsigned char* status = octx->status;

    uint64 edges = 0;
    uint64 redges = 0;
    uint64 symdiscard = 0;

    uint64* nleft = octx->nleft;
    OgEdge* left = octx->left;
    int* curleft = octx->curleft;

    uint64* nright = octx->nright;
    OgEdge* right = octx->right;
    int* curright = octx->curright;

    int aread = ovls->aread;
    int trim_ab, trim_ae;
    int alen = DB_READ_LEN(octx->db, aread);

    drop_parallel_edges(ovls, novl);

    get_trim(octx->db, octx->trimtrack, aread, &trim_ab, &trim_ae);

    int i;
    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl = ovls + i;

        if (ovl->aread == ovl->bread)
        {
            continue;
        }

        int ab = ovl->path.abpos;
        int ae = ovl->path.aepos;

        if (ab == trim_ab && ae == trim_ae)
        {
            status[aread] = STATUS_CONTAINED;
            continue;
        }

        int bread = ovl->bread;
        int blen = DB_READ_LEN(octx->db, bread);
        int trim_bb, trim_be;

        get_trim(octx->db, octx->trimtrack, bread, &trim_bb, &trim_be);

        if ( ovl->flags & OVL_COMP )
        {
            int t = trim_bb;
            trim_bb = blen - trim_be;
            trim_be = blen - t;
        }

        int bb = ovl->path.bbpos;
        int be = ovl->path.bepos;

        if ( bb == trim_bb && be == trim_be )
        {
            status[bread] = STATUS_CONTAINED;
            continue;
        }

        if ( (ovl->flags & OVL_DISCARD) || (ovl->flags & OVL_SYMDISCARD) )
        {
            continue;
        }

        int j = octx->db->reads[aread].flags;

        if ( j != -1 )
        {
            int discard = 0;
            for ( j = 0; j < octx->ndiscard; j += 2)
            {
                int a_disc = octx->discard[j];
                int b_disc = octx->discard[j + 1];

                if ( a_disc > aread )
                {
                    break;
                }

                if ( bread == b_disc )
                {
                    discard = 1;
                    break;
                }
            }

            if (discard)
            {
                // printf("sym-discard %d -> %d\n", aread, bread);
                symdiscard += 1;

                continue;
            }
        }

        if ( ab == trim_ab )
        {
            int ovh = bb - trim_bb;

            if (ovh <= 0)
            {
                if (ovh < 0)
                {
                    fprintf(stderr, "error: ovh %5d <= 0 %7d -> %7d. Trim track most likely incompatible with overlaps.\n", ovh, aread, bread);
                    exit(1);
                }
            }
            else
            {
                OgEdge* edge = left + nleft[aread] + curleft[aread];
                assign_edge(edge, ovl);
                edge->ovh = ovh;
                curleft[aread]++;

                edges++;

                if ( ae < trim_ae )
                {
                    if ( ovl->flags & OVL_COMP )
                    {
                        edge = left + nleft[bread] + curleft[bread];
                        curleft[bread]++;

                    }
                    else
                    {
                        edge = right + nright[bread] + curright[bread];
                        curright[bread]++;
                    }

                    if (status[bread] == STATUS_WIDOW)
                    {
                        status[bread] = STATUS_PROPER;
                    }

                    assign_edge_reversed(edge, ovl, alen, blen);
                    edge->ovh = trim_ae - ae;
                    redges++;
                }
            }
        }

        if ( ae == trim_ae )
        {
            int ovh = trim_be - be;

            if ( ovh <= 0 )
            {
                if (ovh < 0)
                {
                    fprintf(stderr, "error: ovh %5d <= 0 %7d -> %7d. Trim track most likely incompatible with overlaps.\n", ovh, aread, bread);
                    exit(1);
                }
            }
            else
            {
                OgEdge* edge = right + nright[aread] + curright[aread];
                assign_edge(edge, ovl);
                edge->ovh = ovh;
                curright[aread]++;

                edges++;

                if ( ab > trim_ab )
                {
                    if ( ovl->flags & OVL_COMP )
                    {
                        edge = right + nright[bread] + curright[bread];
                        curright[bread]++;
                    }
                    else
                    {
                        edge = left + nleft[bread] + curleft[bread];
                        curleft[bread]++;
                    }

                    if (status[bread] == STATUS_WIDOW)
                    {
                        status[bread] = STATUS_PROPER;
                    }

                    assign_edge_reversed(edge, ovl, alen, blen);
                    edge->ovh = ab - trim_ab;
                    redges++;
                }
            }
        }
    }

    if ( status[aread] == STATUS_WIDOW && edges > 0 )
    {
        status[aread] = STATUS_PROPER;
    }

    octx->stats_symdiscard += symdiscard;
    octx->stats_edges += edges;
    octx->stats_redges += redges;

    return 1;
}

// initialize data structures for pass(es)

static void pre_build(PassContext* pctx, OgBuildContext* octx)
{
    UNUSED(pctx);

    HITS_DB* db = octx->db;
    int nreads = db->nreads;

    octx->trimtrack = track_load(db, octx->trimName);

    if (!octx->trimtrack)
    {
        fprintf(stderr, "ERROR: failed to open %s\n", octx->trimName);
        exit(1);
    }

    octx->status = malloc( nreads );

    int i;
    for ( i = 0; i < nreads; i++)
    {
        octx->status[i] = STATUS_WIDOW;
    }

    octx->nleft = calloc( nreads + 1, sizeof(uint64) );
    octx->nright = calloc( nreads + 1, sizeof(uint64) );

    octx->curleft = calloc( nreads, sizeof(int) );
    octx->curright = calloc( nreads, sizeof(int) );

    octx->comp = malloc( sizeof(int) * nreads );
}

static int handler_count(void* _ctx, Overlap* ovls, int novl)
{
    OgBuildContext* octx = (OgBuildContext*)_ctx;
    uint64* nleft = octx->nleft;
    uint64* nright = octx->nright;

    int aread = ovls->aread;
    int trim_ab, trim_ae;
    // int alen = DB_READ_LEN(octx->db, aread);

    drop_parallel_edges(ovls, novl);

    get_trim(octx->db, octx->trimtrack, aread, &trim_ab, &trim_ae);

    int i;
    for ( i = 0; i < novl; i++ )
    {
        Overlap* ovl = ovls + i;

        if (ovl->aread == ovl->bread)
        {
            continue;
        }

        if ( ovl->flags & OVL_SYMDISCARD )
        {
            if (octx->ndiscard + 2 >= octx->maxdiscard)
            {
                octx->maxdiscard = octx->maxdiscard * 1.2 + 100;
                octx->discard = realloc(octx->discard, sizeof(int) * octx->maxdiscard);
            }

            octx->discard[ octx->ndiscard ] = ovl->bread;
            octx->discard[ octx->ndiscard + 1 ] = ovl->aread;

            octx->ndiscard += 2;
        }

        if ( (ovl->flags & OVL_DISCARD) )
        {
            continue;
        }

        int ab = ovl->path.abpos;
        int ae = ovl->path.aepos;

        if (ab == trim_ab && ae == trim_ae)
        {
            continue;
        }

        int bread = ovl->bread;
        int blen = DB_READ_LEN(octx->db, bread);
        int trim_bb, trim_be;

        get_trim(octx->db, octx->trimtrack, bread, &trim_bb, &trim_be);

        if ( ovl->flags & OVL_COMP )
        {
            int t = trim_bb;
            trim_bb = blen - trim_be;
            trim_be = blen - t;
        }

        int bb = ovl->path.bbpos;
        int be = ovl->path.bepos;

        if ( bb == trim_bb && be == trim_be )
        {
            continue;
        }

        if ( ab == trim_ab )
        {
            int ovh = bb - trim_bb;

            if (ovh <= 0)
            {
                if (ovh < 0)
                {
                    fprintf(stderr, "error: ovh %5d <= 0 %7d -> %7d. Trim track most likely incompatible with overlaps.\n", ovh, aread, bread);
                }
            }
            else
            {
                nleft[aread]++;

                if ( ae < trim_ae )
                {
                    if ( ovl->flags & OVL_COMP )
                    {
                        nleft[bread]++;
                    }
                    else
                    {
                        nright[bread]++;
                    }
                }
            }
        }

        if ( ae == trim_ae )
        {
            int ovh = trim_be - be;

            if ( ovh <= 0 )
            {
                if (ovh < 0)
                {
                    fprintf(stderr, "error: ovh %5d <= 0 %7d -> %7d. Trim track most likely incompatible with overlaps.\n", ovh, aread, bread);
                }
            }
            else
            {
                nright[aread]++;

                if ( ab > trim_ab )
                {
                    if ( ovl->flags & OVL_COMP )
                    {
                        nright[bread]++;
                    }
                    else
                    {
                        nleft[bread]++;
                    }
                }
            }
        }
    }

    return 1;
}

static uint64 to_offsets(uint64* counts, int n)
{
    uint64 off = 0;

    int i;
    for (i = 0; i < n; i++)
    {
        uint64 coff = counts[i];
        counts[i] = off;

        off += coff;
    }

    counts[i] = off;

    return counts[i];
}

static void post_count(OgBuildContext* octx)
{
    HITS_DB* db = octx->db;

    qsort(octx->discard, octx->ndiscard / 2, 2 * sizeof(int), cmp_2int);

    int i;

    for ( i = 0; i < DB_NREADS(octx->db); i++ )
    {
        octx->db->reads[i].flags = -1;
    }

    for ( i = 0; i < octx->ndiscard; i += 2 )
    {
        int rid = octx->discard[i];

        octx->db->reads[ rid ].flags = i;
    }

    int needed;

    needed = to_offsets(octx->nleft, db->nreads);
    octx->left = calloc( needed, sizeof(OgEdge) );

    printf("%d left edges\n", needed);

    needed = to_offsets(octx->nright, db->nreads);
    octx->right = calloc( needed, sizeof(OgEdge) );

    printf("%d right edges\n", needed);
}

static void usage()
{
    printf( "usage: [-s] [-c <int>] [-t <track>] [-f gml|graphml|tgf] [-p ovl|ovh] database input.las output.format\n\n" );

    printf( "Builds the overlap graph based on the alignments in the input las file.\n\n" );

    printf( "options: -c n      try to add otherwise containted reads to dead ends in the graph\n" );
    printf( "                   0 disabled, >0 maximum number of edges, -1 all edges (default %d)\n", DEF_ARG_C );
    printf( "         -p mode   which edges should be added when running in -c mode (default %s)\n" , DEF_ARG_P);
    printf( "                   ovl longest overlap, ovh longer overhang\n" );

    printf( "         -f frmt   output graph format. gml, graphml or tgf (default %s)\n", DEF_ARG_F );
    printf( "         -s        write on file for each component of the overlap graph.\n" );
    printf( "                   files are named output.<component.number>.format\n" );
    printf( "         -t track  which trim track to use (%s)\n", DEF_ARG_T );
}

int main(int argc, char* argv[])
{
    HITS_DB db;
    FILE* fileOvlIn;

    PassContext* pctx;
    OgBuildContext octx;

    bzero(&octx, sizeof(OgBuildContext));

    // process arguments

    char* edgeprio = DEF_ARG_P;
    char* gformat = DEF_ARG_F;
    octx.contained = DEF_ARG_C;
    octx.trimName = DEF_ARG_T;

    opterr = 0;

    int c;
    while ((c = getopt(argc, argv, "sc:f:p:t:")) != -1)
    {
        switch (c)
        {
            case 'p':
                      edgeprio = optarg;
                      break;

            case 'f':
                      gformat = optarg;
                      break;

            case 's':
                      octx.split = 1;
                      break;

            case 't':
                      octx.trimName = optarg;
                      break;

            case 'c':
                      octx.contained = atoi(optarg);
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
    char* pcPathOverlaps = argv[optind++];
    octx.path_graph = argv[optind++];

    if ( strcmp(gformat, "gml") == 0 )
    {
        octx.gformat = FORMAT_GML;
    }
    else if ( strcmp(gformat, "graphml") == 0 )
    {
        octx.gformat = FORMAT_GRAPHML;
    }
    else if ( strcmp(gformat, "tgf") == 0 )
    {
        octx.gformat = FORMAT_TGF;
    }
    else
    {
        fprintf(stderr, "error: unknown graph format %s\n", gformat);
        usage();
        exit(1);
    }

    if ( strcmp(edgeprio, "ovh") == 0 )
    {
        octx.edgeprio = EDGE_SORT_OVH;
    }
    else if ( strcmp(edgeprio, "ovl") == 0 )
    {
        octx.edgeprio = EDGE_SORT_OVL;
    }
    else
    {
        fprintf(stderr, "error: unknown edge priority %s\n", edgeprio);
        usage();
        exit(1);
    }


    if ( (fileOvlIn = fopen(pcPathOverlaps, "r")) == NULL )
    {
        fprintf(stderr, "could not open '%s'\n", pcPathOverlaps);
        exit(1);
    }

    if (Open_DB(pcPathReadsIn, &db))
    {
        fprintf(stderr, "could not open '%s'\n", pcPathReadsIn);
        exit(1);
    }

    if (octx.contained < -1)
    {
        fprintf(stderr, "unvalid value %d for -c\n", octx.contained);
        exit(1);
    }

    // init

    octx.db = &db;

    pctx = pass_init(fileOvlIn, NULL);

    pctx->split_b = 0;
    pctx->load_trace = 0;
    pctx->progress = 1;
    pctx->data = &octx;

    // pass

    pre_build(pctx, &octx);

    printf(ANSI_COLOR_GREEN "PASS - calculating memory requirements" ANSI_COLOR_RESET "\n");

    pass(pctx, handler_count);

    post_count(&octx);

    printf(ANSI_COLOR_GREEN "PASS - building graph" ANSI_COLOR_RESET "\n");

    octx.stats_pedges_dropped = 0;

    pass(pctx, handler_build);

    printf(ANSI_COLOR_GREEN "processing graph" ANSI_COLOR_RESET "\n");

    post_build(&octx);

    // cleanup

    Close_DB(&db);

    pass_free(pctx);

    fclose(fileOvlIn);

    return 0;
}
