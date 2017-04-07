
/*******************************************************************************************
 *
 *  Tours the overlap graph
 *  (C implementation of touring_v2.py)
 *
 *  Date    : February 2016
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

// defaults


// switches

// used to store overlaps, basically a copy of everything we need from an Overlap struct

typedef struct
{
    int source, target;

    unsigned char end;

    int flags;
    int ovh;

    unsigned short div;
} OgEdge;

// maintains the state of the app

typedef struct
{
    HITS_DB* db;
    HITS_TRACK* trimtrack;

    // command line args

    char* path_graph_in;
    char* path_graph_out;

    // overlap graph

    uint64* nedges;
    OgEdge* edges;

} OgTourContext;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

static int read_graph_tgf(OgTourContext* octx)
{
    FILE* fileIn = fopen(octx->path_graph_in, "r");

    if (fileIn == NULL)
    {
        return 0;
    }

    char* line = NULL;
    size_t maxline = 0;

    int nedgestot = 0;
    int nreads = octx->db->nreads;

    OgEdge* edges;

    int* curedges = malloc(sizeof(int) * nreads);
    bzero(curedges, sizeof(int) * nreads);

    uint64* nedges = octx->nedges = malloc(sizeof(uint64) * (nreads + 1));
    bzero(nedges, sizeof(uint64) * (nreads + 1));

    int parsing_edges = 0;
    int nline = 0;
    int len;
    while ( (len = getline(&line, &maxline, fileIn)) > 0 )
    {
        nline++;

        if (line[0] == '#')
        {
            printf("%d edges\n", nedgestot);

            edges = octx->edges = malloc(sizeof(OgEdge) * nedgestot);
            bzero(edges, sizeof(OgEdge) * nedgestot);

            // convert counts to offsets
            int i;
            uint64 off = 0;
            for ( i = 0 ; i <= nreads ; i++ )
            {
                uint64 coff = nedges[i];
                nedges[i] = off;
                off += coff;
            }

            parsing_edges = 1;

            continue;
        }

        if (parsing_edges)
        {
            char end;
            int source, target, ovh, flags, div;

            if ( sscanf(line, "%d %d %d %d %d %c\n", &source, &target, &ovh, &flags, &div, &end) != 6 )
            {
                fprintf(stderr, "error: parsing tgf failed at line %d. '%s'\n", nline, line);
                exit(1);
            }

            OgEdge* e = edges + nedges[source] + curedges[source];

            e->source = source;
            e->target = target;
            e->ovh = ovh;
            e->flags = flags;
            e->div = div;
            e->end = end;

            curedges[source]++;

            e = edges + nedges[target] + curedges[target];

            e->source = source;
            e->target = target;
            e->ovh = ovh;
            e->flags = flags;
            e->div = div;
            e->end = end;

            curedges[target]++;
        }
        else
        {
            int rid, cont, edges_in, edges_out;

            if ( sscanf(line, "%d %d %d %d\n", &rid, &cont, &edges_in, &edges_out) != 4 )
            {
                fprintf(stderr, "error: parsing tgf failed at line %d. '%s'\n", nline, line);
                exit(1);
            }

            if (rid < 0 || rid >= nreads)
            {
                fprintf(stderr, "error: invalid read id %d\n", rid);
                exit(1);
            }

            nedges[rid] += edges_in + edges_out;
            nedgestot += edges_in + edges_out;
        }
    }

    free(curedges);

    fclose(fileIn);

    return 1;
}

static void usage()
{
    printf("OGtour <db> <overlap_graph.tgf>\n");
    printf("options:\n");
};

int main(int argc, char* argv[])
{
    HITS_DB db;
    OgTourContext octx;

    bzero(&octx, sizeof(OgTourContext));

    // process arguments

    opterr = 0;

    int c;
    while ((c = getopt(argc, argv, "")) != -1)
    {
        switch (c)
        {
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
    octx.path_graph_in = argv[optind++];

    if (Open_DB(pcPathReadsIn, &db))
    {
        fprintf(stderr, "could not open '%s'\n", pcPathReadsIn);
        exit(1);
    }

    // init

    octx.db = &db;

    // work

    printf("reading graph\n");

    if (!read_graph_tgf(&octx))
    {
        fprintf(stderr, "error: failed to read %s\n", octx.path_graph_in);
        exit(1);
    }

    // cleanup

    free(octx.nedges);
    free(octx.edges);

    Close_DB(&db);

    return 0;
}
