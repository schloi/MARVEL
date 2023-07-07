/*******************************************************************************************
 *
 * extract overlaps for specific a-read ids
 *
 *  Author :  MARVEL Team
 *
 *  Date   :  September 2014
 *
 *******************************************************************************************/

#include <string.h>
#include <stdlib.h>

#include "lib/colors.h"
#include "lib/pass.h"
#include "lib/oflags.h"

#include "db/DB.h"
#include "dalign/align.h"

#define VERBOSE


static void usage()
{
    printf("<db> <overlaps_in> <overlaps_out>\n");
}

typedef struct
{

    int* rid;
    int nrid;
    int currid;

} FilterContext;

static int filter_handler(void* _ctx, Overlap* ovl, int novl)
{
    FilterContext* ctx = (FilterContext*)_ctx;

    int* rid = ctx->rid;
    int nrid = ctx->nrid;
    int currid = ctx->currid;

    int a = ovl->aread;

    while ( currid < nrid && rid[currid] < a )
    {
        currid++;
    }

    if ( currid < nrid && rid[currid] != a )
    {
        int i;
        for ( i = 0 ; i < novl ; i++ )
        {
            ovl[i].flags |= OVL_DISCARD;
        }
    }
    else if ( rid[currid] == a )
    {
        currid++;
    }

    ctx->currid = currid;

    if (currid == nrid)
    {
        return 0;
    }

    return 1;
}

static int cmp_int(const void* x, const void* y)
{
    int* a = (int*)x;
    int* b = (int*)y;

    return (*a) - (*b);
}

int main(int argc, char* argv[])
{
    HITS_DB db;
    PassContext* pctx;
    FilterContext fctx;
    FILE* fileOvlIn;
    FILE* fileOvlOut;

    bzero(&fctx, sizeof(FilterContext));

    // args

    if (argc < 4)
    {
        usage();
        exit(1);
    }

    int* rid = malloc(sizeof(int) * argc);
    int nrid = 0;

    int i;
    for (i = 4; i < argc; i++)
    {
        char* end;
        int t = strtol(argv[i], &end, 10);

        if (*end == '\0')
        {
            rid[nrid] = t;
            nrid++;
        }
        else
        {
            fprintf(stderr, "invalid argument %s\n", argv[i]);
        }
    }

    if (nrid == 0)
    {
        fprintf(stderr, "nothing to do\n");
        exit(0);
    }

    qsort(rid, nrid, sizeof(int), cmp_int);

    fctx.nrid = nrid;
    fctx.rid = rid;


    char* pcPathReadsIn = argv[1];
    char* pcPathOverlapsIn = argv[2];
    char* pcPathOverlapsOut = argv[3];

    if ( (fileOvlIn = fopen(pcPathOverlapsIn, "r")) == NULL )
    {
        fprintf(stderr, "could not open %s\n", pcPathOverlapsIn);
        exit(1);
    }

    if ( (fileOvlOut = fopen(pcPathOverlapsOut, "w")) == NULL )
    {
        fprintf(stderr, "could not open %s\n", pcPathOverlapsOut);
        exit(1);
    }

    if (Open_DB(pcPathReadsIn, &db))
    {
        fprintf(stderr, "could not open %s\n", pcPathReadsIn);
        exit(1);
    }

     // passes

     pctx = pass_init(fileOvlIn, fileOvlOut);

     pctx->split_b = 0;
     pctx->load_trace = 1;
     pctx->unpack_trace = 0;
     pctx->data = &fctx;
     pctx->write_overlaps = 1;
     pctx->purge_discarded = 1;
     pctx->progress = 1;

#ifdef VERBOSE
    printf(ANSI_COLOR_GREEN "PASS extracting\n" ANSI_COLOR_RESET);
#endif

     pass(pctx, filter_handler);

     // cleanup

     pass_free(pctx);

     Close_DB(&db);

     free(rid);

     fclose(fileOvlOut);
     fclose(fileOvlIn);

     return 0;
}
