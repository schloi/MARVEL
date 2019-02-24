/*******************************************************************************************
 *
 * ... TODO
 *
 *******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <assert.h>
#include <unistd.h>

#include "lib/tracks.h"
#include "lib/pass.h"
#include "lib/oflags.h"
#include "lib/colors.h"
#include "lib/utils.h"
#include "lib/trim.h"

#include "db/DB.h"
#include "dalign/align.h"

// switches

#define VERBOSE

// constants

typedef struct
{
    HITS_DB* db;
    HITS_TRACK* trackExclude;

    // statistics

    uint64_t stats_discarded;

} DiscardContext;

// for getopt()

extern char* optarg;
extern int optind, opterr, optopt;

static void discard_pre(PassContext* pctx, DiscardContext* ctx)
{
#ifdef VERBOSE
    printf(ANSI_COLOR_GREEN "PASS discard" ANSI_COLOR_RESET "\n");
#endif

}

static void discard_post(DiscardContext* ctx)
{
#ifdef VERBOSE
    printf("dropped %" PRIu64 " overlaps\n", ctx->stats_discarded);
#endif
}

static int discard_handler(void* _ctx, Overlap* ovl, int novl)
{
    DiscardContext* ctx = (DiscardContext*)_ctx;
    HITS_READ* reads = ctx->db->reads;

    uint64_t a = ovl->aread;
    uint64_t stats_discarded = 0;

    if ( reads[a].flags )
    {
        int i;
        for ( i = 0; i < novl; i++ )
        {
            if ( ! (ovl[i].flags & OVL_DISCARD) )
            {
                ovl[i].flags |= OVL_DISCARD;
                stats_discarded += 1;
            }
        }
    }
    else
    {
        int i;
        for ( i = 0; i < novl; i++ )
        {
            int b = ovl[i].bread;

            if ( reads[b].flags && !(ovl[i].flags & OVL_DISCARD) )
            {
                ovl[i].flags |= OVL_DISCARD;
                stats_discarded += 1;
            }
        }
    }

    ctx->stats_discarded += stats_discarded;

    return 1;
}

static void usage(FILE* fout, const char* app)
{
    fprintf( fout, "usage: %s [-p] database input.las output.las rid [rid ...]\n\n", app );
}

int main(int argc, char* argv[])
{
    HITS_DB db;
    PassContext* pctx;
    FILE* fileOvlIn;
    FILE* fileOvlOut;
    DiscardContext dctx;
    char* app = argv[ 0 ];
    int purge = 0;

    bzero(&dctx, sizeof(DiscardContext));

    opterr = 0;

    int c;
    while ((c = getopt(argc, argv, "p")) != -1)
    {
        switch (c)
        {
            case 'p':
                    purge = 1;
                    break;

            default:
                    usage(stdout, app);
                    exit(1);
        }
    }

    if (argc - optind < 3)
    {
        usage(stdout, app);
        exit(1);
    }

    char* pcPathReadsIn = argv[optind++];
    char* pcPathOverlapsIn = argv[optind++];
    char* pcPathOverlapsOut = argv[optind++];

    if ( (fileOvlIn = fopen(pcPathOverlapsIn, "r")) == NULL )
    {
        fprintf(stderr, "could not open input track '%s'\n", pcPathOverlapsIn);
        exit(1);
    }

    if ( (fileOvlOut = fopen(pcPathOverlapsOut, "w")) == NULL )
    {
        fprintf(stderr, "could not open output track '%s'\n", pcPathOverlapsOut);
        exit(1);
    }

    if ( Open_DB(pcPathReadsIn, &db) )
    {
        fprintf(stderr, "could not open database '%s'\n", pcPathReadsIn);
        exit(1);
    }

    int i;
    for ( i = 0; i < db.nreads; i++ )
    {
        db.reads[i].flags = 0;
    }

    while ( optind < argc )
    {
        char* end;
        int rid = strtol(argv[optind], &end, 10);

        if ( rid < 0 || rid > db.nreads || *end != '\0' )
        {
            fprintf(stderr, "invalid read id %s\n", argv[optind]);
            exit(1);
        }

        db.reads[rid].flags = 1;

        optind += 1;
    }

    dctx.db = &db;

    pctx = pass_init(fileOvlIn, fileOvlOut);

    pctx->split_b = 0;
    pctx->data = &dctx;
    pctx->write_overlaps = 1;
    pctx->load_trace = 1;
    pctx->purge_discarded = purge;

    discard_pre(pctx, &dctx);

    pass(pctx, discard_handler);

    discard_post(&dctx);

    // cleanup

    Close_DB(&db);

    pass_free(pctx);

    fclose(fileOvlIn);
    fclose(fileOvlOut);

    return 0;
}

