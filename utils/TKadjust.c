/*******************************************************************************************
 *
 *  move track from one database to another one. used for reusing parts of tracks
 *  when adding additional blocks to a database
 *
 *******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#if defined(__APPLE__)
    #include <sys/syslimits.h>
#else
    #include <linux/limits.h>
#endif

#include "db/DB.h"
#include "lib/tracks.h"
#include "lib/pass.h"
#include "lib/colors.h"
#include "lib/compression.h"

// toggles

#undef DEBUG

extern char* optarg;
extern int optind, opterr, optopt;

static void usage()
{
    printf( "usage: [-d] database.from database.to track\n\n" );
}

int main(int argc, char* argv[])
{
    // command line arguments

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

    if (argc - optind != 3)
    {
        usage();
        exit(1);
    }

    char* pcDbFrom = argv[optind++];
    char* pcDbTo = argv[optind++];
    char* pcTrack = argv[optind++];

    // open databases

    HITS_DB dbFrom;
    if (Open_DB(pcDbFrom, &dbFrom))
    {
        fprintf(stderr, "could not open database '%s'\n", pcDbFrom);
        exit(1);
    }

    HITS_DB dbTo;
    if (Open_DB(pcDbTo, &dbTo))
    {
        fprintf(stderr, "could not open database '%s'\n", pcDbTo);
        exit(1);
    }

    uint64_t nreadsFrom = DB_NREADS(&dbFrom);
    uint64_t nreadsTo = DB_NREADS(&dbTo);

    // only allow moving to a larger database

    if ( nreadsFrom > nreadsTo )
    {
        fprintf(stderr, "moving to a smaller database is not supported\n");
        exit(1);
    }

    // make sure read lengths are the same

    uint64_t i;
    for ( i = 0; i < nreadsFrom; i++)
    {
        if ( dbFrom.reads[i].rlen != dbTo.reads[i].rlen )
        {
            fprintf(stderr, "reads %" PRIu64 " mismatch in length\n", i);
            exit(1);
        }
    }

    // load track through the source database

    HITS_TRACK* track = track_load(&dbFrom, pcTrack);

    if ( track == NULL )
    {
        fprintf(stderr, "failed to open track %s\n", pcTrack);
        exit(1);
    }

    // pad annotation track to make it fit the new database

    track_anno* anno = track->anno;

    anno = track->anno = realloc( anno, sizeof(track_anno) * (nreadsTo + 1) );

    track_anno fill = anno[ nreadsFrom ];
    for ( i = nreadsFrom; i <= nreadsTo ; i++ )
    {
        anno[i] =fill;
    }

    // write track and clean up

    track_write(&dbTo, pcTrack, 0, anno, track->data, anno[ nreadsFrom ] / sizeof(track_data));

    Close_DB(&dbFrom);
    Close_DB(&dbTo);

    return 0;
}
