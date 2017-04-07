/*******************************************************************************************
 *
 *  merge multiple tracks of the same type into one
 *
 *  Date   : October 2016
 *
 *  Author : MARVEL Team
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
    printf("[-d] <db> <track>\n");
    printf("Options: -d ... remove single tracks after merging\n");
}

int main(int argc, char* argv[])
{
    // args

    int delete = 0;
    opterr = 0;

    int c;

    while ((c = getopt(argc, argv, "d")) != -1)
    {
        switch (c)
        {
            case 'd':
                delete = 1;
                break;
        }
    }

    if (argc - optind != 2)
    {
        usage();
        exit(1);
    }

    char* pcDb = argv[optind++];
    char* pcTrack = argv[optind++];

    HITS_DB db;
    if (Open_DB(pcDb, &db))
    {
        fprintf(stderr, "could not open database '%s'\n", pcDb);
        exit(1);
    }

    int nblocks = DB_Blocks(pcDb);
    if (nblocks < 1)
    {
        fprintf(stderr, "failed to get number of blocks\n");
        exit(1);
    }

    uint64_t nreads = DB_NREADS(&db);

    track_anno* offset = (track_anno*) malloc(sizeof(track_anno) * (nreads + 1));
    bzero(offset, sizeof(track_anno) * (nreads + 1));

    FILE* fileDataOut;
    char path[PATH_MAX];

    sprintf(path, "%s.%s.d2", db.path, pcTrack);

    if ((fileDataOut = fopen(path, "w")) == NULL)
    {
        fprintf(stderr, "ERROR [mergeTracks]: Cannot open file %s!\n", path);
        exit(1);
    }

    int i;
    uint64_t cdata_total = 0;

    for ( i = 1 ; i <= nblocks ; i++)
    {
        sprintf(path, "%d.%s", i, pcTrack);
        HITS_TRACK* track = track_load(&db, path);

        if (!track)
        {
            fprintf(stderr, "Unable to merge all tracks, stopped at block %d. Cannot open file %s\n", i, path);
            exit(1);
        }

        track_anno* offset_in = track->anno;

        uint64_t j;
        for (j = 0; j < nreads; j++)
        {
            track_anno ob = offset_in[j];
            track_anno oe = offset_in[j + 1];

            if (ob > oe)
            {
                fprintf(stderr, "ERROR: ob > oe read %" PRIu64 " ob %lld oe %lld\n", j, ob, oe);
                exit(1);
            }

            if (ob < oe && offset[j] != 0)
            {
                fprintf(stderr, "ERROR: not merging in proper order\n");
                exit(1);
            }

            offset[j] += (oe - ob);
        }

        void* canno = NULL;
        uint64_t clen = 0;

        compress_chunks(track->data, offset_in[nreads], &canno, &clen);

        cdata_total += clen;

        // fwrite(track->data, offset_in[nreads], 1, fileDataOut);
        fwrite(canno, clen, 1, fileDataOut);

        free(canno);

        Close_Track(&db, track->name);
    }

    fclose(fileDataOut);

    track_anno off = 0;
    track_anno coff;

    uint64_t j;
    for (j = 0; j <= nreads; j++)
    {
        coff = offset[j];
        offset[j] = off;
        off += coff;
    }

    for (j = 0; j < nreads; j++)
    {
        assert(offset[j] <= offset[j + 1]);
    }

    track_write(&db, pcTrack, 0, offset, NULL, cdata_total);

    if (delete)
    {
        for ( i = 1 ; i <= nblocks ; i++)
        {
            sprintf(path, "%d.%s", i, pcTrack);
            track_delete(&db, path);
        }
    }

    free(offset);

    Close_DB(&db);

    return 0;
}
