
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
#include "lib/lasidx.h"

#include "db/DB.h"
#include "dalign/align.h"

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

static void usage()
{
    printf("<db> <gff> <track>\n");
}

static int cmp_ints(const void* x, const void* y)
{
    int* a = (int*)x;
    int* b = (int*)y;

    int l = a[0];
    int r = b[0];

    if (l == r)
    {
        l = a[1];
        r = b[1];
    }

    return l - r;
}

int main(int argc, char* argv[])
{
    HITS_DB db;

    // process arguments

    int c;

    opterr = 0;

    while ((c = getopt(argc, argv, "")) != -1)
    {
        switch (c)
        {
            default:
                printf("Unknow option: %s\n", argv[optind - 1]);
                usage();
                exit(1);
        }
    }

    if (argc - optind < 3)
    {
        usage();
        exit(1);
    }

    char* pathDb = argv[optind++];
    char* pathGff = argv[optind++];
    char* nameTrack = argv[optind++];

    if (Open_DB(pathDb, &db))
    {
        printf("could not open '%s'\n", pathDb);
        return 1;
    }

    FILE* fileIn;

    if (strcmp(pathGff, "-") == 0)
    {
        fileIn = stdin;
    }
    else
    {
        fileIn = fopen(pathGff, "r");

        if (fileIn == NULL)
        {
            fprintf(stderr, "failed to open '%s'\n", pathGff);
            exit(1);
        }
    }

    char seq[128];
    char source[128];
    char feature[128];
    int b, e, len;
    char strand;
    char dummy[128];

    int* ints = NULL;
    int nints = 0;
    int maxints = 0;

    while ( 1 )
    {
        int count = fscanf(fileIn, "%s\t%s\t%s\t%d\t%d\t%d\t%c\t%s\n", seq, source, feature, &b, &e, &len, &strand, dummy);

        if (count != 8)
        {
            break;
        }

        if (nints + 3 >= maxints)
        {
            maxints = 1.2 * maxints + 1000;
            ints = realloc(ints, maxints * sizeof(int));
        }

        int seqid = -1;
        if (strstr(seq, "read_") != NULL)
        {
            sscanf(seq, "read_%d", &seqid);
        }
        else
        {
            seqid = atoi(seq);
        }

        ints[nints + 0] = seqid;
        ints[nints + 1] = b;
        ints[nints + 2] = e;

        int rlen = DB_READ_LEN(&db, seqid);

        if ( b < 0 || b > rlen || e < b || e > rlen )
        {
            printf("interval out of bounds %d..%d (%d)\n", b, e, rlen);
        }


        nints += 3;
    }

    qsort(ints, nints/3, sizeof(int) * 3, cmp_ints);

    track_anno* anno = malloc( sizeof(track_anno) * (DB_NREADS(&db) + 1) );
    track_data* data = NULL;
    int ndata = 0;
    int maxdata = 0;

    bzero(anno, sizeof(track_anno) * (DB_NREADS(&db) + 1));

    int i;
    for (i = 0; i < nints; i += 3)
    {
        if (ndata + 2 >= maxdata)
        {
            maxdata = maxdata * 1.2 + 1000;
            data = realloc( data, sizeof(track_data) * maxdata );
        }

        anno[ ints[i + 0] ] += 2 * sizeof(track_data);
        data[ ndata + 0 ] = ints[ i + 1 ];
        data[ ndata + 1 ] = ints[ i + 2 ];

        ndata += 2;
    }

    track_anno coff, off;
    off = 0;

    for (i = 0; i <= DB_NREADS(&db); i++)
    {
        coff = anno[i];
        anno[i] = off;
        off += coff;
    }

    track_write(&db, nameTrack, 0, anno, data, ndata);

    free(anno);
    free(data);

    Close_DB(&db);

    return 0;
}
