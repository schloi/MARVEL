
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "lib/pass.h"
#include "db/DB.h"
#include "lib/utils.h"
#include "lib/tracks.h"
#include "dalign/align.h"
#include "lib/oflags.h"

#undef DEBUG

#define OUTPUT_DOT   0
#define OUTPUT_NODES 1

// #define STATUS_UNDEFINED    0
#define STATUS_NEIGHBOR     1
#define STATUS_CONTAINED    2


extern char* optarg;
extern int optind, opterr, optopt;

static void usage()
{
    printf("usage:   [-r<int(1)>] <db> <overlaps_in:ovl> <overlaps_out:ovl> <read:int>\n");
    printf("options: ...\n");
}

int main(int argc, char* argv[])
{
    Overlap ovl;
    HITS_DB db;
    HITS_TRACK* trackTrim;
    FILE* fileOvlIn;
    FILE* fileOvlOut;
    // ovl_header_novl novl_out;
    ovl_header_novl novl;
    ovl_header_twidth twidth;

    int output = OUTPUT_DOT;
    int radius = 1;
    int drop_contained = 1;

    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "r:")) != -1)
    {
        switch (c)
        {
            case 'r':
                    radius = atoi(optarg);
                    break;

            default:
                    usage();
                    exit(1);
        }
    }

    if (argc - optind != 4)
    {
        usage();
        exit(1);
    }

    char* pcPathDb = argv[optind++];
    char* pcPathOverlapsIn = argv[optind++];
    char* pcPathOverlapsOut = argv[optind++];
    int read = atoi( argv[optind++] );

    if ( Open_DB(pcPathDb, &db) )
    {
        fprintf(stderr, "could not open %s\n", pcPathDb);
        exit(1);
    }

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

    trackTrim = track_load(&db, TRACK_TRIM);

    if (!trackTrim)
    {
        fprintf(stderr, "failed to open track %s\n", TRACK_TRIM);
        exit(1);
    }

//    Trim_DB(&db);

    if ( fread(&novl, sizeof(ovl_header_novl), 1, fileOvlIn) != 1 )
    {
        fprintf(stderr, "ERROR: reading novl failed\n");
        exit(1);
    }


    if ( fread(&twidth, sizeof(ovl_header_twidth), 1, fileOvlIn) != 1)
    {
        fprintf(stderr, "ERROR: reading twidth failed\n");
        exit(1);
    }

    int tbytes = TBYTES(twidth);

    int* offsets = (int*)malloc(sizeof(int) * db.nreads);
    bzero(offsets, sizeof(int) * db.nreads);

    unsigned char* status = (unsigned char*)malloc((db.nreads));
    bzero(status, db.nreads);

    uint64 dmax = 100;
    int* data = (int*)malloc(sizeof(int) * dmax);
    uint64 dcur = 0;

    // create edge list

    printf("creating edge list\n");

    while (!Read_Overlap(fileOvlIn, &ovl))
    {
        fseeko(fileOvlIn, tbytes * ovl.path.tlen, SEEK_CUR);

        if (ovl.flags & OVL_DISCARD)
        {
            continue;
        }

        offsets[ ovl.aread ]++;

        data[ dcur ] = ovl.bread;
        dcur++;

        int trim_ab, trim_ae, trim_bb, trim_be;
        int ovlALen, ovlBLen;

        ovlALen = DB_READ_LEN(&db, ovl.aread);
        ovlBLen = DB_READ_LEN(&db, ovl.bread);

        if (trackTrim)
        {
            get_trim(&db, trackTrim, ovl.aread, &trim_ab, &trim_ae);
            get_trim(&db, trackTrim, ovl.bread, &trim_bb, &trim_be);
        }
        else
        {
            trim_ab = 0;
            trim_ae = ovlALen;

            trim_bb = 0;
            trim_be = ovlBLen;
        }

        if (ovl.flags & OVL_COMP)
        {
            int t = trim_bb;
            trim_bb = ovlBLen - trim_be;
            trim_be = ovlBLen - t;
        }

        int begpos = (ovl.path.abpos - trim_ab) - (ovl.path.bbpos - trim_bb);
        int endpos = (ovl.path.aepos - trim_ab) - (ovl.path.bepos - trim_bb) - ( (trim_ae - trim_ab) - (trim_be - trim_bb) );

        // begpos = o.path.abpos - o.path.bbpos;
        // endpos = (o.path.aepos - o.path.bepos) - (len[aread] - len[bread]);

        if (begpos == 0 && endpos == 0)
        {
            if (ovl.aread > ovl.bread)
            {
                status[ovl.aread] = STATUS_CONTAINED;
            }
        }
        else if (begpos <= 0 && endpos >= 0)
        {
            status[ovl.aread] = STATUS_CONTAINED;
        }

        if (dcur >= dmax)
        {
            dmax = dmax * 1.2 + 100;
            data = (int*)realloc(data, sizeof(int)*dmax);
        }
    }

    int nreads = ovl.aread;
    int i, j;

    uint64 off, coff;
    off = 0;

    for (i = 0; i <= nreads; i++)
    {
        coff = offsets[i];
        offsets[i] = off;
        off += coff;
    }

    // collect neighborhood

    printf("collecting neighborhood");

    int smax = 100;
    int scur = 0;
    int* stack = (int*)malloc(sizeof(int)*smax);
    int* rsoff = (int*)malloc(sizeof(int)*(radius+1));
    bzero(rsoff, sizeof(int)*(radius+1));

    rsoff[0] = 0;

    stack[ scur ] = read;
    status[ read ] |= STATUS_NEIGHBOR;
    scur++;

    int rcur = 1;
    while (rcur <= radius)
    {
        rsoff[rcur] = scur;

        for (j = rsoff[rcur-1]; j < rsoff[rcur]; j++)
        {
            read = stack[j];

            printf("r %d of %d\n", rcur, read);

            uint64 ob = offsets[read];
            uint64 oe = offsets[read+1];

            while (ob < oe)
            {
                read = data[ob];

                if (!(status[read] & STATUS_NEIGHBOR))
                {
                    status[read] |= STATUS_NEIGHBOR;
                    stack[ scur ] = read;

                    scur++;

                    printf("  -> %d", read);
                    if (status[read] & STATUS_CONTAINED)
                    {
                        printf(" C\n");
                    }
                    else
                    {
                        printf("\n");
                    }

                    if (scur >= smax)
                    {
                        smax = 1.2 * smax + 100;
                        stack = (int*)realloc(stack, sizeof(int)*smax);
                    }
                }

                ob++;
            }
        }

        rcur++;
    }

    printf("writing overlaps\n");

    if (output == OUTPUT_NODES)
    {
        for (i = 0; i < scur; i++)
        {
            if (drop_contained && (status[ stack[i] ] & STATUS_CONTAINED))
            {
                continue;
            }

            fprintf(fileOvlOut, "%d\n", stack[i]);
        }
    }
    else
    {
        fseeko(fileOvlIn, sizeof(ovl_header_novl) + sizeof(ovl_header_twidth), SEEK_SET);

        fprintf(fileOvlOut, "digraph nh {\n");

        while (!Read_Overlap(fileOvlIn, &ovl))
        {
            fseeko(fileOvlIn, tbytes * ovl.path.tlen, SEEK_CUR);

            if (ovl.flags & OVL_DISCARD)
            {
                continue;
            }

            if ( (status[ ovl.aread ] & STATUS_NEIGHBOR) &&
                 (status[ ovl.bread ] & STATUS_NEIGHBOR) )
            {
                if (drop_contained && ((status[ ovl.aread ] & STATUS_CONTAINED) ||
                                       (status[ ovl.bread ] & STATUS_CONTAINED)))
                {
                    continue;
                }

                if (ovl.aread < ovl.bread)
                {
                    fprintf(fileOvlOut, "%d -> %d [color=orange];\n", ovl.aread, ovl.bread);
                }
                else
                {
                    fprintf(fileOvlOut, "%d -> %d;\n", ovl.aread, ovl.bread);
                }
            }

        }

        fprintf(fileOvlOut, "}\n");
    }

    fclose(fileOvlOut);
    fclose(fileOvlIn);

    free(status);
    free(rsoff);
    free(stack);
    free(offsets);
    free(data);

    return 0;
}
