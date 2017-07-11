/*******************************************************************************************
 *
 *  Date  :  February 2015
 *
 *******************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "lib/tracks.h"

#include "db/DB.h"
#include "dalign/align.h"
#include "lib/pass.h"

#define DEF_ARG_B 1000
#define DEF_ARG_S 4

extern char* optarg;
extern int optind, opterr, optopt;

static void usage()
{
    printf( "usage: [-ntpdi] [-s n] [-b n] database track\n\n" );

    printf( "Display the contents of an annotation track.\n\n" );

    printf( "options: -p  prefix each line printed with the track name\n" );
    printf( "         -d  print the distance between intervals\n" );
    printf( "         -i  annotation is in form of intervals\n" );
    printf( "         -s  each entry in the annotation track has n bytes (default %d)\n", DEF_ARG_S );
    printf( "         -S  display a histogram of the interval lengths\n" );
    printf( "         -b  bin size for histogram (%d)\n", DEF_ARG_B );
    printf( "         -n  new line after each track entry\n" );
}

static int isPowerOfTwo(unsigned int x)
{
    return ((x != 0) && ((x & (~x + 1)) == x));
}

static uint64_t value(void* v, int bytes)
{
    if (bytes == 1)
    {
        return *(unsigned char*) (v);
    }
    else if (bytes == 2)
    {
        return *(unsigned short*) (v);
    }
    else if (bytes == 4)
    {
        return *(uint32_t*) (v);
    }
    else if (bytes == 8)
    {
        return *(uint64_t*) (v);
    }

    return 0;
}

int main(int argc, char* argv[])
{
    HITS_DB db;

    HITS_TRACK* track = NULL;

    int prefix, dist, intervals, stats, newline;
    char* pcDb;
    char* pcTrack;
    int binsize = DEF_ARG_B;
    int dsize   = DEF_ARG_S;

    // args

    newline = prefix = dist = intervals = stats = 0;

    opterr = 0;

    int c;

    while ((c = getopt(argc, argv, "nStTpdis:b:")) != -1)
    {
        switch (c)
        {
            case 'n':
                newline = 1;
                break;

            case 'b':
                binsize = atoi(optarg);
                break;

            case 'p':
                prefix = 1;
                break;

            case 'd':
                dist = 1;
                break;

            case 'i':
                intervals = 1;
                break;

            case 's':
                dsize = atoi(optarg);
                break;

            case 'S':
                stats = 1;
                break;

            default:
                usage();
                break;
        }
    }

    if (argc - optind != 2)
    {
        usage();
        exit(1);
    }

    if (!isPowerOfTwo(dsize))
    {
        fprintf(stderr, "-s must be a power of 2\n");
        exit(1);
    }

    pcDb = argv[optind++];
    pcTrack = argv[optind++];

    if (Open_DB(pcDb, &db))
    {
        fprintf(stderr, "failed to open database '%s'\n", pcDb);
        exit(1);
    }

    track = track_load(&db, pcTrack);

    if (track == NULL)
    {
        fprintf(stderr, "could not open track '%s'\n", pcTrack);
        exit(1);
    }

    void* anno = track->anno;
    void* data = track->data;

    int i;
    uint64_t b, e;

    uint64_t stats_total = 0;
    uint64_t stats_intervals = 0;
    uint64_t stats_itotal = 0;

    int maxlen = db.maxlen;
    int nbin = (maxlen - 1) / binsize + 1;

    int* hist = malloc(sizeof(int) * nbin);
    uint64_t* tsum = malloc(sizeof(uint64_t) * nbin);

    if (hist == NULL || tsum == NULL)
    {
        exit(1);
    }

    bzero(hist, nbin * sizeof(int));
    bzero(tsum, nbin * sizeof(uint64_t));

    for (i = 0; i < db.nreads; i++)
    {
        if (track->size == sizeof(int))
        {
            b = ((uint32_t*) anno)[i];
            e = ((uint32_t*) anno)[i + 1];
        }
        else
        {
            b = ((uint64_t*) anno)[i];
            e = ((uint64_t*) anno)[i + 1];
        }

        if (stats)
        {
            stats_total += DB_READ_LEN(&db, i);

            while (b < e)
            {
                stats_intervals++;
                int len = value(data + b + dsize, dsize) - value(data + b, dsize);
                stats_itotal += len;

                hist[len / binsize] += 1;
                tsum[len / binsize] += len;
                b += 2 * dsize;
            }
        }
        else
        {
            if (b >= e)
            {
                continue;
            }

            if (dist)
            {
                b += 2 * dsize;
            }

            if (!newline)
            {
                if (prefix)
                {
                    printf("%s ", pcTrack);
                }

                printf( "%d", i );
            }

            if (strcmp(pcTrack, TRACK_PACBIO_CHEM) == 0)
            {
                printf(" %.*s\n", (int)((e - b) / dsize), (char*) data + b);
                continue;
            }

            while (b < e)
            {
                if (newline)
                {
                    if (prefix)
                    {
                        printf("%s ", pcTrack);
                    }

                    printf("%d", i);
                }

                if (dist)
                {
                    printf(" %" PRIu64, value(data + b, dsize) - value(data + b - dsize, dsize));
                    b += 2 * dsize;
                }
                else if (intervals)
                {
                    printf(" %" PRIu64 "-%" PRIu64, value(data + b, dsize), value(data + b + dsize, dsize));
                    b += 2 * dsize;
                }
                else
                {
                    printf(" %" PRIu64, value(data + b, dsize));
                    b += 1 * dsize;
                }

                if (newline)
                {
                    printf("\n");
                }
            }

            if (!newline)
            {
                printf("\n");
            }
        }
    }

    if (stats)
    {
        printf("%'10d reads\n", db.nreads);
        printf("%'10" PRIu64 " bases\n", stats_total);
        printf("%'10" PRIu64 " intervals\n", stats_intervals);
        printf("%'10" PRIu64 " bases in intervals (%.1f%%)\n\n", stats_itotal, 100.0 * stats_itotal / stats_total);

        printf("Statistics for %s-track\n\n", track->name);

        printf("There are %" PRIu64 " intervals totaling %" PRIu64 " bases (%.1f%% of all data)\n\n", stats_intervals, stats_itotal, (100. * stats_itotal) / db.totlen);

        {
            int64_t btot;
            uint64_t cum;
            int k;

            printf("Distribution of %s intervals (Bin size = %d)\n\n", track->name, binsize);

            printf("%8s %8s %12s %7s %10s %10s\n", "Bin", "Count", "% Intervals", "% Bases", "cum avg", "bin avg");

            cum = 0;
            btot = 0;

            for (k = nbin - 1; k >= 0; k--)
            {
                cum += hist[k];
                btot += tsum[k];

                if (hist[k] > 0)
                {
                    printf("%'8d %'8d %12.1f %7.1f %'10" PRIu64 " %'10" PRIu64 "\n",
                            k * binsize,
                            hist[k],
                            (100. * cum) / stats_intervals,
                            (100. * btot) / stats_itotal,
                            btot / cum,
                            tsum[k] / hist[k]);

                    if (cum == stats_intervals)
                    {
                        break;
                    }
                }
            }

            printf("\n");
        }
    }

    // track_close(track);

    Close_DB(&db);

    return 0;
}
