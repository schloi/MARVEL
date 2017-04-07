/*******************************************************************************************
 *
 *  Date  :  November 2016
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

#define DEF_ARG_E 1500
#define DEF_ARG_S 4

extern char* optarg;
extern int optind, opterr, optopt;

static void usage()
{
	fprintf(stderr, "[-voS] [-es <int>] <db> <trackIn> <trackOut>\n");
	fprintf(stderr, "Options: -v ... verbose\n");
	fprintf(stderr, "         -S ... summary stats for trimmed track\n");
	fprintf(stderr, "		  -e ... remove n bases from given interval track (%d)\n", DEF_ARG_E);
	fprintf(stderr, "         -s ... # bytes for an entry (%d)\n", DEF_ARG_S);
	fprintf(stderr, "         -o ... old version (-1)\n");
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
		return *(uint32*) (v);
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

	HITS_TRACK* trackIn =  NULL;
	HITS_TRACK* trackOut = NULL;

	char* pcDb;
	char* pcTrackIn;
	char* pcTrackOut;

	int numBases = DEF_ARG_E;
	int dsize = DEF_ARG_S;
	int verbose = 0;
	int stats = 0;
	int oldVersion = 0;

	// args

	opterr = 0;

	int c;

	while ((c = getopt(argc, argv, "voSe:s:")) != -1)
	{
		switch (c)
		{
		case 'v':
			verbose = 1;
			break;
		case 'S':
			stats = 1;
			break;
		case 'o':
			oldVersion = 1;
			break;
		case 'e':
			numBases = atoi(optarg);
			break;
        case 's':
            dsize = atoi(optarg);
            break;


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

	pcDb = argv[optind++];
	pcTrackIn = argv[optind++];
	pcTrackOut = argv[optind++];

    if (!isPowerOfTwo(dsize))
    {
        fprintf(stderr, "-s must be a power of 2\n");
        exit(1);
    }

	if (strcmp(pcTrackIn, pcTrackOut) == 0)
	{
		fprintf(stderr, "trackIn and trackOut cannot be the same!\n");
		exit(1);
	}

	if (Open_DB(pcDb, &db))
	{
		fprintf(stderr, "failed to open database '%s'\n", pcDb);
		exit(1);
	}

	trackIn = track_load(&db, pcTrackIn);

	if (trackIn == NULL)
	{
		fprintf(stderr, "could not open track '%s'\n", pcTrackIn);
		exit(1);
	}

	void* annoIn = trackIn->anno;
	void* dataIn = trackIn->data;

	int nreads = DB_NREADS(&db);

	track_anno* annoOut = (track_anno*)malloc(sizeof(track_anno) * (nreads + 1));
	track_data* dataOut = (track_data*)malloc( ((track_anno*)annoIn)[nreads] );
	track_anno tcur = 0;
	bzero(annoOut, sizeof(track_anno) * (nreads + 1));

    track_anno_header header;
    bzero(&header, sizeof(track_anno_header));

    trackOut = malloc(sizeof(HITS_TRACK));
    trackOut->name = strdup(pcTrackOut);
    trackOut->data = dataOut;
    trackOut->anno = annoOut;
    trackOut->size = header.size;

	int i, j, rlen;
	uint64_t bi, ei;
	track_data v1 ,v2;
	uint64_t nTrimIn, nTrimOut;
	nTrimIn = nTrimOut = 0;

	for (i = 0; i < db.nreads; i++)
	{
		if (trackIn->size == sizeof(int))
		{
			bi = ((uint32*) annoIn)[i];
			ei = ((uint32*) annoIn)[i + 1];
		}
		else
		{
			bi = ((uint64_t*) annoIn)[i];
			ei = ((uint64_t*) annoIn)[i + 1];
		}

		if (bi >= ei)
		{
			continue;
		}

		if (((ei - bi) / dsize) & 1)
		{
			fprintf(stderr, "ERROR: Track %s is not an interval track!\n", pcTrackIn);
				exit(1);
		}

		if(verbose)
			printf("%d (%" PRIu64 ")", i, (ei - bi) / dsize);
		rlen = DB_READ_LEN(&db, i);
		if(verbose)
			printf(" l(%d)", rlen);

		while (bi < ei)
		{
			v1 = value(dataIn + bi, dsize);
			v2 = value(dataIn + bi + dsize, dsize);

			nTrimIn += v2 - v1;

			if(verbose)
				printf(" %d %d", v1, v2);
			if(v2 < numBases || v1 > rlen - numBases)
			{
				v1 = -1;
				v2 = -1;
			}
			else
			{
				if(v1 < numBases)
				{
					v1 = numBases;
				}
				if(v2 > rlen - numBases)
				{
					v2 = rlen - numBases;
				}
			}

			if(v2 > v1)
			{
				dataOut[ tcur++ ] = v1;
			    dataOut[ tcur++ ] = v2;
			    annoOut[ i ] += 2 * sizeof(track_data);
			    nTrimOut += v2 - v1;
			}

			if(verbose)
			{
				if(v2 > v1)
					printf(" --> [%d %d]", v1, v2);
				else
					printf(" --> skip");
			}
			bi += 2 * dsize;

		}
		if(verbose)
			printf("\n");
	}

    track_anno qoff, coff;
    qoff = 0;

    for (j = 0; j <= nreads; j++)
    {
        coff = annoOut[j];
        annoOut[j] = qoff;
        qoff += coff;
    }

    if(oldVersion)
    	write_track_trimmed(&db, trackOut->name, 0, trackOut->anno, trackOut->data, tcur);
    else
    	track_write(&db, trackOut->name, 0, trackOut->anno, trackOut->data, tcur);

	if (db.tracks == NULL)
		track_close(trackIn);

	track_close(trackOut);

	Close_DB(&db);

	if(stats)
	{
		printf("pre bases: %10" PRIu64 "\n", nTrimIn);
		printf("post bases:%10" PRIu64 "\n", nTrimOut);
	}

	return 0;
}

