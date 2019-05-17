
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

typedef struct
{
    uint64_t rid;
    uint64_t nvalues;
    uint64_t idxvalues;
} READ_ANNO;

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

static void usage()
{
    printf("[-c] <db> <text> <track>\n");
    printf("-c ... ensure values are valid positions within the reads\n");
}

static int cmp_read_anno(const void* x, const void* y)
{
    return ((const READ_ANNO*)x)->rid - ((const READ_ANNO*)y)->rid;
}

int main(int argc, char* argv[])
{
    HITS_DB db;

    // process arguments

    int c;
    int check = 0;

    opterr = 0;

    while ((c = getopt(argc, argv, "c")) != -1)
    {
        switch (c)
        {
            case 'c':
                check = 1;
                break;

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
    char* pathTxt = argv[optind++];
    char* nameTrack = argv[optind++];

    if (Open_DB(pathDb, &db))
    {
        printf("could not open '%s'\n", pathDb);
        return 1;
    }

    FILE* fileIn;

    if (strcmp(pathTxt, "-") == 0)
    {
        fileIn = stdin;
    }
    else
    {
        fileIn = fopen(pathTxt, "r");

        if (fileIn == NULL)
        {
            fprintf(stderr, "failed to open '%s'\n", pathTxt);
            exit(1);
        }
    }

    track_data* ints = NULL;
    uint64_t nints = 0;
    size_t maxints = 0;

    char* line = NULL;
    size_t maxline = 0;
    ssize_t linelen;

    READ_ANNO* anno = NULL;
    size_t maxanno = 0;
    uint64_t nanno = 0;

    while ( ( linelen = getline(&line, &maxline, fileIn) ) > 0 )
    {
        assert( line[ strlen( line ) - 1 ] == '\n' );

        uint32_t col = 0;
        char* token;
        uint64_t readid;

        while ( ( token = strsep(&line, " \t") ) )
        {
            int64_t value = strtoll(token, NULL, 10);
            col += 1;

            if ( col == 1 )
            {
                readid = value;
                continue;
            }

            int rlen = DB_READ_LEN(&db, readid);

            if ( check && ( value < 0 || value > rlen ) )
            {
                fprintf(stderr, "invalid position %" PRId64 " for read %" PRIu64 "\n", value, readid);
            }

            if ( nints + 1 >= maxints )
            {
                maxints = 1.2 * maxints + 1000;
                ints = realloc(ints, sizeof(track_data) * maxints);
            }

            ints[ nints ] = value;
            nints += 1;

        }

        if (nanno + 1 >= maxanno )
        {
            maxanno = maxanno * 1.2 + 1000;
            anno = realloc(anno, sizeof(READ_ANNO) * maxanno);
        }

        anno[ nanno ].rid = readid;
        anno[ nanno ].nvalues = col - 1;
        anno[ nanno ].idxvalues = nints - ( col - 1 );

        nanno += 1;
    }

    qsort(anno, nanno, sizeof(READ_ANNO), cmp_read_anno);

    track_anno* tanno = malloc( sizeof(track_anno) * (DB_NREADS(&db) + 1) );
    track_data* tdata = NULL;
    uint64_t ntdata = 0;
    size_t maxtdata = 0;

    bzero(tanno, sizeof(track_anno) * (DB_NREADS(&db) + 1));

    uint64_t i;
    for (i = 0; i < nanno; i++ )
    {
        READ_ANNO* ra = anno + i;

        if (ntdata + ra->nvalues >= maxtdata)
        {
            maxtdata = maxtdata * 1.2 + ra->nvalues;
            tdata = realloc( tdata, sizeof(track_data) * maxtdata );
        }

        tanno[ ra->rid ] += ra->nvalues * sizeof(track_data);

        uint64_t j;
        for ( j = 0; j < ra->nvalues; j++ )
        {
            tdata[ ntdata ] = ints[ ra->idxvalues + j ];
            ntdata += 1;
        }
    }

    free(anno);

    track_anno coff, off;
    off = 0;

    for (i = 0; i <= (uint64_t)DB_NREADS(&db); i++)
    {
        coff = tanno[i];
        tanno[i] = off;
        off += coff;
    }

    track_write(&db, nameTrack, 0, tanno, tdata, ntdata);

    free(tanno);
    free(tdata);

    Close_DB(&db);

    return 0;
}
