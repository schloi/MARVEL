/************************************************************************************\
*                                                                                    *
 * Copyright (c) 2014, Dr. Eugene W. Myers (EWM). All rights reserved.                *
 *                                                                                    *
 * Redistribution and use in source and binary forms, with or without modification,   *
 * are permitted provided that the following conditions are met:                      *
 *                                                                                    *
 *  · Redistributions of source code must retain the above copyright notice, this     *
 *    list of conditions and the following disclaimer.                                *
 *                                                                                    *
 *  · Redistributions in binary form must reproduce the above copyright notice, this  *
 *    list of conditions and the following disclaimer in the documentation and/or     *
 *    other materials provided with the distribution.                                 *
 *                                                                                    *
 *  · The name of EWM may not be used to endorse or promote products derived from     *
 *    this software without specific prior written permission.                        *
 *                                                                                    *
 * THIS SOFTWARE IS PROVIDED BY EWM ”AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES,    *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND       *
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL EWM BE LIABLE   *
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS  *
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY      *
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING     *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN  *
 * IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                      *
 *                                                                                    *
 * For any issues regarding this software and its use, contact EWM at:                *
 *                                                                                    *
 *   Eugene W. Myers Jr.                                                              *
 *   Bautzner Str. 122e                                                               *
 *   01099 Dresden                                                                    *
 *   GERMANY                                                                          *
 *   Email: gene.myers@gmail.com                                                      *
 *                                                                                    *
 \************************************************************************************/

/********************************************************************************************
 *
 *  Recreate all the .fasta files that have been loaded into a specified database.
 *
 *  Author:  Gene Myers
 *  Date  :  May 2014
 *
 ********************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>

#include "DB.h"
#include "lib/tracks.h"

static void usage()
{
    fprintf(stderr, "usage: [-vhUS] [-m <track>] [-w<int(80)>] <path:db>\n");
    fprintf(stderr, "options: -v ... verbose\n");
    fprintf(stderr, "         -h ... print this help message\n");
    fprintf(stderr, "         -U ... upper case bases\n");
    fprintf(stderr, "         -w ... line width (default: 80)\n");
    fprintf(stderr, "         -m ... add track to fasta header (multiple -m possible)\n");
    fprintf(stderr, "         -S ... sort DB according read length (longest first) \n");
}

extern char *optarg;
extern int optind, opterr, optopt;

HITS_DB _db;

static int cmp_read_by_len(const void* a, const void* b)
{
    return (_db.reads[*((int*) b)].rlen - _db.reads[*((int*) a)].rlen);
}

typedef struct
{
    int from;
    int to;
    char *fileName;
    char *readName;
} Prolog;

typedef struct
{
    int numHeader;
    int maxHeader;
    Prolog *headers;
} DB_Header;

static void addProlog(DB_Header* dbh, int from, int to, char *fname, char *rname)
{
#if DEBUG
    printf("addProlog, %d %d %s %s %d %d\n", from, to, fname, rname, strlen(fname), strlen(rname));
    fflush(stdout);
#endif
    if (dbh->maxHeader == 0)
    {
        dbh->maxHeader = 20;
        dbh->headers = (Prolog*) malloc(sizeof(Prolog) * dbh->maxHeader);
        if (dbh->headers == NULL)
        {
            fprintf(stderr, "[ERROR] - DB2fasta: Cannot allocate Prolog buffer\n.");
            exit(1);
        }
    }
    else if (dbh->maxHeader == dbh->numHeader)
    {
        dbh->maxHeader = (1.2 * dbh->numHeader) + 10;
        dbh->headers = (Prolog*) realloc(dbh->headers, sizeof(Prolog) * dbh->maxHeader);
        if (dbh->headers == NULL)
        {
            fprintf(stderr, "[ERROR] - DB2fasta: Cannot allocate Prolog buffer\n.");
            exit(1);
        }
    }
    Prolog *p = dbh->headers + dbh->numHeader;
    bzero(p, sizeof(Prolog));

    char *fileName = (char*) malloc(50);
    if (fileName == NULL)
    {
        printf("[ERROR] - DB2fasta: Cannot allocate Prolog header buffer\n.");
        exit(1);
    }

    char *readName = (char*) malloc(strlen(rname) + 10);

    if (readName == NULL)
    {
        fprintf(stderr, "[ERROR] - DB2fasta: Cannot allocate Prolog header buffer\n.");
        exit(1);
    }

    p->from = from;
    p->to = to;

    strcpy(fileName, fname);
    strcpy(readName, rname);

    p->fileName = fileName;
    p->readName = readName;
    dbh->numHeader++;

    fflush(stdout);
}

static int getPrologIndexOfRead(DB_Header* dbh, int readID)
{
    if (readID < 0 || readID > dbh->headers[dbh->numHeader - 1].to)
        return -1;

    else if (dbh->numHeader == 1)
        return 0;

    int min = 0;
    int max = dbh->numHeader - 1;

    Prolog *headers = dbh->headers;

    while (max >= min)
    {
        int i = (min + max) / 2;
        if (readID < headers[i].from)
            max = i - 1;
        else if (readID > headers[i].to)
            min = i + 1;
        else
            return i;
    }
    return -1;
}

int main(int argc, char *argv[])
{
    HITS_DB *db = &_db;
    FILE *dbfile;
    char *dbName;
    int nfiles;
    int VERBOSE, UPPER, WIDTH, SORT;

    HITS_TRACK **out_tracks = NULL;
    int curTracks = 0;
    int maxTracks = 0;

    int *readIDs = NULL;

    //  Process arguments
    {
        VERBOSE = 0;
        UPPER = 1;
        WIDTH = 80;
        SORT = 0;

        int c;
        opterr = 0;

        while ((c = getopt(argc, argv, "hvUSw:m:")) != -1)
        {
            switch (c)
            {
                case 'v':
                    VERBOSE += 1;
                    break;
                case 'U':
                    UPPER = 2;
                    break;
                case 'm':
                    if (curTracks >= maxTracks)
                    {
                        maxTracks += 10;
                        out_tracks = (HITS_TRACK**)realloc(out_tracks, sizeof(HITS_TRACK*) * maxTracks);
                    }

                    // use the HITS_TRACK* array as temporary storage of the track names
                    out_tracks[curTracks] = (HITS_TRACK*) optarg;
                    curTracks++;

                    break;
                case 'S':
                    SORT = 1;
                    break;
                case 'h':
                    usage();
                    return 0;
                case 'w':
                {
                    WIDTH = atoi(optarg);
                    if (WIDTH < 1)
                    {
                        fprintf(stderr, "Invalid line width of %d\n", WIDTH);
                        exit(1);
                    }
                }
                    break;
                default:
                    fprintf(stderr, "Unsupported argument: %s\n", argv[optind - 1]);
                    usage();
                    exit(1);
            }
        }
    }

    if (optind == argc)
    {
        fprintf(stderr, "[ERROR] - DB2fasta: A database is required\n\n");
        usage();
        exit(1);
    }

    //  Open db

    {
        int status;

        dbName = argv[optind];
        status = Open_DB(dbName, db);
        if (status < 0)
            exit(1);
        if (status == 1)
        {
            fprintf(stderr, "%s: Cannot be called on a .dam index: %s\n", argv[0], dbName);
            exit(1);
        }
        if (db->part > 0)
        {
            fprintf(stderr, "%s: Cannot be called on a block: %s\n", argv[0], dbName);
            exit(1);
        }
    }

    // Load Tracks
    {
        int i;
        for (i = 0; i < curTracks; i++)
        {
            char* track = (char*) out_tracks[i];
            out_tracks[i] = track_load(db, track);

            if (out_tracks[i] == NULL)
            {
                fprintf(stderr, "could not open track '%s'\n", track);
                exit(1);
            }
        }
    }
    {
        char *pwd, *root;

        pwd = PathTo(dbName);
        root = Root(dbName, ".db");
        dbfile = Fopen(Catenate(pwd, "/", root, ".db"), "r");
        free(pwd);
        free(root);
        if (dbfile == NULL)
            exit(1);
    }

    if (SORT)
    {
        int num = db->nreads;

        readIDs = (int*) malloc(sizeof(int) * num);
        if (readIDs == NULL)
        {
            fprintf(stderr, "[ERROR] - DB2fasta: Cannot allocate read id buffer for sorting!\n");
            exit(1);
        }
        int i;
        for (i = 0; i < num; i++)
            readIDs[i] = i;

        if (VERBOSE)
            printf("sorting ...");

        qsort(readIDs, num, sizeof(int), cmp_read_by_len);

        if (VERBOSE)
            printf(" done\n");
    }

    //  nfiles = # of files in data base

    if (fscanf(dbfile, DB_NFILE, &nfiles) != 1)
        SYSTEM_ERROR

    DB_Header *dbh = (DB_Header*) malloc(sizeof(DB_Header));
    if (dbh == NULL)
    {
        fprintf(stderr, "[ERROR] - DB2fasta: Cannot allocate hprolog buffer\n");
        exit(1);
    }
    bzero(dbh, sizeof(DB_Header));

    // parse prolog
    {
        int i;
        int first = 0;
        int last = 0;
        char file[MAX_NAME], fname[MAX_NAME];

        for (i = 0; i < nfiles; i++)
        {
            if (fscanf(dbfile, DB_FDATA, &last, fname, file) != 3)
            {
                fprintf(stderr, "[ERROR] - DB2fasta: Cannot read prolog at line %d of file %s\n", i + 2, argv[optind]);
                exit(1);
            }
            addProlog(dbh, first, last, fname, file);
        }
    }

    //  For each file do:

    {
        HITS_READ *reads;
        char *read;
        int f, first;
        int b, e;

        reads = db->reads;
        read = New_Read_Buffer(db);
        first = 0;
        for (f = 0; f < nfiles; f++)
        {
            int i, last;
            FILE *ofile;

            if (SORT)
            {
                ofile = stdout;
                last = db->nreads;
                nfiles = 1;
            }
            else
            {
                if ((ofile = Fopen(Catenate(".", "/", dbh->headers[f].fileName, ".fasta"), "w")) == NULL)
                    exit(1);
                last = dbh->headers[f].to;
                if (VERBOSE)
                {
                    fprintf(stderr, "Creating %s.fasta ...\n", dbh->headers[f].fileName);
                    fflush(stdout);
                }

            }

            //   For the relevant range of reads, write each to the file
            //     recreating the original headers with the index meta-data about each read

            for (i = first; i < last; i++)
            {
                int j, len;
                HITS_READ *r;

                int h = i;
                if (SORT)
                    h = readIDs[i];

                r = reads + h;

                len = r->rlen;

                if (SORT)
                {
                    int pidx = getPrologIndexOfRead(dbh, h);
                    if (pidx < 0)
                    {
                        printf("Cannot find read: %d\n", h);
                        fflush(stdout);
                        exit(1);
                    }

                    fprintf(ofile, ">%s", dbh->headers[pidx].readName);
                    fprintf(ofile, " fileID=%d", pidx + 1);
                }
                else
                    fprintf(ofile, ">%s", dbh->headers[f].readName);

                for (j = 0; j < curTracks; j++)
                {
                    track_anno *anno = out_tracks[j]->anno;
                    track_data *data = out_tracks[j]->data;

                    fprintf(ofile, " %s=", out_tracks[j]->name);
                    b = anno[h] / sizeof(track_data);
                    e = anno[h + 1] / sizeof(track_data);

                    while (b < e)
                    {
                        fprintf(ofile, "%d", data[b]);
                        b++;
                        if (b < e)
                            fprintf(ofile, ",");
                    }
                }

                fprintf(ofile, "\n");

                Load_Read(db, h, read, UPPER);

                for (j = 0; j + WIDTH < len; j += WIDTH)
                    fprintf(ofile, "%.*s\n", WIDTH, read + j);
                if (j < len)
                    fprintf(ofile, "%s\n", read + j);
            }

            first = last;

            if (SORT)
                break;
        }
    }

    fclose(dbfile);
    Close_DB(db);

    exit(0);
}
