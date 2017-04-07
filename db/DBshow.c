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

/*******************************************************************************************
 *
 *  Display a specified set of reads of a database in fasta format.
 *
 *  Author:  Gene Myers
 *  Date  :  September 2013
 *  Mod   :  With DB overhaul, made this a routine strictly for printing a selected subset
 *             and created DB2fasta for recreating all the fasta files of a DB
 *  Date  :  April 2014
 *  Mod   :  Added options to display QV streams
 *  Date  :  July 2014
 *
 ********************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

#include "DB.h"
#include "fileUtils.h"
#include "lib/tracks.h"
#include "lib/utils.h"

#ifdef HIDE_FILES
#define PATHSEP "/."
#else
#define PATHSEP "/"
#endif

static void usage()
  {
    fprintf(stderr, "usage: [-unqUQ] [-w <int(80)>] [-x <int(0)>] [-tm <track>] <path:db|dam> [ <reads:FILE> | <reads:range> ... ]\n");
    fprintf(stderr, "options: -n ... DNA sequence is not displayed\n");
    fprintf(stderr, "         -q ... report QV streams\n");
    fprintf(stderr, "         -U ... report DNA sequence in upper case\n");
    fprintf(stderr, "         -Q ... QV streams are reported into a .quiva file\n");
    fprintf(stderr, "         -w ... specify number of characters per line\n");
    fprintf(stderr, "         -x ... filter out reads that are shorter than -x length\n");
    fprintf(stderr, "         -t ... trim reads according to trim track \n");
    fprintf(stderr, "         -m ... report interval track as header and in DNA sequence in defined lower/upper format\n");
  }

extern char *optarg;
extern int optind, opterr, optopt;

#define LAST_READ_SYMBOL   '$'

int main(int argc, char *argv[])
  {
    HITS_DB _db, *db = &_db;
    HITS_TRACK* pacbio_track = NULL;
    HITS_TRACK* seqID_track = NULL;
    HITS_TRACK* source_track = NULL;
    HITS_TRACK* trim_track = NULL;
    FILE *hdrs = NULL;

    int nfiles;
    char **flist = NULL;
    int *findx = NULL;

    int reps, *pts;
    int input_pts;
    Read_Iterator *iter = NULL;
    FILE *input;
    char *trim = NULL;

    int UPPER;
    int DOSEQ, DOQVS, QUIVA, DAM;
    int WIDTH, MINLEN;

    int MMAX, MTOP;
    char **MASK;

    //  Process arguments
      {
        DAM = 0;
        UPPER = 1;
        DOQVS = 0;
        DOSEQ = 1;
        QUIVA = 0;

        WIDTH = 80;
        MINLEN= 0;
        MTOP = 0;
        MMAX = 10;
        MASK = (char **) Malloc(MMAX * sizeof(char *), "Allocating mask track array");
        if (MASK == NULL)
          exit(1);

        int c;
        opterr = 0;

        while ((c = getopt(argc, argv, "nqUQw:m:x:t:")) != -1)
          {
            switch (c)
            {
              case 'n':
                DOSEQ = 0;
                break;
              case 'q':
                DOQVS = 1;
                break;
              case 'U':
                UPPER = 2;
                break;
              case 't':
                trim = optarg;
                break;
              case 'Q':
                QUIVA = 1;
                break;
              case 'w':
                {
                  WIDTH = atoi(optarg);
                  if (WIDTH <= 0)
                    {
                      fprintf(stderr, "[ERROR] invalid line width of %d\n", WIDTH);
                      exit(1);
                    }
                }
                break;
              case 'x':
                {
                  MINLEN = atoi(optarg);
                  if (MINLEN < 0)
                    {
                      fprintf(stderr, "[ERROR] invalid minimum read length of %d\n", MINLEN);
                      exit(1);
                    }
                }
                break;
              case 'm':
                {
                  if (MTOP >= MMAX)
                    {
                      MMAX = 1.2 * MTOP + 10;
                      MASK = (char **) Realloc(MASK, MMAX * sizeof(char *), "Reallocating mask track array");
                      if (MASK == NULL)
                        exit(1);
                    }
                  // ignore pacbio track
                  if (strcasecmp(optarg, TRACK_PACBIO_HEADER) == 0)
                    break;
                  // ignore seq id track
                  if (strcasecmp(optarg, TRACK_SEQID) == 0)
                    break;
                  if (strcasecmp(optarg, TRACK_SOURCE) == 0)
                    break;

                  MASK[MTOP] = optarg;
                  MTOP++;
                }
                break;
              default:
                fprintf(stderr, "[ERROR] Unsupported argument %s\n", argv[optind]);
                usage();
                exit(1);
            }
          }

        if (QUIVA && (!DOSEQ || MTOP > 0))
          {
            fprintf(stderr, "[ERROR] -Q (quiva) format request inconsistent with -n and -m options\n");
            exit(1);
          }
        if (QUIVA)
          DOQVS = 1;

        if (optind + 1 > argc)
          {
            fprintf(stderr, "[ERROR]: Database is required\n");
            usage();
            exit(1);
          }

      }

    //  Open DB or DAM, and if a DAM open also .hdr file

      {
        char *pwd, *root;
        int status;

        status = Open_DB(argv[optind], db);
        if (status < 0)
          exit(1);
        if (status == 1)
          {
            root = Root(argv[optind], ".dam");
            pwd = PathTo(argv[optind]);

            hdrs = Fopen(Catenate(pwd, PATHSEP, root, ".hdr"), "r");
            if (hdrs == NULL)
              exit(1);
            DAM = 1;
            if (QUIVA || DOQVS)
              {
                fprintf(stderr, "%s: -Q and -q options not compatible with a .dam DB\n", Prog_Name);
                exit(1);
              }

            free(root);
            free(pwd);
          }
      }

    //  Load QVs if requested

    if (DOQVS)
      {
        if (Load_QVs(db) < 0)
          {
            fprintf(stderr, "%s: QVs requested, but no .qvs for data base\n", Prog_Name);
            exit(1);
          }
      }

    //  Check tracks and load tracks for DB

      {
        int i, status;

        for (i = 0; i < MTOP; i++)
          {
            status = Check_Track(db, MASK[i]);
            if (status == -2)
              printf("[WARNING] - DBshow: -m%s option given but no track found.\n", MASK[i]);
            else if (status == -1)
              printf("[WARNING] - DBshow: %s track not sync'd with db.\n", MASK[i]);
            else
              track_load(db, MASK[i]);
          }
      }

    //  If not a DAM then get prolog names and index ranges from the .db file
    if (!DAM)
      {
        char *pwd, *root;
        FILE *dstub;
        int i;

        root = Root(argv[optind], ".db");
        pwd = PathTo(argv[optind]);
        if (db->part > 0)
        {
          char* sep = rindex(root, '.');
          *sep = '\0';
        }
        dstub = Fopen(Catenate(pwd, "/", root, ".db"), "r");
        if (dstub == NULL)
          exit(1);
        free(pwd);
        free(root);

        if (fscanf(dstub, DB_NFILE, &nfiles) != 1)
          SYSTEM_ERROR

        flist = (char **) Malloc(sizeof(char *) * nfiles, "Allocating file list");
        findx = (int *) Malloc(sizeof(int *) * (nfiles + 1), "Allocating file index");
        if (flist == NULL || findx == NULL)
          exit(1);

        findx += 1;
        findx[-1] = 0;

        for (i = 0; i < nfiles; i++)
          {
            char prolog[MAX_NAME], fname[MAX_NAME];

            if (fscanf(dstub, DB_FDATA, findx + i, fname, prolog) != 3)
              SYSTEM_ERROR
            if (Check_Track(db, TRACK_PACBIO_HEADER) == 0)
              {
                if ((flist[i] = Strdup(prolog, "Adding to file list")) == NULL)
                  exit(1);
              }
            else
              {
                if ((flist[i] = Strdup(fname, "Adding to file list")) == NULL)
                  exit(1);
              }
          }

        fclose(dstub);

        if (db->part > 0)
          {
            for (i = 0; i < nfiles; i++)
              findx[i] -= db->ufirst;
          }
      }

    // Load Tracks pacbio and seqID if present
      {
        int status;
        status = Check_Track(db, TRACK_PACBIO_HEADER);
        if (status == 0)
          pacbio_track = track_load(db, TRACK_PACBIO_HEADER);

        status = Check_Track(db, TRACK_SEQID);
        if (status == 0)
          seqID_track = track_load(db, TRACK_SEQID);

        status = Check_Track(db, TRACK_SOURCE);
        if (status == 0)
          source_track = track_load(db, TRACK_SOURCE);

        if(trim != NULL)
        {
        	for (trim_track = db->tracks; trim_track != NULL; trim_track = trim_track->next)
              if (strcmp(trim_track->name, trim) == 0)
                break;

        	if(trim_track == NULL)
        		trim_track = track_load(db, trim);
        }
      }

//  Process read index arguments into a list of read ranges

    optind++;
    input_pts = 0;
    if (optind + 1 == argc)
      {
        if (argv[optind][0] != LAST_READ_SYMBOL || argv[optind][1] != '\0')
          {
            char *eptr, *fptr;
            int b, e;
            b = strtol(argv[optind], &eptr, 10);
            if (eptr > argv[optind] && b >= 0)
              {
                if (*eptr == '-')
                  {
                    if (eptr[1] != LAST_READ_SYMBOL || eptr[2] != '\0')
                      {
                        e = strtol(eptr + 1, &fptr, 10);
                        input_pts = (fptr <= eptr + 1 || *fptr != '\0' || e <= 0);
                      }
                  }
                else
                  input_pts = (*eptr != '\0');
              }
            else
              input_pts = 1;
          }
      }

    if (input_pts)
      {
        input = Fopen(argv[optind], "r");
        if (input == NULL)
          exit(1);

        iter = init_read_iterator(input);
      }
    else
      {
        pts = (int *) Malloc(sizeof(int) * 2 * (optind - 1), "Allocating read parameters");
        if (pts == NULL)
          exit(1);

        reps = 0;
        if (argc - optind >= 1)
          {
            int c, b, e;
            char *eptr, *fptr;
            for (c = optind; c < argc; c++)
              {
                if (argv[c][0] == LAST_READ_SYMBOL)
                  {
                    b = db->nreads;
                    eptr = argv[c] + 1;
                  }
                else
                  b = strtol(argv[c], &eptr, 10);
                if (eptr > argv[c])
                  {
                    if (b < 0)
                      {
                        fprintf(stderr, "%s: %d is not a valid index\n", Prog_Name, b);
                        exit(1);
                      }
                    if (*eptr == 0)
                      {
                        pts[reps++] = b;
                        pts[reps++] = b + 1;
                        continue;
                      }
                    else if (*eptr == '-')
                      {
                        if (eptr[1] == LAST_READ_SYMBOL)
                          {
                            e = db->nreads;
                            fptr = eptr + 2;
                          }
                        else
                          e = strtol(eptr + 1, &fptr, 10);
                        if (fptr > eptr + 1 && *fptr == 0 && e > 0)
                          {
                            pts[reps++] = b;
                            pts[reps++] = e;
                            if (b > e)
                              {
                                fprintf(stderr, "%s: Empty range '%s'\n", Prog_Name, argv[c]);
                                exit(1);
                              }
                            continue;
                          }
                      }
                  }
                fprintf(stderr, "%s: argument '%s' is not an integer range\n", Prog_Name, argv[c]);
                exit(1);
              }
          }
        else
          {
            pts[reps++] = 0;
            pts[reps++] = db->nreads;
          }
      }

    //  Display each read (and/or QV streams) in the active DB according to the
    //    range pairs in pts[0..reps) and according to the display options.

      {
        HITS_READ *reads;
        HITS_TRACK *first;
        char *read, **entry;
        int c, b, e, i;
        int hilight, substr;
        int map;
        int (*iscase)(int);
        track_anno *pacbio_anno, *seqID_anno, *source_anno;
        track_data *pacbio_data, *seqID_data, *source_data;

        pacbio_anno = seqID_anno = source_anno = NULL;
        pacbio_data = seqID_data = source_data = NULL;

        int trim_b, trim_e;

        if (pacbio_track)
          {
            pacbio_anno = pacbio_track->anno;
            pacbio_data = pacbio_track->data;
          }

        if (seqID_track)
          {
            seqID_anno = seqID_track->anno;
            seqID_data = seqID_track->data;
          }

        if (source_track)
          {
            source_anno = source_track->anno;
            source_data = source_track->data;
          }

        read = New_Read_Buffer(db);
        if (DOQVS)
          {
            entry = New_QV_Buffer(db);
            first = db->tracks->next;
          }
        else
          {
            entry = NULL;
            first = db->tracks;
          }

        if (UPPER == 1)
          {
            hilight = 'A' - 'a';
            iscase = islower;
          }
        else
          {
            hilight = 'a' - 'A';
            iscase = isupper;
          }

        map = 0;
        reads = db->reads;
        substr = 0;

        c = 0;
        while (1)
          {
            if (input_pts)
              {
                if (next_read(iter))
                  break;
                e = iter->read;
                b = e - 1;
                substr = (iter->beg >= 0);
              }
            else
              {
                if (c >= reps)
                  break;
                b = pts[c];
                e = pts[c + 1];
                if (e > db->nreads)
                  e = db->nreads; // - 1;
                c += 2;
              }

            for (i = b; i < e; i++)
              {
                int len;
                int fst, lst;
                // int flags, qv;
                HITS_READ *r;
                HITS_TRACK *track;

                r = reads + i;

                if(trim_track != NULL)
                {
                	get_trim(db,trim_track, i, &trim_b, &trim_e);
                }
                else
                {
                	trim_b = 0;
                	trim_e = r->rlen;
                }

                len = trim_e - trim_b;

                if(len < MINLEN)
                  continue;

                // flags = r->flags;
                // qv = (flags & DB_QV);
                if (DAM)
                  {
                    char header[MAX_NAME];

                    fseeko(hdrs, r->coff, SEEK_SET);
                    if ( fgets(header, MAX_NAME, hdrs) == NULL )
                    {
                      fprintf(stderr, "ERROR: failed to read header\n");
                      exit(1);
                    }

                    header[strlen(header) - 1] = '\0';
                    printf("%s :: Contig %d[%d]", header, b, len);
                  }
                else
                  {
                    while (i < findx[map - 1])
                      map -= 1;
                    while (i >= findx[map])
                      map += 1;
                    if (QUIVA)
                      printf("@%s_%d_%d", flist[map], i, len);
                    else
                      {
                        if (pacbio_track)
                          {
                            int s, f;
                            s = pacbio_anno[i] / sizeof(track_data);
                            f = pacbio_anno[i + 1] / sizeof(track_data);
                            if (s < f)
                              printf(">%s/%d/%d_%d", flist[map], pacbio_data[s], pacbio_data[s + 1], pacbio_data[s + 2]);
                            else
                              printf(">%s_%d_%d_%d", flist[map], -1, i, len);
                          }
                        else if (seqID_track)
                          {
                            int s, f;
                            s = seqID_anno[i] / sizeof(track_data);
                            f = seqID_anno[i + 1] / sizeof(track_data);
                            if (s < f)

                              printf(">%s_%d_%d_%d", flist[map], seqID_data[s], i, len);
                          }
                        else
                          {
                            printf(">%s_%d_%d_%d", flist[map], -1, i, len);
                          }

                        if(source_track)
                          {
                            int s, f;
                            s = source_anno[i] / sizeof(track_data);
                            f = source_anno[i + 1] / sizeof(track_data);
                            if (s < f)
                              {
                                printf(" %s=%d", TRACK_SOURCE, source_data[s]);
                              }
                          }
                        if(trim_track)
                        {
                        	printf(" trim=%d,%d", trim_b, trim_e);
                        }

                      }
                  }

                if (DOQVS)
                  Load_QVentry(db, i, entry, UPPER);
                if (DOSEQ)
                  Load_Read(db, i, read, UPPER);

                for (track = first; track != NULL; track = track->next)
                  {
                    if (strcmp(track->name, TRACK_PACBIO_HEADER) == 0)
                      continue;

                    if (strcmp(track->name, TRACK_SEQID) == 0)
                      continue;

                    if (strcmp(track->name, TRACK_SOURCE) == 0)
                      continue;

                    if (trim != NULL && strcmp(track->name, trim) == 0)
                      continue;

                    int64 *anno;
                    int *data;
                    int64 s, f, j;
                    int bd, ed, m;

                    anno = (int64 *) track->anno;
                    data = (int *) track->data;

                    s = (anno[i] >> 2);
                    f = (anno[i + 1] >> 2);
                    if (s < f)
                      {
                        for (j = s; j < f; j += 2)
                          {
                            bd = data[j];
                            ed = data[j + 1];
                            if (DOSEQ)
                              for (m = bd; m < ed; m++)
                                if (iscase(read[m]))
                                  read[m] = (char) (read[m] + hilight);
                            if (j == s)
                              printf(" %s=", track->name);
                            if(ed < trim_b || bd > trim_e)
                            	continue;
                            if(bd < trim_b)
                            	bd=trim_e;
                            if(ed > trim_e)
                            	ed = trim_e;
                            if(j+2 < f)
                            	printf("%d,%d,", bd, ed);
                            else
                            	printf("%d,%d", bd, ed);
                          }
                      }
                  }

                printf("\n");

                if (substr)
                  {
                    fst = iter->beg;
                    lst = iter->end;
                  }
                else
                  {
                    fst = trim_b;
                    lst = trim_e;
                  }

                if (QUIVA)
                  {
                    int k;

                    for (k = 0; k < 5; k++)
                      printf("%.*s\n", lst - fst, entry[k] + fst);
                  }
                else
                  {
                    if (DOQVS)
                      {
                        int j, k;

                        printf("\n");
                        for (j = fst; j + WIDTH < lst; j += WIDTH)
                          {
                            if (DOSEQ)
                              printf("%.*s\n", WIDTH, read + j);
                            for (k = 0; k < 5; k++)
                              printf("%.*s\n", WIDTH, entry[k] + j);
                            printf("\n");
                          }
                        if (j < lst)
                          {
                            if (DOSEQ)
                              printf("%.*s\n", lst - j, read + j);
                            for (k = 0; k < 5; k++)
                              printf("%.*s\n", lst - j, entry[k] + j);
                            printf("\n");
                          }
                      }
                    else if (DOSEQ)
                      {
                        int j;
                        for (j = fst; j + WIDTH < lst; j += WIDTH)
                          printf("%.*s\n", WIDTH, read + j);
                        if (j < lst)
                          printf("%.*s\n", lst - j, read + j);
                      }
                  }
              }
          }
      }

    if (input_pts)
      {
        fclose(input);
        free(iter);
      }
    else
      free(pts);

    if (DAM)
      fclose(hdrs);
    else
      {
        int i;

        for (i = 0; i < nfiles; i++)
          free(flist[i]);
        free(flist);
        free(findx - 1);
      }
    Close_DB(db);

    exit(0);
  }
