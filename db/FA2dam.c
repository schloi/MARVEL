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
 *  Add .fasta files to a DB:
 *     Adds the given fasta files in the given order to <path>.db.  If the db does not exist
 *     then it is created.  All .fasta files added to a given data base must have the same
 *     header format and follow Pacbio's convention.  A file cannot be added twice and this
 *     is enforced.  The command either builds or appends to the .<path>.idx and .<path>.bps
 *     files, where the index file (.idx) contains information about each read and their offsets
 *     in the base-pair file (.bps) that holds the sequences where each base is compessed
 *     into 2-bits.  The two files are hidden by virtue of their names beginning with a '.'.
 *     <path>.db is effectively a stub file with given name that contains an ASCII listing
 *     of the files added to the DB and possibly the block partitioning for the DB if DBsplit
 *     has been called upon it.
 *
 *  Author:  Gene Myers
 *  Date  :  May 2013
 *  Modify:  DB upgrade: now *add to* or create a DB depending on whether it exists, read
 *             multiple .fasta files (no longer a stdin pipe).
 *  Date  :  April 2014
 *
 ********************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include "lib/tracks.h"
#include "DB.h"
#include "fileUtils.h"

#include "FA2x.h"

#ifdef HIDE_FILES
#define PATHSEP "/."
#else
#define PATHSEP "/"
#endif

extern char *optarg;
extern int optind, opterr, optopt;

static char *Usage = "[-v] <path:string> ( -f<file> | <input:fasta> ... )";

static char number[128] =
    { 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 2,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 3, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 2,
      0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 3, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
    };


int main(int argc, char *argv[])
{ FILE  *ostub;
  char  *dbname;
  char  *root, *pwd;

  FILE  *bases, *indx, *hdrs;
  int64  boff, hoff;

  int    ifiles, ofiles;
  char **flist;

  HITS_DB db;
  int     ureads;

  int     VERBOSE;
  FILE   *IFILE;

  CreateContext ctx;
  bzero(&ctx, sizeof(CreateContext));

  ctx.db = &db;

  //   Process command line

  { int   i, j, k;
    int   flags[128];

    ARG_INIT("fasta2DAM")

    IFILE = NULL;

    j = 1;
    for (i = 1; i < argc; i++)
      if (argv[i][0] == '-')
        switch (argv[i][1])
        { default:
            ARG_FLAGS("v")
            break;
          case 'f':
            IFILE = fopen(argv[i]+2,"r");
            if (IFILE == NULL)
              { fprintf(stderr,"%s: Cannot open file of inputs '%s'\n",Prog_Name,argv[i]+2);
                exit (1);
              }
            break;
        }
      else
        argv[j++] = argv[i];
    argc = j;

    VERBOSE = flags['v'];

    if ((IFILE == NULL && argc <= 2) || (IFILE != NULL && argc != 2))
      { fprintf(stderr,"Usage: %s %s\n",Prog_Name,Usage);
        exit (1);
      }
  }

  //  Try to open DB file, if present then adding to DB, otherwise creating new DB.  Set up
  //  variables as follows:
  //    dbname = full name of map index = <pwd>/<root>.dam
  //    ostub  = new image of db file (will overwrite old image at end)
  //    bases  = .bps file positioned for appending
  //    indx   = .idx file positioned for appending
  //    ureads = # of reads currently in db
  //    boff   = offset in .bps at which to place next sequence
  //    hoff   = offset in .hdr at which to place next header prefix
  //    ifiles = # of .fasta files to add
  //    ofiles = # of .fasta files added so far
  //    flist  = [0..ifiles] list of file names (root only) added to db so far

  root   = Root(argv[1],".dam");
  pwd    = PathTo(argv[1]);
  dbname = Strdup(Catenate(pwd,"/",root,".dam"),"Allocating map index name");
  if (dbname == NULL)
    exit (1);

  if (IFILE == NULL)
    ifiles = argc-2;
  else
    { File_Iterator *ng;

      ifiles = 0;
      ng = init_file_iterator(argc,argv,IFILE,2);
      while (next_file(ng))
        ifiles += 1;
      free(ng);
    }
  ofiles = 0;

  bases = Fopen(Catenate(pwd,PATHSEP,root,".bps"),"w");
  indx  = Fopen(Catenate(pwd,PATHSEP,root,".idx"),"w");
  hdrs  = Fopen(Catenate(pwd,PATHSEP,root,".hdr"),"w");
  if (bases == NULL || indx == NULL || hdrs == NULL)
    exit (1);

  flist  = (char **) Malloc(sizeof(char *)*ifiles,"Allocating file list");
  fwrite(&db,sizeof(HITS_DB),1,indx);

  ureads  = 0;
  boff    = 0;
  hoff    = 0;

  ostub  = Fopen(dbname,"w+");
  if (ostub == NULL)
    exit (1);

  fprintf(ostub,DB_NFILE,argc-2);

  { int            maxlen;
    int64          totlen, count[4];
    int            rmax;
    HITS_READ      prec;
    char          *read;
    int            c;
    File_Iterator *ng;

    //  Buffer for accumulating .fasta sequence over multiple lines

    rmax  = MAX_NAME + 60000;
    read  = (char *) Malloc(rmax+1,"Allocating line buffer");
    if (read == NULL)
      goto error;

    totlen = 0;              //  total # of bases in new .fasta files
    maxlen = 0;              //  longest read in new .fasta files
    for (c = 0; c < 4; c++)  //  count of acgt in new .fasta files
      count[c] = 0;

    //  For each .fasta file do:

    ng = init_file_iterator(argc,argv,IFILE,2);
    while (next_file(ng))
      { FILE *input;
        char *path, *core;
        int   nline, eof, rlen;

        if (ng->name == NULL) goto error;

        //  Open it: <path>/<core>.fasta, check that core is not too long,
        //           and checking that it is not already in flist.

        path  = PathTo(ng->name);
        core  = Root(ng->name,".fasta");
        if ((input = Fopen(Catenate(path,"/",core,".fasta"),"r")) == NULL)
          goto error;
        free(path);

        { int j;

          for (j = 0; j < ofiles; j++)
            if (strcmp(core,flist[j]) == 0)
              { fprintf(stderr,"%s: File %s.fasta is already in database %s.db\n",
                               Prog_Name,core,Root(argv[1],".db"));
                goto error;
              }
        }

        //  Get the header of the first line.  If the file is empty skip.

        rlen  = 0;
        nline = 1;
        eof   = (fgets(read,MAX_NAME,input) == NULL);
        if (eof || strlen(read) < 1)
          { fprintf(stderr,"Skipping '%s', file is empty!\n",core);
            fclose(input);
            free(core);
            continue;
          }

        //   Add the file name to flist

        if (VERBOSE)
          { fprintf(stderr,"Adding '%s' ...\n",core);
            fflush(stderr);
          }
        flist[ofiles++] = core;

        // Check that the first line has PACBIO format, and record prolog in 'prolog'.

        if (read[strlen(read)-1] != '\n')
          { fprintf(stderr,"File %s.fasta, Line 1: Fasta line is too long (> %d chars)\n",
                           core,MAX_NAME-2);
            goto error;
          }
        if (!eof && read[0] != '>')
          { fprintf(stderr,"File %s.fasta, Line 1: First header in fasta file is missing\n",core);
            goto error;
          }

        //  Read in all the sequences until end-of-file

        { int i, x, n;

          while (!eof)
            { int hlen;

              read[rlen] = '>';
              hlen = strlen(read+rlen);
              fwrite(read+rlen,1,hlen,hdrs);

              rlen  = 0;
              while (1)
                { eof = (fgets(read+rlen,MAX_NAME,input) == NULL);
                  nline += 1;
                  x = strlen(read+rlen)-1;
                  if (read[rlen+x] != '\n')
                    { fprintf(stderr,"File %s.fasta, Line %d:",core,nline);
                      fprintf(stderr," Fasta line is too long (> %d chars)\n",MAX_NAME-2);
                      goto error;
                    }
                  if (eof || read[rlen] == '>')
                    break;
                  rlen += x;
                  if (rlen + MAX_NAME > rmax)
                    { rmax = ((int) (1.2 * rmax)) + 1000 + MAX_NAME;
                      read = (char *) realloc(read,rmax+1);
                      if (read == NULL)
                        { fprintf(stderr,"File %s.fasta, Line %d:",core,nline);
                          fprintf(stderr," Out of memory (Allocating line buffer)\n");
                          goto error;
                        }
                    }
                }
              read[rlen] = '\0';

              n = 0;
              i = -1;
              while (i < rlen)
                { int pbeg, plen, clen;
                  printf("i: %d\n",i);
                  while (i < rlen)
                    if (number[(int) read[++i]] < 4)
                      break;

                  if (i >= rlen) break;


                  pbeg = i;

                  add_to_track(&ctx, find_track(&ctx, TRACK_SCAFFOLD), ureads, n++);
                  add_to_track(&ctx, find_track(&ctx, TRACK_SCAFFOLD), ureads, pbeg);
                  prec.boff   = boff;
                  prec.coff   = hoff;
                  prec.flags  = DB_BEST;
                  while (i < rlen)
                    { x = number[(int) read[i]];
                      if (x >= 4) break;
                      count[x] += 1;
                      read[i++] = (char) x;
                    }
                  prec.rlen = plen = i-pbeg;
                  ureads += 1;
                  totlen += plen;
                  if (plen > maxlen)
                    maxlen = plen;

                  Compress_Read(plen,read+pbeg);
                  clen = COMPRESSED_LEN(plen);
                  fwrite(read+pbeg,1,clen,bases);
                  boff += clen;

                  fwrite(&prec,sizeof(HITS_READ),1,indx);
                }
              hoff += hlen;
            }

          fprintf(ostub,DB_FDATA,ureads,core,core);

          fclose(input);
        }
      }

    //  Update relevant fields in db record

    db.ureads = ureads;
//    db.treads = ureads;
    for (c = 0; c < 4; c++)
      db.freq[c] = (float) ((1.*count[c])/totlen);
    db.totlen = totlen;
    db.maxlen = maxlen;
//    db.cutoff = -1;
  }

  rewind(indx);
  fwrite(&db,sizeof(HITS_DB),1,indx);   //  Write the finalized db record into .idx

  write_tracks(&ctx, argv[optind]);

  free_tracks(&ctx);

  fclose(ostub);
  fclose(indx);
  fclose(bases);
  fclose(hdrs);

  exit (0);

  //  Error exit:  Remove the .idx, .bps, and .dam files

error:
  fclose(ostub);
  fclose(indx);
  fclose(hdrs);
  fclose(bases);
  unlink(Catenate(pwd,PATHSEP,root,".idx"));
  unlink(Catenate(pwd,PATHSEP,root,".bps"));
  unlink(Catenate(pwd,PATHSEP,root,".hdr"));
  unlink(Catenate(pwd,"/",root,".dam"));

  exit (1);
}
