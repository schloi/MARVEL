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
 *  Recreate all the .fasta files that are in a specified DAM.
 *
 *  Author:  Gene Myers
 *  Date  :  May 2014
 *
 ********************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "lib/tracks.h"
#include "DB.h"

#ifdef HIDE_FILES
#define PATHSEP "/."
#else
#define PATHSEP "/"
#endif

static char *Usage = "[-vU] [-w<int(80)>] <path:dam>";

int main(int argc, char *argv[])
{ HITS_DB    _db, *db = &_db;
  FILE       *dbfile, *hdrs;
  int         nfiles;
  int         VERBOSE, UPPER, WIDTH;

  HITS_TRACK* scaffolds_track;

  //  Process arguments

  { int   i, j, k;
    int   flags[128];
    char *eptr;

    ARG_INIT("DAM2fasta")

    WIDTH = 80;

    j = 1;
    for (i = 1; i < argc; i++)
      if (argv[i][0] == '-')
        switch (argv[i][1])
        { default:
            ARG_FLAGS("vU")
            break;
          case 'w':
            ARG_NON_NEGATIVE(WIDTH,"Line width")
            break;
        }
      else
        argv[j++] = argv[i];
    argc = j;

    UPPER   = 1 + flags['U'];
    VERBOSE = flags['v'];

    if (argc != 2)
      { fprintf(stderr,"Usage: %s %s\n",Prog_Name,Usage);
        exit (1);
      }
  }

  //  Open db

  { int   status;

    status = Open_DB(argv[1],db);
    if (status < 0)
      exit (1);
    if (status == 0)
      { fprintf(stderr,"%s: Cannot be called on a .db: %s\n",Prog_Name,argv[1]);
        exit (1);
      }
    if (db->part > 0)
      { fprintf(stderr,"%s: Cannot be called on a block: %s\n",Prog_Name,argv[1]);
        exit (1);
      }
  }

  // Load Track
  {
    scaffolds_track = track_load(db, TRACK_SCAFFOLD);
    if(scaffolds_track == NULL)
    {
      fprintf(stderr,"%s: Cannot read scaffold track: %s\n",Prog_Name,argv[1]);
      exit (1);
    }
  }


  { char *pwd, *root;

    pwd    = PathTo(argv[1]);
    root   = Root(argv[1],".dam");
    dbfile = Fopen(Catenate(pwd,"/",root,".dam"),"r");
    hdrs   = Fopen(Catenate(pwd,PATHSEP,root,".hdr"),"r");
    free(pwd);
    free(root);
    if (dbfile == NULL || hdrs == NULL)
      exit (1);
  }

  //  nfiles = # of files in data base

  if (fscanf(dbfile,DB_NFILE,&nfiles) != 1)
    SYSTEM_ERROR

  //  For each file do:

  { HITS_READ  *reads;
    char       *read;
    int         f, first;
    char        nstring[WIDTH+1];
    int         b,e;
    track_anno *scaf_anno;
    track_data *scaf_data;

    if(scaffolds_track)
    {
      scaf_anno = scaffolds_track->anno;
      scaf_data = scaffolds_track->data;
    }


    if (UPPER == 2)
      for (f = 0; f < WIDTH; f++)
        nstring[f] = 'N';
    else
      for (f = 0; f < WIDTH; f++)
        nstring[f] = 'n';
    nstring[WIDTH] = '\0';

    reads = db->reads;
    read  = New_Read_Buffer(db);
    first = 0;
    for (f = 0; f < nfiles; f++)
      { int   i, last, wpos;
        FILE *ofile;
        char  prolog[MAX_NAME], fname[MAX_NAME], header[MAX_NAME];

        //  Scan db image file line, create .fasta file for writing

        if (fscanf(dbfile,DB_FDATA,&last,fname,prolog) != 3)
          SYSTEM_ERROR

        if ((ofile = Fopen(Catenate(".","/",fname,".fasta"),"w")) == NULL)
          exit (1);

        if (VERBOSE)
          { fprintf(stderr,"Creating %s.fasta ...\n",fname);
            fflush(stdout);
          }

        //   For the relevant range of reads, write each to the file
        //     recreating the original headers with the index meta-data about each read

        wpos        = 0;
        int pfpulse = 0;

        for (i = first; i < last; i++)
          { int        j, len, nlen, w;

            HITS_READ *r;

            r     = reads + i;
            len   = r->rlen;


            if(scaffolds_track)
            {
              b = scaf_anno[i] / sizeof(track_data);
              e = scaf_anno[i+1] / sizeof(track_data);

              if(b<e)
              {
                int origin = scaf_data[b];
                int fpulse = scaf_data[b+1];

                if (origin == 0)
                { if (i != first && wpos != 0)
                  { fprintf(ofile,"\n");
                    wpos = 0;
                  }
                  fseeko(hdrs,r->coff,SEEK_SET);
                  if (fgets(header,MAX_NAME,hdrs) == NULL)
                  {
                    fprintf(stderr, "failed to read header\n");
                    exit(1);
                  }

                  fputs(header,ofile);
                }

                if (fpulse != 0)
                { if (origin != 0)
                    nlen = fpulse - (pfpulse + reads[i-1].rlen);
                  else
                    nlen = fpulse;

                  for (j = 0; j+(w = WIDTH-wpos) <= nlen; j += w)
                  { fprintf(ofile,"%.*s\n",w,nstring);
                    wpos = 0;
                  }
                  if (j < nlen)
                  { fprintf(ofile,"%.*s",nlen-j,nstring);
                    if (j == 0)
                      wpos += nlen;
                    else
                      wpos = nlen-j;
                  }
                }

                pfpulse = fpulse;
                }
                b+=2;
              }

            Load_Read(db,i,read,UPPER);

            for (j = 0; j+(w = WIDTH-wpos) <= len; j += w)
              { fprintf(ofile,"%.*s\n",w,read+j);
                wpos = 0;
              }
            if (j < len)
              { fprintf(ofile,"%s",read+j);
                if (j == 0)
                  wpos += len;
                else
                  wpos = len-j;
              }
          }
        if (wpos > 0)
          fprintf(ofile,"\n");

        first = last;
      }
  }

  fclose(hdrs);
  fclose(dbfile);
  Close_DB(db);

  exit (0);
}
