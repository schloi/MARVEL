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
 *  Recreate all the .quiva files that have been loaded into a specified database.
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
#include "QV.h"

#ifdef HIDE_FILES
#define PATHSEP "/."
#else
#define PATHSEP "/"
#endif

static char *Usage = "[-vU] <path:db>";

int main(int argc, char *argv[])
{ HITS_DB    _db, *db = &_db;
  FILE       *dbfile, *quiva;
  int         VERBOSE, UPPER;

  HITS_TRACK* pacbio_track;
  HITS_TRACK* rq_track;

  //  Process arguments

  { int   i, j, k;
    int   flags[128];

    ARG_INIT("DB2quiva")

    j = 1;
    for (i = 1; i < argc; i++)
      if (argv[i][0] == '-')
        { ARG_FLAGS("vU") }
      else
        argv[j++] = argv[i];
    argc = j;

    VERBOSE = flags['v'];
    UPPER   = flags['U'];

    if (argc != 2)
      { fprintf(stderr,"Usage: %s %s\n",Prog_Name,Usage);
        exit (1);
      }
  }

  //  Open db, db stub file, and .qvs file

  { char *pwd, *root;
    int   status;

    status = Open_DB(argv[1],db);
    if (status < 0)
      exit (1);
    if (status == 1)
      { fprintf(stderr,"%s: Cannot be called on a .dam index: %s\n",Prog_Name,argv[1]);
        exit (1);
      }
    if (db->part > 0)
      { fprintf(stderr,"%s: Cannot be called on a block: %s\n",Prog_Name,argv[1]);
        exit (1);
      }

    pwd    = PathTo(argv[1]);
    root   = Root(argv[1],".db");
    dbfile = Fopen(Catenate(pwd,"/",root,".db"),"r");
    quiva  = Fopen(Catenate(pwd,PATHSEP,root,".qvs"),"r");
    free(pwd);
    free(root);
    if (dbfile == NULL || quiva == NULL)
      exit (1);
  }

  // Load Tracks
  {
    pacbio_track = track_load(db, TRACK_PACBIO_HEADER);
    rq_track = track_load(db, TRACK_PACBIO_RQ);
  }

  //  For each file do:

  { HITS_READ  *reads;
    int         f, first, nfiles;
    QVcoding   *coding;
    char      **entry;

    if (fscanf(dbfile,DB_NFILE,&nfiles) != 1)
      SYSTEM_ERROR

    entry = New_QV_Buffer(db);
    reads = db->reads;
    first = 0;
    for (f = 0; f < nfiles; f++)
      { int   i, last;
        FILE *ofile;
        char  prolog[MAX_NAME], fname[MAX_NAME];

        //  Scan db image file line, create .quiva file for writing

        if (reads[first].coff < 0) break;

        if (fscanf(dbfile,DB_FDATA,&last,fname,prolog) != 3)
          SYSTEM_ERROR

        if ((ofile = Fopen(Catenate(".","/",fname,".quiva"),"w")) == NULL)
          exit (1);

        if (VERBOSE)
          { fprintf(stderr,"Creating %s.quiva ...\n",fname);
            fflush(stderr);
          }

        coding = Read_QVcoding(quiva);

        track_anno* rq_anno = rq_track->anno;
        track_data* rq_data = rq_track->data;

        track_anno* pacbio_anno = pacbio_track->anno;
        track_data* pacbio_data = pacbio_track->data;

        //   For the relevant range of reads, write the header for each to the file
        //     and then uncompress and write the quiva entry for each

        for (i = first; i < last; i++)
          { int        b, e, rlen;
            HITS_READ *r;
            int        sequCnt = 0;
            r     = reads + i;
            rlen  = r->rlen;

           fprintf(ofile,"@%s_%d_%d",prolog,sequCnt++,rlen);
           if (rq_track)
              {
                b = rq_anno[i] / sizeof(track_data);
                e = rq_anno[i+1] / sizeof(track_data);

                if(b<e)
                  fprintf(ofile," RQ=0.%3d",rq_data[b]);
              }

            if(pacbio_track)
              {
                b = pacbio_anno[i] / sizeof(track_data);
                e = pacbio_anno[i+1] / sizeof(track_data);
                if(b<e)
                  fprintf(ofile," pacbio=%d,%d,%d", pacbio_data[b], pacbio_data[b+1], pacbio_data[b+2]);
              }
            fprintf(ofile,"\n");

            Uncompress_Next_QVentry(quiva,entry,coding,rlen);

            if (UPPER)
              { char *deltag = entry[1];
                int   j;

                for (j = 0; j < rlen; j++)
                  deltag[j] -= 32;
              }

            for (e = 0; e < 5; e++)
              fprintf(ofile,"%.*s\n",rlen,entry[e]);
          }

        first = last;
      }
  }

  fclose(quiva);
  fclose(dbfile);
  Close_DB(db);

  exit (0);
}
