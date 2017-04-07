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
 *  Adds the given .quiva files to an existing DB "path".  The input files must be added in
 *  the same order as the .fasta files were and have the same root names, e.g. FOO.fasta
 *  and FOO.quiva.  The files can be added incrementally but must be added in the same order  
 *  as the .fasta files.  This is enforced by the program.  With the -l option set the
 *  compression scheme is a bit lossy to get more compression (see the description of dexqv
 *  in the DEXTRACTOR module).
 *
 *  Author:  Gene Myers
 *  Date  :  July 2014
 *
 ********************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#include "DB.h"
#include "QV.h"
#include "fileUtils.h"

#ifdef HIDE_FILES
#define PATHSEP "/."
#else
#define PATHSEP "/"
#endif

static char *Usage = "[-vl] <path:string> ( -f<file> | <input:quiva> ... )";

int main(int argc, char *argv[])
{ FILE      *istub, *quiva, *indx;
  int64      coff;
  int        ofile;
  HITS_DB    db;
  HITS_READ *reads;

  int        VERBOSE;
  int        LOSSY;
  FILE      *IFILE;

  //  Process command line

  { int   i, j, k;
    int   flags[128];

    ARG_INIT("quiva2DB")

    IFILE = NULL;

    j = 1;
    for (i = 1; i < argc; i++)
      if (argv[i][0] == '-')
        switch (argv[i][1])
        { default:
            ARG_FLAGS("vl")
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
    LOSSY   = flags['l'];

    if ((IFILE == NULL && argc <= 2) || (IFILE != NULL && argc != 2))
      { fprintf(stderr,"Usage: %s %s\n",Prog_Name,Usage);
        exit (1);
      }
  }

  //  Open DB stub file and index, load db and read records.  Confirm that the .fasta files
  //    corresponding to the command line .quiva files are in the DB and in order where the
  //    index of the first file is ofile and the index of the first read to be added is ofirst.
  //    Record in coff the current size of the .qvs file in case an error occurs and it needs
  //    to be truncated back to its size at the start.

  { int            i;
    char          *pwd, *root;
    int            nfiles;
    File_Iterator *ng;

    root   = Root(argv[1],".db");
    pwd    = PathTo(argv[1]);
    istub  = Fopen(Catenate(pwd,"/",root,".db"),"r");
    if (istub == NULL)
      exit (1);

    indx  = Fopen(Catenate(pwd,PATHSEP,root,".idx"),"r+");
    if (indx == NULL)
      exit (1);
    if (fread(&db,sizeof(HITS_DB),1,indx) != 1)
      SYSTEM_ERROR

    reads = (HITS_READ *) Malloc(sizeof(HITS_READ)*db.ureads,"Allocating DB index");
    if (reads == NULL)
      exit (1);
    if (fread(reads,sizeof(HITS_READ),db.ureads,indx) != (size_t) (db.ureads))
      SYSTEM_ERROR

    { int   first, last;
      char  prolog[MAX_NAME], fname[MAX_NAME];
      char *core;

      ng = init_file_iterator(argc,argv,IFILE,2);
      if ( ! next_file(ng))
        { fprintf(stderr,"%s: file list is empty!\n",Prog_Name);
          exit (1);
        }
      if (ng->name == NULL) exit (1);

      core = Root(ng->name,".quiva");

      if (fscanf(istub,DB_NFILE,&nfiles) != 1)
        SYSTEM_ERROR
      first = 0;
      for (i = 0; i < nfiles; i++)
        { if (fscanf(istub,DB_FDATA,&last,fname,prolog) != 3)
            SYSTEM_ERROR
          if (strcmp(core,fname) == 0)
            break;
          first = last;
        }
      if (i >= nfiles)
        { fprintf(stderr,"%s: %s.fasta has never been added to DB\n",Prog_Name,core);
          exit (1);
        }

      ofile  = i;
      if (first > 0 && reads[first-1].coff < 0)
        { fprintf(stderr,"%s: Predecessor of %s.quiva has not been added yet\n",Prog_Name,core);
          exit (1);
        }
      if (reads[first].coff >= 0)
        { fprintf(stderr,"%s: %s.quiva has already been added\n",Prog_Name,core);
          exit (1);
        }

      while (next_file(ng))
        { if (ng->name == NULL)
            exit (1);
          core = Root(ng->name,".quiva");
          if (++i >= nfiles)
            { fprintf(stderr,"%s: %s.fasta has never been added to DB\n",Prog_Name,core);
              exit (1);
            }
          if (fscanf(istub,DB_FDATA,&last,fname,prolog) != 3)
            SYSTEM_ERROR
          if (strcmp(core,fname) != 0)
            { fprintf(stderr,"%s: Files not being added in order (expect %s, given %s)",
                             Prog_Name,fname,core);
              exit (1);
            }
        }

      if (ofile == 0)
        quiva = Fopen(Catenate(pwd,PATHSEP,root,".qvs"),"w");
      else
        quiva = Fopen(Catenate(pwd,PATHSEP,root,".qvs"),"r+");
      if (quiva == NULL)
        exit (1);

      fseeko(quiva,0,SEEK_END);
      coff = ftello(quiva);

      free(core);
      free(ng);
    }

    free(root);
    free(pwd);
  }

  //  For each .quiva file, determine its compression scheme in a fast scan and append it to
  //    the .qvs file  Then compress every .quiva entry in the file, appending its compressed
  //    form to the .qvs file as you go and recording the offset in the .qvs in the .coff field
  //    of each read record (*except* the first, that points at the compression scheme immediately
  //    preceding it).  Ensure that the # of .quiva entries matches the # of .fasta entries
  //    in each added file.

  { int            i;
    int            last, cur;
    File_Iterator *ng;

    //  For each .quiva file do:

    rewind(istub);
    if (fscanf(istub,"files = %*d\n") != 0)
      SYSTEM_ERROR

    last = 0;
    for (i = 0; i < ofile; i++)
      if (fscanf(istub,"  %9d %*s %*s\n",&last) != 1)
        SYSTEM_ERROR

    ng  = init_file_iterator(argc,argv,IFILE,2);
    cur = last;
    while (next_file(ng))
      { FILE     *input;
        int64     qpos;
        char     *pwd, *root;
        QVcoding *coding;

        //  Open next .quiva file and create its compression scheme

        pwd  = PathTo(ng->name);
        root = Root(ng->name,".quiva");
        if ((input = Fopen(Catenate(pwd,"/",root,".quiva"),"r")) == NULL)
          goto error;

        if (VERBOSE)
          { fprintf(stderr,"Analyzing '%s' ...\n",root);
            fflush(stderr);
          }

        QVcoding_Scan(input);
        coding = Create_QVcoding(LOSSY);
        coding->prefix = Strdup(".qvs","Allocating header prefix");

        qpos = ftello(quiva);
        Write_QVcoding(quiva,coding);

        //  Then compress and append to the .qvs each compressed QV entry
 
        if (VERBOSE)
          { fprintf(stderr,"Compressing '%s' ...\n",root);
            fflush(stderr);
          }

        rewind(input);
        while (Read_Lines(input,1) > 0)
          { reads[cur++].coff = qpos;
            Compress_Next_QVentry(input,quiva,coding,LOSSY);
            qpos = ftello(quiva);
          }

        if (fscanf(istub,"  %9d %*s %*s\n",&last) != 1)
          SYSTEM_ERROR
        if (last != cur)
          { fprintf(stderr,"%s: Number of reads in %s.quiva doesn't match number in %s.fasta\n",
                           Prog_Name,root,root);
            goto error;
          }

        Free_QVcoding(coding);
        free(root);
        free(pwd);
    }

    free(ng);
  }

  //  Write the db record and read index into .idx and clean up

  rewind(indx);
  fwrite(&db,sizeof(HITS_DB),1,indx);
  fwrite(reads,sizeof(HITS_READ),db.ureads,indx);

  fclose(istub);
  fclose(indx);
  fclose(quiva);

  exit (0);

  //  Error exit:  Either truncate or remove the .qvs file as appropriate.

error:
  if (coff != 0)
    { fseeko(quiva,0,SEEK_SET);
      if (ftruncate(fileno(quiva),coff) < 0)
        SYSTEM_ERROR
    }
  fclose(istub);
  fclose(indx);
  fclose(quiva);
  if (coff == 0)
    { char *root = Root(argv[1],".db");
      char *pwd  = PathTo(argv[1]);
      unlink(Catenate(pwd,PATHSEP,root,".qvs"));
      free(pwd);
      free(root);
     }

  exit (1);
}
