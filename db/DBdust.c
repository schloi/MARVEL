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
 *  My implementation of the SDUST algorithm (Morgulis et al., JCB 13, 5 (2006), 1028-1040)
 *
 *  Author:  Gene Myers
 *  Date  :  September 2013
 *  Mod   :  Is now incremental
 *  Date  :  April 2014
 *
 ********************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <unistd.h>

#include "DB.h"

#ifdef HIDE_FILES
#define PATHSEP "/."
#else
#define PATHSEP "/"
#endif

#undef DEBUG

#ifdef DEBUG

static int Caps[4] = { 'A', 'C', 'G', 'T' };
static int Lowr[4] = { 'a', 'c', 'g', 't' };

#endif
static void usage()
  {
    fprintf(stderr, "usage: [-b] [-w <int(64)>] [-t<double(2.)>] [-m<int(10)>] <path:db|dam>\n");
    fprintf(stderr, "options: -b ... biased, i.e. take into account the frequency of a given base\n");
    fprintf(stderr, "         -w ... scan window size\n");
    fprintf(stderr, "         -t ... thershold for being a low complexity interval\n");
    fprintf(stderr, "         -m ... intervals greater than -m bases are reported\n");
  }

extern char *optarg;
extern int optind, opterr, optopt;

typedef struct _cand
  { struct _cand *next;
    struct _cand *prev;
    int           beg;
    int           end;
    double        score;
  } Candidate;

int main(int argc, char *argv[])
{ HITS_DB   _db, *db = &_db;
  FILE      *afile, *dfile;
  int64      indx;
  int        nreads;
  int       *mask;
  Candidate *cptr;

  int        WINDOW=64;
  double     THRESH=2.;
  int        MINLEN=9;
  int        BIASED=0;

  // parse arguments
  {
    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "bw:t:m:")) != -1)
      {
        switch (c)
        {
          case 'b':
            BIASED = 1;
            break;
          case 'w':
            {
                WINDOW = atoi(optarg);
                if (WINDOW <= 0)
                {
                  fprintf(stderr,"[ERROR] Window must be positive (%d)\n",WINDOW);
                  exit(1);
                }
            } 
            break;
          case 't':
            {
                THRESH = atof(optarg);
                if (THRESH <= 0.)
                {
                  fprintf(stderr,"[ERROR] Threshold must be positive (%f)\n",THRESH);
                  exit(1);
                }
            }
            break;
          case 'm':
            {
                MINLEN = atof(optarg) - 1;
                if (MINLEN <= 0)
                {
                  fprintf(stderr,"[ERROR] Minimum hit size must be positive (%d)\n",MINLEN);
                  exit(1);
                }
            }
            break;

          default:
            fprintf(stderr,"Unsupported option: %s\n", argv[optind]);
            usage();
            exit(1);
        }
      }
      if (optind + 1 > argc)
      { fprintf(stderr,"[ERROR] DB is required!\n");
        usage();
        exit (1);
      }
  }

  //  Open .db or .dam

  { int status;

    status = Open_DB(argv[optind],db);
    if (status < 0)
      exit (1);
  }

  mask = (int *) Malloc((db->maxlen+1)*sizeof(int),"Allocating mask vector");
  cptr = (Candidate *) Malloc((WINDOW+1)*sizeof(Candidate),"Allocating candidate vector");
  if (mask == NULL || cptr == NULL)
    exit (1);

  { char *pwd, *root, *fname;
    int   size;

    pwd   = PathTo(argv[optind]);
    root  = Root(argv[optind],".db");
    size  = 8;

    fname = Catenate(pwd,PATHSEP,root,".dust.anno");
    if ((afile = fopen(fname,"r+")) == NULL || db->part > 0)
      { if (afile != NULL)
          fclose(afile);
        afile = Fopen(fname,"w");
        dfile = Fopen(Catenate(pwd,PATHSEP,root,".dust.data"),"w");
        if (dfile == NULL || afile == NULL)
          exit (1);
        fwrite(&(db->nreads),sizeof(int),1,afile);
        fwrite(&size,sizeof(int),1,afile);
        nreads = 0;
        indx = 0;
        fwrite(&indx,sizeof(int64),1,afile);
      }
    else
      { dfile = Fopen(Catenate(pwd,PATHSEP,root,".dust.data"),"r+");
        if (dfile == NULL)
          exit (1);
        if (fread(&nreads,sizeof(int),1,afile) != 1)
          SYSTEM_ERROR
        if (nreads >= db->nreads)
          { fclose(afile);
            fclose(dfile);
            exit(0);
          }
        fseeko(afile,0,SEEK_SET);
        fwrite(&(db->nreads),sizeof(int),1,afile);
        fwrite(&size,sizeof(int),1,afile);
        fseeko(afile,0,SEEK_END);
        fseeko(dfile,0,SEEK_END);
        indx = ftello(dfile);
      }

    free(pwd);
    free(root);
  }

  { int       *mask1;
    char      *read, *lag2;
    int        wcount[64], lcount[64];
    Candidate *aptr;
    double     skew[64], thresh2r;
    int        thresh2i;
    int        i;

    read = New_Read_Buffer(db);
    lag2 = read-2;

    mask1 = mask+1;
    *mask = -2;

    aptr  = cptr+1;
    for (i = 1; i < WINDOW; i++)
      cptr[i].next = aptr+i;
    cptr[WINDOW].next = NULL;

    cptr->next = cptr->prev = cptr;
    cptr->beg  = -2;

    thresh2r = 2.*THRESH;
    thresh2i = (int) ceil(thresh2r);

    if (BIASED)
      { int a, b, c, p;

        p = 0;
        for (a = 0; a < 4; a++)
         for (b = 0; b < 4; b++)
          for (c = 0; c < 4; c++)
            skew[p++] = .015625 / (db->freq[a]*db->freq[b]*db->freq[c]);
      }

    for (i = nreads; i < db->nreads; i++)
      { Candidate *lptr, *jptr;
        int       *mtop;
        double     mscore;
        int        len;
        int        wb, lb;
        int        j, c, d;

        len = db->reads[i].rlen;	  //  Fetch read
        Load_Read(db,i,read,0);

        c = (read[0] << 2) | read[1];     //   Convert to triple codes
        for (j = 2; j < len; j++)
          { c = ((c << 2) & 0x3f) | read[j];
            lag2[j] = (char) c;
          }
        len -= 2;

        for (j = 0; j < 64; j++)		//   Setup counter arrays
          wcount[j] = lcount[j] = 0;

        mtop = mask;                      //   The dust algorithm
        lb   = wb   = -1;

        if (BIASED)

          { double lsqr, wsqr, trun;      //   Modification for high-compositional bias

            wsqr = lsqr = 0.;
            for (j = 0; j < len; j++)
              { c = read[j];

#define ADDR(e,cnt,sqr)	 sqr += (cnt[e]++) * skew[e];

#define DELR(e,cnt,sqr)	 sqr -= (--cnt[e]) * skew[e];

#define WADDR(e) ADDR(e,wcount,wsqr)
#define WDELR(e) DELR(e,wcount,wsqr)
#define LADDR(e) ADDR(e,lcount,lsqr)
#define LDELR(e) DELR(e,lcount,lsqr)

                if (j > WINDOW-3)
                  { d = read[++wb];
                    WDELR(d)
                  }
                WADDR(c)

                if (lb < wb)
                  { d = read[++lb];
                    LDELR(d)
                  }
                trun  = (lcount[c]++) * skew[c];
                lsqr += trun;
                if (trun >= thresh2r)
                  { while (lb < j)
                      { d = read[++lb];
                        LDELR(d)
                        if (d == c) break;
                      }
                  }

                jptr = cptr->prev;
                if (jptr != cptr && jptr->beg <= wb)
                  { c = jptr->end + 2;
                    if (*mtop+1 >= jptr->beg)
                      { if (*mtop < c)
                          *mtop = c;
                      }
                    else
                      { *++mtop = jptr->beg;
                        *++mtop = c;
                      }
                    lptr = jptr->prev;
                    cptr->prev = lptr;
                    lptr->next = cptr;
                    jptr->next = aptr;
                    aptr = jptr;
                  }

                if (wsqr <= lsqr*THRESH) continue;

                jptr   = cptr->next;
                lptr   = cptr;
                mscore = 0.;
                for (c = lb; c > wb; c--)
                  { d = read[c];
                    LADDR(d)
                    if (lsqr >= THRESH * (j-c))
                      { for ( ; jptr->beg >= c; jptr = (lptr = jptr)->next)
                          if (jptr->score > mscore)
                            mscore = jptr->score;
                        if (lsqr >= mscore * (j-c))
                          { mscore = lsqr / (j-c);
                            if (lptr->beg == c)
                              { lptr->end   = j;
                                lptr->score = mscore;
                              }
                            else
                              { aptr->beg   = c;
                                aptr->end   = j;
                                aptr->score = mscore;
                                aptr->prev  = lptr;
                                lptr = lptr->next = aptr;
                                aptr = aptr->next;
                                jptr->prev = lptr;
                                lptr->next = jptr;
                              }
                          }
                      }
                  }

                for (c++; c <= lb; c++)
                  { d = read[c];
                    LDELR(d)
                  }
              }
          }

        else

          { int lsqr, wsqr, trun;                 //  Algorithm for GC-balanced sequences

            wsqr = lsqr = 0;
            for (j = 0; j < len; j++)
              { c = read[j];

#define ADDI(e,cnt,sqr)	 sqr += (cnt[e]++);

#define DELI(e,cnt,sqr)	 sqr -= (--cnt[e]);

#define WADDI(e) ADDI(e,wcount,wsqr)
#define WDELI(e) DELI(e,wcount,wsqr)
#define LADDI(e) ADDI(e,lcount,lsqr)
#define LDELI(e) DELI(e,lcount,lsqr)

                if (j > WINDOW-3)
                  { d = read[++wb];
                    WDELI(d)
                  }
                WADDI(c)

                if (lb < wb)
                  { d = read[++lb];
                    LDELI(d)
                  }
                trun  = lcount[c]++;
                lsqr += trun;
                if (trun >= thresh2i)
                  { while (lb < j)
                      { d = read[++lb];
                        LDELI(d)
                        if (d == c) break;
                      }
                  }

                jptr = cptr->prev;
                if (jptr != cptr && jptr->beg <= wb)
                  { c = jptr->end + 2;
                    if (*mtop+1 >= jptr->beg)
                      { if (*mtop < c)
                          *mtop = c;
                      }
                    else
                      { *++mtop = jptr->beg;
                        *++mtop = c;
                      }
                    lptr = jptr->prev;
                    cptr->prev = lptr;
                    lptr->next = cptr;
                    jptr->next = aptr;
                    aptr = jptr;
                  }

                if (wsqr <= lsqr*THRESH) continue;

                jptr   = cptr->next;
                lptr   = cptr;
                mscore = 0.;
                for (c = lb; c > wb; c--)
                  { d = read[c];
                    LADDI(d)
                    if (lsqr >= THRESH * (j-c))
                      { for ( ; jptr->beg >= c; jptr = (lptr = jptr)->next)
                          if (jptr->score > mscore)
                            mscore = jptr->score;
                        if (lsqr >= mscore * (j-c))
                          { mscore = (1. * lsqr) / (j-c);
                            if (lptr->beg == c)
                              { lptr->end   = j;
                                lptr->score = mscore;
                              }
                            else
                              { aptr->beg   = c;
                                aptr->end   = j;
                                aptr->score = mscore;
                                aptr->prev  = lptr;
                                lptr = lptr->next = aptr;
                                aptr = aptr->next;
                                jptr->prev = lptr;
                                lptr->next = jptr;
                              }
                          }
                      }
                  }

                for (c++; c <= lb; c++)
                  { d = read[c];
                    LDELI(d)
                  }
              }
          }

        while ((jptr = cptr->prev) != cptr)
          { c = jptr->end + 2;
            if (*mtop+1 >= jptr->beg)
              { if (*mtop < c)
                  *mtop = c;
              }
            else
              { *++mtop = jptr->beg;
                *++mtop = c;
              }
            cptr->prev = jptr->prev;
            jptr->prev->next = cptr;
            jptr->next = aptr;
            aptr = jptr;
          }

        { int *jtop, ntop;

          ntop = 0;
          for (jtop = mask1; jtop < mtop; jtop += 2)
            if (jtop[1] - jtop[0] >= MINLEN)
              { mask[++ntop] = jtop[0];
                mask[++ntop] = jtop[1]+1;
              }
          mtop  = mask + ntop;
          indx += ntop*sizeof(int);
          fwrite(&indx,sizeof(int64),1,afile);
          fwrite(mask1,sizeof(int),ntop,dfile);
        }

#ifdef DEBUG

        { int *jtop;

          printf("\nREAD %d\n",i);
          for (jtop = mask1; jtop < mtop; jtop += 2)
            printf(" [%5d,%5d]\n",jtop[0],jtop[1]);

          Load_Read(db,i,read,0);

          jtop = mask1;
          for (c = 0; c < len; c++)
            { while (jtop < mtop && c > jtop[1])
                jtop += 2;
              if (jtop < mtop && c >= *jtop)
                printf("%c",Caps[(int) read[c]]);
              else
                printf("%c",Lowr[(int) read[c]]);
              if ((c%80) == 79)
                printf("\n");
            }
          printf("\n");
        }

#endif
      }
  }

  fclose(afile);
  fclose(dfile);

  Close_DB(db);

  exit (0);
}
