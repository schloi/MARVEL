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
 *  Synthetic DNA shotgun dataset simulator
 *     Generate a fake genome of size genlen*1Mb long, that has an AT-bias of -b.  Then
 *     sample reads of mean length -m from a log-normal length distribution with
 *     standard deviation -s, but ignore reads of length less than -x.  Collect enough
 *     reads to cover the genome -c times.   Introduce -e fraction errors into each
 *     read where the ratio of insertions, deletions, and substitutions are set by
 *     defined constants INS_RATE and DEL_RATE within generate.c.  One can also control
 *     the rate at which reads are picked from the forward and reverse strands by setting
 *     the defined constant FLIP_RATE.
 *
 *     The -r parameter seeds the random number generator for the generation of the genome
 *     so that one can reproducbile produce the same underlying genome to sample from.  If
 *     missing, then the job id of the invocation seeds the generator.  The output is sent
 *     to the standard output (i.e. it is a pipe).  The output is in fasta format (i.e. it is
 *     a UNIX pipe).  The output is in Pacbio .fasta format suitable as input to fasta2DB.
 *
 *     The -M option requests that the coordinates from which each read has been sampled are
 *     written to the indicated file, one line per read, ASCII encoded.  This "map" file
 *     essentially tells one where every read belongs in an assembly and is very useful for
 *     debugging and testing purposes.  If a read pair is say b,e then if b < e the read was
 *     sampled from [b,e] in the forward direction, and from [e,b] in the reverse direction
 *     otherwise.
 *
 *  Author:  Gene Myers
 *  Date  :  July 2013
 *  Mod   :  April 2014 (made independent of "mylib")
 *
 ********************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "DB.h"

static char *Usage[] = { "<genlen:double> [-c<double(20.)>] [-b<double(.5)>] [-r<int>]",
                         "                [-m<int(10000)>]  [-s<int(2000)>]  [-x<int(4000)>]",
                         "                [-e<double(.15)>] [-M<file>]"
                       };

static int    GENOME;     // -g option * 1Mbp
static double COVERAGE;   // -c option
static double BIAS;       // -b option
static int    HASR = 0;   // -r option is set?
static int    SEED;       // -r option
static int    RMEAN;      // -m option
static int    RSDEV;      // -s option
static int    RSHORT;     // -x option
static double ERROR;      // -e option
static FILE  *MAP;        // -M option

#define INS_RATE  .73333  // insert rate
// #define DEL_RATE  .20000  // deletion rate
#define IDL_RATE  .93333  // insert + delete rate
#define FLIP_RATE .5      // orientation rate (equal)

//  Generate a random 4 letter string of length *len* with every letter having equal probability.

static char *random_genome()
{ char  *seq;
  int    i;
  double x, PRA, PRC, PRG;

  PRA = BIAS/2.;
  PRC = (1.-BIAS)/2. + PRA;
  PRG = (1.-BIAS)/2. + PRC;

  if (HASR)
    srand48(SEED);
  else
    srand48(getpid());

  if ((seq = (char *) Malloc(GENOME+1,"Allocating genome sequence")) == NULL)
    exit (1);
  for (i = 0; i < GENOME; i++)
    { x = drand48();
      if (x < PRA)
        seq[i] = 0;
      else if (x < PRC)
        seq[i] = 1;
      else if (x < PRG)
        seq[i] = 2;
      else
        seq[i] = 3;
    }
  seq[GENOME] = 4;
  return (seq);
}

//  Complement (in the DNA sense) string *s*.

static void complement(int elen, char *s)
{ char *t;
  int   c;

  t = s + (elen-1);
  while (s <= t)
    { c = *s;
      *s = (char) (3-*t);
      *t = (char) (3-c);
      s += 1;
      t -= 1;
    }
}

#define UNORM_LEN 60000
#define UNORM_MAX   6.0

static double unorm_table[UNORM_LEN+1];  // Upper half of cdf of N(0,1)
static double unorm_scale;

static void init_unorm()
{ double del, sum, x;
  int    i;

  unorm_scale = del = UNORM_MAX / UNORM_LEN;

  sum = 0;                            // Integrate pdf, x >= 0 half only.
  for (i = 0; i < UNORM_LEN; i++)
    { x = i * del;
      unorm_table[i] = sum;
      sum += exp(-.5*x*x) * del;
    }
  unorm_table[UNORM_LEN] = sum;

                /* Normalize cdf */
  sum *= 2.;
  for (i = 0; i < UNORM_LEN; i++)
    unorm_table[i] /= sum;
  unorm_table[UNORM_LEN] = 1.;

#ifdef DEBUG
  printf("Truncated tail is < %g\n",
          exp(-.5*UNORM_MAX*UNORM_MAX)/(sum*(1.-exp(-UNORM_MAX))) );
  printf("Diff between last two entries is %g\n",.5-unorm_table[UNORM_LEN-1]);

  printf("\n  CDF:\n");
  for (i = 0; i <= UNORM_LEN; i += 100)
    printf("%6.2f: %10.9f\n",i*del,unorm_table[i]);
#endif
}

static int bin_search(int len, double *tab, double y)
{ int l, m, r;

  // Searches tab[0..len] for min { r : y < tab[r] }.
  //   Assumes y < 1, tab[0] = 0 and tab[len] = 1.
  //   So returned index is in [1,len].

  l = 0;
  r = len;
  while (l < r)
    { m = (l+r) >> 1;
      if (y < tab[m])
        r = m;
      else
        l = m+1;
    }
  return (r);
}

static double sample_unorm(double x)
{ double y;
  int    f;

  if (x >= .5)  // Map [0,1) random var to upper-half of cdf */
    y = x-.5;
  else
    y = .5-x;

  f = bin_search(UNORM_LEN,unorm_table,y);    // Bin. search upper-half cdf
#ifdef DEBUG
  printf("Normal search %g -> %g -> %d",x,y,f);
#endif

  // Linear interpolate between table points

  y = (f - (unorm_table[f]-y) / (unorm_table[f] - unorm_table[f-1]) ) * unorm_scale;

  if (x < .5) y = -y;       // Map upper-half var back to full range
#ifdef DEBUG
  printf(" -> %g\n",y);
#endif

  return (y);
}


//  Generate reads (a) whose lengths are exponentially distributed with mean *mean* and
//    standard deviation *stdev*, (b) that are never shorter than *shortest* and never
//    longer than the string *source*.  Each read is a randomly sampled interval of
//    *source* (each interval is equally likely) that has insertion, deletion, and/or
//    substitution errors introduced into it and which is oriented in either the forward
//    or reverse strand direction with probability FLIP_RATE.  The number of errors
//    introduced is the length of the string times *erate*, and the probability of an
//    insertion, deletion, or substitution is controlled by the defined constants INS_RATE
//    and DEL_RATE.  Generate reads until the sum of the lengths of the reads is greater
//    than slen*coverage.  The reads are output as fasta entries with a specific header
//    format that contains the sampling interval, read length, and a read id.

static void shotgun(char *source)
{ int       maxlen, nreads, qv;
  int64     totlen, totbp;
  char     *rbuffer;
  double    nmean, nsdev;

  nsdev = (1.*RSDEV)/RMEAN;
  nsdev = log(1.+nsdev*nsdev);
  nmean = log(1.*RMEAN) - .5*nsdev;
  nsdev = sqrt(nsdev);

  if (GENOME < RSHORT)
    { fprintf(stderr,"Genome length is less than shortest read length !\n");
      exit (1);
    }

  init_unorm();

  qv = (int) (1000 * (1.-ERROR));

  rbuffer = NULL;
  maxlen  = 0;
  totlen  = 0;
  totbp   = COVERAGE*GENOME;
  nreads  = 0;
  while (totlen < totbp)
    { int   len, sdl, ins, del, elen, rbeg, rend;
      int   j;
      char *s, *t;

      len = (int) exp(nmean + nsdev*sample_unorm(drand48()));    //  Determine length of read.
      if (len > GENOME) len = GENOME;
      if (len < RSHORT)
        continue;

      sdl = (int) (len*ERROR);      //  Determine number of inserts *ins*, deletions *del,
      ins = del = 0;                //    and substitions+deletions *sdl*.
      for (j = 0; j < sdl; j++)
        { double x = drand48();
          if (x < INS_RATE)
            ins += 1;
          else if (x < IDL_RATE)
            del += 1; 
        }
      sdl -= ins;
      elen = len + (ins-del);
      rbeg = (int) (drand48()*((GENOME-len)+.9999999));
      rend = rbeg + len;

      if (elen > maxlen)
        { maxlen  = ((int) (1.2*elen)) + 1000;
          rbuffer = (char *) Realloc(rbuffer,maxlen+3,"Allocating read buffer");
          if (rbuffer == NULL)
            exit (1);
        }

      t = rbuffer;
      s = source + rbeg;

      //   Generate the string with errors.  NB that inserts occur randomly between source
      //     characters, while deletions and substitutions occur on source characters.

      while ((len+1) * drand48() < ins)
        { *t++ = (char) (4.*drand48());
          ins -= 1;
        }
      for ( ; len > 0; len--)
        { if (len * drand48() >= sdl)
            *t++ = *s;
          else if (sdl * drand48() >= del)
            { double x = 3.*drand48();
              if (x >= *s)
                x += 1.;
              *t++ = (char) x;
              sdl -= 1;
            }
          else
            { del -= 1;
              sdl -= 1;
            }
          s += 1;
          while (len * drand48() < ins)
            { *t++ = (char) (4.*drand48());
              ins -= 1;
            }
        }
      *t = 4;

      if (drand48() >= FLIP_RATE)    //  Complement the string with probability FLIP_RATE.
        { printf(">Sim/%d/%d_%d RQ=0.%d\n",nreads+1,0,elen,qv);
          complement(elen,rbuffer);
          j = rend;
          rend = rbeg;
          rbeg = j;
        }
      else
        printf(">Sim/%d/%d_%d RQ=0.%d\n",nreads+1,0,elen,qv);

      Lower_Read(rbuffer);
      for (j = 0; j+80 < elen; j += 80)
        printf("%.80s\n",rbuffer+j);
      if (j < elen)
        printf("%s\n",rbuffer+j);

       if (MAP != NULL)
         fprintf(MAP," %9d %9d\n",rbeg,rend);

       totlen += elen;
       nreads += 1;
    }
}

int main(int argc, char *argv[])
{ char  *source;

//  Usage: <GenomeLen:double> [-c<double(20.)>] [-b<double(.5)>] [-r<int>]
//                            [-m<int(10000)>]  [-s<int(2000)>]  [-x<int(4000)>]
//                            [-e<double(.15)>] [-M<file]"

  { int    i, j;
    char  *eptr;
    double glen;

    Prog_Name = Strdup("simulator","");

    COVERAGE = 20.;
    BIAS     = .5;
    HASR     = 0;
    RMEAN    = 10000;
    RSDEV    = 2000;
    RSHORT   = 4000;
    ERROR    = .15;
    MAP      = NULL;

    j = 1;
    for (i = 1; i < argc; i++)
      if (argv[i][0] == '-')
        switch (argv[i][1])
        { default:
            fprintf(stderr,"%s: -%c is an illegal option\n",Prog_Name,argv[i][2]);
            exit (1);
          case 'c':
            ARG_REAL(COVERAGE)
            if (COVERAGE < 0.)
              { fprintf(stderr,"%s: Coverage must be non-negative (%g)\n",Prog_Name,COVERAGE);
                exit (1);
              }
            break;
          case 'b':
            ARG_REAL(BIAS)
            if (BIAS < 0. || BIAS > 1.)
              { fprintf(stderr,"%s: AT-bias must be in [0,1] (%g)\n",Prog_Name,BIAS);
                exit (1);
              }
            break;
          case 'r':
            SEED = strtol(argv[i]+2,&eptr,10);
            HASR = 1;
            if (*eptr != '\0' || argv[i][2] == '\0')
              { fprintf(stderr,"%s: -r argument is not an integer\n",Prog_Name);
                exit (1);
              }
            break;
          case 'M':
            MAP = Fopen(argv[i]+2,"w");
            if (MAP == NULL)
              exit (1);
            break;
          case 'm':
            ARG_POSITIVE(RMEAN,"Mean read length")
            break;
          case 's':
            ARG_POSITIVE(RSDEV,"Read length standard deviation")
            break;
          case 'x':
            ARG_NON_NEGATIVE(RSHORT,"Read length minimum")
            break;
          case 'e':
            ARG_REAL(ERROR)
            if (ERROR < 0. || ERROR > .5)
              { fprintf(stderr,"%s: Error rate must be in [0,.5] (%g)\n",Prog_Name,ERROR);
                exit (1);
              }
            break;
        }
      else
        argv[j++] = argv[i];
    argc = j;

    if (argc != 2)
      { fprintf(stderr,"Usage: %s %s\n",Prog_Name,Usage[0]);
        fprintf(stderr,"       %*s %s\n",(int) strlen(Prog_Name),"",Usage[1]);
        fprintf(stderr,"       %*s %s\n",(int) strlen(Prog_Name),"",Usage[2]);
        exit (1);
      }

    glen = strtod(argv[1],&eptr);
    if (*eptr != '\0')
      { fprintf(stderr,"%s: genome length is not a real number\n",Prog_Name);
        exit (1);
      }
    if (glen < 0.)
      { fprintf(stderr,"%s: Genome length must be positive (%g)\n",Prog_Name,glen);
        exit (1);
      }
    GENOME = (int) (glen*1000000.);
  }

  source = random_genome();

  shotgun(source);

  if (MAP != NULL)
    fclose(MAP);

  exit (0);
}
