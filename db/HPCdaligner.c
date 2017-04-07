/************************************************************************************\
*                                                                                    *
 * Copyright (c) 2014, Dr. Eugene W. Myers (EWM). All rights reserved.                *
 *                                                                                    *
 * Redistribution and use in source and binary forms, with or without modification,   *
 * are permitted provided that the following conditions are met:                      *
 *                                                                                    *
 *  - Redistributions of source code must retain the above copyright notice, this     *
 *    list of conditions and the following disclaimer.                                *
 *                                                                                    *
 *  - Redistributions in binary form must reproduce the above copyright notice, this  *
 *    list of conditions and the following disclaimer in the documentation and/or     *
 *    other materials provided with the distribution.                                 *
 *                                                                                    *
 *  - The name of EWM may not be used to endorse or promote products derived from     *
 *    this software without specific prior written permission.                        *
 *                                                                                    *
 * THIS SOFTWARE IS PROVIDED BY EWM "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,    *
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

/*********************************************************************************************\
 *
 *  Produce a script to compute overlaps for all block pairs of a DB, and then sort and merge
 *    them into as many .las files as their are blocks.
 *x
 *  Author:  Gene Myers
 *  Date  :  June 1, 2014
 *
 *********************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <getopt.h>

#include "db/DB.h"
#include "filter.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static char *Usage =
{ " [-vbAIiR] [-k<int(14)>] [-w<int(6)>] [-h<int(35)>] [-t<int>] [-H<int>]\n"
    " [-M<int>] [-e<double(.70)] [-l<int(1000)>] [-r<int>] [-s<int(100)>]\n"
    " [--dal<int(4)>] [--dalDiag<int(1)>] [--mrg<int(25)>] [-D host[:port]]\n"
    " [-o fileSuffix] [-mtrack]+  <path:db> [<block:int>[-<range:int>]" };

static void printUsage(char* prog, FILE *out)
  {
    fprintf(out, "\nUsage:\t%s\t%s\n\n", prog, Usage);
    fprintf(out, "  -h            prints this usage info\n");
    fprintf(out, "  -v            enable verbose mode for daligner and LAmerge\n");
    fprintf(out, "  -A            asymmetric, for block X and Y the symmetric alignments for Y vs X are suppressed.\n");
    fprintf(out, "  -I            identity, overlaps of the same read will found and reported\n");
    fprintf(out, "  -i            ignore kmer seeds that are within another kmer hit\n");
    fprintf(out, "  -K            keep intermediate merge results (default: 0)\n");
    fprintf(out, "  -C            check intermediate merge results for correctness (default: 0)\n");
    fprintf(out, "  -b            bias genome (AT content is greater then 70%% or lower then 30%%)\n");
    fprintf(out, "  -k ARG        kmer length (default: 14)\n");
    fprintf(out, "  -w ARG        diagonal band width (default: 6)\n");
    fprintf(out, "  -h ARG        number of kmer hits (default: 35)\n");
    fprintf(out, "  -t ARG        suppresses the use of any k-mer that occurs more than t times in either the subject or target block (default: not set)\n");
    fprintf(out, "  -H ARG        report only overlaps where the a-read is over N base-pairs long (default: 0)\n");
    fprintf(out, "  -M ARG        memory usage limit in Gb (default: unlimited)\n");
    fprintf(out, "  -e ARG        average correlation rate (default: 0.7). Must be in [.7,1.)\n");
    fprintf(out, "  -l ARG        minimum length for local alignments (default: 1000)\n");
    fprintf(out, "  -r ARG        run identifier (default: 1). That means all overlap files of Block X a written to a subdirectory: dRUN-IDENTIFIER_X\n");
    fprintf(out, "  -s ARG        record trace point of the alignment, every -s ARG base pairs (default: 100)\n");
    fprintf(out, "  --dal ARG     number of block comparisons per call to daligner (default: 4)\n");
    fprintf(out, "  --dalDiag ARG number of block comparisons per call to daligner for jobs including the diagonal element (default: 1 if ddust is set, otherwise 4)\n");
    fprintf(out, "  --mrg ARG     gives the maximum number of files that will be merged in a single LAmerge command (default: 25)\n");
    fprintf(out, "  -D host:port  specify host and port where the dynamic dust server is running (default: not set)\n");
    fprintf(out, "  -o ARG        specify a file suffix, if set the daligner plan is written to planDalignARG and the merge plan is written\n"
        "                to planMergeARG (default: not set, i.e. everything goes to stdout)\n");
    fprintf(out, "  -m            specify an interval track that is to be softmasked\n");
    fprintf(out, "  path          database\n");
    fprintf(out, "  bID[-bID]     specify a block or a range of blocks\n");
  }

typedef struct
{
  int DAL_JOBS, DAL_DIAG_JOBS;
  int VERBOSE, BIAS;
  int WINT, TINT, HGAP, HINT, KINT, SINT, LINT;
  double EREL, MREL;
  int RUN_ID, MEM;
  int ASYMMETRY, IDENTITY, IGNORE;

  int fblock, lblock;
  char *db;					// full name dir + name + .db
  char *dbDir;
  char *dbName;

  int dbBlocks;

  char* host;
  int port;
  FILE *dalignOut;
  FILE *mergeOut;

  // track info
  int    MMAX, MTOP;
  char **MASK;

  // parameter for LAmerge
  int CHECK;
  int KEEP;
  int MERGE_JOBS;

} HPC_OPT;

HPC_OPT* parseOptions(int argc, char* argv[])
  {
    HPC_OPT *hopt = (HPC_OPT*) malloc(sizeof(HPC_OPT));

    // set default values
    hopt->VERBOSE = 0;
    hopt->BIAS = 0;
    hopt->KEEP = 0;
    hopt->CHECK = 0;
    hopt->DAL_JOBS = 4;
    hopt->DAL_DIAG_JOBS = 0;
    hopt->MERGE_JOBS = 25;
    hopt->KINT = 14;
    hopt->WINT = 6;
    hopt->HINT = 35;
    hopt->TINT = 0;
    hopt->HGAP = 0;
    hopt->EREL = 0.;
    hopt->MREL = 0.;
    hopt->SINT = 100;
    hopt->LINT = 1000;
    hopt->RUN_ID = 1;
    hopt->MEM = 0;
    hopt->ASYMMETRY = 0;
    hopt->IDENTITY = 0;
    hopt->IGNORE = 0;

    hopt->MTOP  = 0;
    hopt->MMAX  = 10;
    hopt->MASK  = (char **) Malloc(hopt->MMAX*sizeof(char *),"Allocating mask track array");
    
    if (hopt->MASK == NULL)
      exit (1);

    hopt->host = NULL;
    hopt->port = -1;
    hopt->dbBlocks = 0;
    hopt->dalignOut = stdout;
    hopt->mergeOut = stdout;

    int c;
    while (1)
      {
        static struct option long_options[] =
        {
        { "help", no_argument, 0, 'h' },
        { "verbose", no_argument, 0, 'v' },
        { "bias", no_argument, 0, 'b' },
        { "keep", no_argument, 0, 'K' },
        { "check", no_argument, 0, 'C' },
        { "asymm", no_argument, 0, 'A' },
        { "identity", no_argument, 0, 'I' },
        { "ignore", no_argument, 0, 'i' },
        { "kmer", required_argument, 0, 'k' },
        { "bwidth", required_argument, 0, 'w' },
        { "hits", required_argument, 0, 'h' },
        { "htimes", required_argument, 0, 't' },
        { "rlen", required_argument, 0, 'H' },
        { "mem", required_argument, 0, 'M' },
        { "cor", required_argument, 0, 'e' },
        { "alen", required_argument, 0, 'l' },
        { "rid", required_argument, 0, 'r' },
        { "trace", required_argument, 0, 's' },
        { "dal", required_argument, 0, 'j' },
        { "dalDiag", required_argument, 0, 'J' },
        { "mrg", required_argument, 0, 'c' },
        { "dServer", required_argument, 0, 'D' },
        { "out", required_argument, 0, 'o' },
        { "track", required_argument, 0, 'm' }
        };

        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long(argc, argv, "?vbKCAIik:w:h:t:H:M:e:l:r:s:j:J:c:D:o:m:", long_options, &option_index);

        /* Detect the end of the options. */
        if (c == -1)
          break;

        switch (c)
        {
          case 0:
            /* If this option set a flag, do nothing else now. */
            if (long_options[option_index].flag != 0)
              break;
            break;

          case '?':
            printUsage(argv[0], stderr);
            exit(1);
          case 'v':
            hopt->VERBOSE = 1;
            break;
          case 'A':
            hopt->ASYMMETRY = 1;
            break;
          case 'I':
            hopt->IDENTITY = 1;
            break;
          case 'i':
            hopt->IGNORE = 1;
            break;
          case 'b':
            hopt->BIAS = 1;
            break;
          case 'K':
            hopt->KEEP = 1;
            break;
          case 'C':
            hopt->CHECK = 1;
            break;
          case 'k':
            hopt->KINT = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from kmer length (-k ARG)! Must be an integer in [4, 16]\n");
                exit(1);
              }
            if (hopt->KINT < 4 || hopt->KINT > 16)
              {
                fprintf(stderr, "Kmer length not accepted! Must be in [4, 16]\n");
                exit(1);
              }
            break;
          case 'w':
            hopt->WINT = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from diagonal band width (-w ARG)! \n");
                exit(1);
              }
            break;
          case 'h':
            hopt->HINT = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -h/--hits! \n");
                exit(1);
              }
            break;
          case 't':
            hopt->TINT = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -t/--htimes! \n");
                exit(1);
              }
            break;
          case 'H':
            hopt->HGAP = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -H/--rlen! \n");
                exit(1);
              }
            break;
          case 'M':
            hopt->MEM = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -M/--mem! \n");
                exit(1);
              }
            break;
          case 'e':
            hopt->EREL = strtod(optarg, NULL);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -e/--cor! \n");
                exit(1);
              }
            if (hopt->EREL < .7 || hopt->EREL >= 1)
              {
                fprintf(stderr, "Average correlation not accepted! Must be in [.7, 1)\n");
                exit(1);
              }
            break;
          case 'l':
            hopt->LINT = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -l/--alen! \n");
                exit(1);
              }
            break;
          case 'r':
            hopt->RUN_ID = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -r/--rid! \n");
                exit(1);
              }
            if (hopt->RUN_ID < 0 || hopt->RUN_ID > 999)
              {
                fprintf(stderr, "Run id not accepted! Must be in [0, 999]\n");
                exit(1);
              }
            break;
          case 's':
            hopt->SINT = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -s/--trace! \n");
                exit(1);
              }
            break;
          case 'j':
            hopt->DAL_JOBS = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -c/--dal! \n");
                exit(1);
              }
            if (hopt->DAL_JOBS < 1)
              {
                fprintf(stderr, "Number of block comparisons per call to daligner not accepted! Must be in > 0! \n");
                exit(1);
              }
            break;
          case 'J':
            hopt->DAL_DIAG_JOBS = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -c/--dal! \n");
                exit(1);
              }
            if (hopt->DAL_DIAG_JOBS < 1)
              {
                fprintf(stderr, "Number of block comparisons per call to daligner not accepted! Must be in > 0! \n");
                exit(1);
              }
            break;
          case 'c':
            hopt->MERGE_JOBS = (int) strtol(optarg, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse argument from option -c/--mrg! \n");
                exit(1);
              }
            if (hopt->MERGE_JOBS < 3)
              {
                fprintf(stderr, "Maximum number of files that will be merged in a single LAmerge command not accepted! Must be in > 3! \n");
                exit(1);
              }
            break;
          case 'D':
            {
              char *ptr;
              int ch = ':';

              ptr = strchr(optarg, ch);
              if (ptr)
                {
                  hopt->port = (int) strtol(ptr + 1, NULL, 10);
                  if (errno)
                    {
                      fprintf(stderr, "Cannot parse argument port number from option -D/--dServer! \n");
                      exit(1);
                    }
                  hopt->host = (char*) malloc(strlen(optarg) + 10);
                  strcpy(hopt->host, optarg);
                  hopt->host[ptr - optarg] = '\0';
                }
              else
                {
                  hopt->host = (char*) malloc(strlen(optarg) + 10);
                  strcpy(hopt->host, optarg);
                }
            }
            break;
          case 'o':
            {
              char *out;
              out = (char*) malloc(strlen(optarg) + 20);
              sprintf(out, "planDalign%s", optarg);
              if ((hopt->dalignOut = fopen(out, "w")) == NULL)
                {
                  fprintf(stderr, "ERROR - Cannot open file %s for writing\n", out);
                  exit(1);
                }
              sprintf(out, "planMerge%s", optarg);
              if ((hopt->mergeOut = fopen(out, "w")) == NULL)
                {
                  fprintf(stderr, "ERROR - Cannot open file %s for writing\n", out);
                  exit(1);
                }
              free(out);
            }
            break;
            case 'm':
                    if (hopt->MTOP >= hopt->MMAX)
                    {
                        hopt->MMAX  = 1.2*hopt->MTOP + 10;
                        hopt->MASK  = (char **) Realloc(hopt->MASK,hopt->MMAX*sizeof(char *),"Reallocating mask track array");
                        if (hopt->MASK == NULL)
                            exit (1);
                    }
                    hopt->MASK[hopt->MTOP] = optarg;
                    hopt->MTOP++;
                    break;            
          default:
            fprintf(stderr, "[ERROR] Unsupported argument: %s\n", argv[optind]);
            printUsage(argv[0], stderr);
            exit(1);
        }
      }

    if (optind + 1 > argc)
      {
        fprintf(stderr, "Required arument: <Database>\n");
        printUsage(argv[0], stderr);
        exit(1);
      }

    // database name
      {
        int len = strlen(argv[optind]);
        hopt->db = (char*) malloc(len + 10);
        strcpy(hopt->db, argv[optind]);

        if (strcasecmp(hopt->db + (len - 3), ".db") != 0)
          {
            strcpy(hopt->db + len, ".db");
          }

        hopt->dbName = Root(hopt->db, ".db");
        hopt->dbDir = PathTo(hopt->db);

        // check if db file is available and parse number blocks
          {
            int nblocks = DB_Blocks(hopt->db);

            if(nblocks < 0)
              exit(1);

            hopt->dbBlocks = nblocks;
            hopt->fblock = 1;
            hopt->lblock = nblocks;
          }

        optind++;
      }

    // parse blocks
    if (optind < argc)
      {
        char * ptr;
        int ch = '-';
        int fblock;
        int lblock;

        ptr = strchr(argv[optind], ch);
        if (ptr)
          {
            lblock = (int) strtol(ptr + 1, NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse end block id! \n");
                exit(1);
              }
            *ptr = '\0';
            fblock = (int) strtol(argv[optind], NULL, 10);
            if (errno)
              {
                fprintf(stderr, "Cannot parse start block id! \n");
                exit(1);
              }
            *ptr = '-';
          }
        else
          {
            fblock = (int) strtol(argv[optind], NULL, 10);
            lblock = fblock;
            if (errno)
              {
                fprintf(stderr, "Cannot parse block id! \n");
                exit(1);
              }
          }

        if (fblock < 1 || fblock > hopt->lblock)
          {
            fprintf(stderr, "Ivalid first block id %d! \n", fblock);
            exit(1);
          }

        if (lblock < 1 || lblock < fblock || lblock > hopt->lblock)
          {
            fprintf(stderr, "Ivalid last block id %d! \n", lblock);
            exit(1);
          }
        hopt->fblock = fblock;
        hopt->lblock = lblock;
      }

    return hopt;
  }

int main(int argc, char *argv[])
  {
      {
        int njobs;
        int i, j;

        HPC_OPT *hopt = parseOptions(argc, argv);

        if(hopt->DAL_DIAG_JOBS == 0)
          {
            if (hopt->host == NULL)
              hopt->DAL_DIAG_JOBS = hopt->DAL_JOBS;
            else
              hopt->DAL_DIAG_JOBS = 1;
          }

        njobs = 0;

        for (i = hopt->fblock; i <= hopt->lblock; i++)
          {
            njobs += 1 + (int) ceil(1. * MAX(i - hopt->fblock + 1 - hopt->DAL_DIAG_JOBS, 0) / hopt->DAL_JOBS);
//						printf("%d --> %d (%d)\n", i, 1 + (int)ceil(1.*MAX(i-hopt->fblock+1-hopt->DAL_DIAG_JOBS, 0) / hopt->DAL_JOBS), njobs);
          }

        int blub = 0;
        int low, hgh;
        int bla = (hopt->lblock - hopt->fblock + 1) * (hopt->lblock - hopt->fblock + 2) / 2;
        int test = 0;
        if (hopt->dalignOut == stdout)
          fprintf(hopt->dalignOut, "# Daligner jobs (%d)\n", njobs);
        int first = 1;
        while (blub < bla)
          {
            for (i = hopt->fblock; i <= hopt->lblock; i++)
              {
                if (first)
                  {
                    hgh = i - test * hopt->DAL_DIAG_JOBS;
                    low = MAX(hopt->fblock, hgh - hopt->DAL_DIAG_JOBS + 1);
                  }
                else
                  {
                    hgh = i - (hopt->DAL_DIAG_JOBS + (test - 1) * hopt->DAL_JOBS);
                    low = MAX(hopt->fblock, hgh - hopt->DAL_JOBS + 1);
                  }
                if (hgh >= low)
                  {
                    fprintf(hopt->dalignOut, "daligner");
                    if (hopt->VERBOSE)
                      fprintf(hopt->dalignOut, " -v");
                    if (hopt->BIAS)
                      fprintf(hopt->dalignOut, " -b");
                    if (hopt->ASYMMETRY)
                      fprintf(hopt->dalignOut, " -A");
                    if (hopt->IDENTITY)
                      fprintf(hopt->dalignOut, " -I");
                    if (hopt->IGNORE)
                      fprintf(hopt->dalignOut, " -i");
                    if (hopt->KINT != 14)
                      fprintf(hopt->dalignOut, " -k%d", hopt->KINT);
                    if (hopt->WINT != 6)
                      fprintf(hopt->dalignOut, " -w%d", hopt->WINT);
                    if (hopt->HINT != 35)
                      fprintf(hopt->dalignOut, " -h%d", hopt->HINT);
                    if (hopt->TINT > 0)
                      fprintf(hopt->dalignOut, " -t%d", hopt->TINT);
                    if (hopt->HGAP > 0)
                      fprintf(hopt->dalignOut, " -H%d", hopt->HGAP);
                    if (hopt->EREL > .1)
                      fprintf(hopt->dalignOut, " -e%g", hopt->EREL);
                    if (hopt->MREL > .1)
                      fprintf(hopt->dalignOut, " -m%g", hopt->MREL);
                    if (hopt->LINT != 1000)
                      fprintf(hopt->dalignOut, " -l%d", hopt->LINT);
                    if (hopt->SINT != 100)
                      fprintf(hopt->dalignOut, " -s%d", hopt->SINT);
                    if (hopt->MEM > 0)
                      fprintf(hopt->dalignOut, " -s%d", hopt->MEM);
                    if (hopt->host)
                      fprintf(hopt->dalignOut, " -D%s", hopt->host);
                    if (hopt->port > 0)
                      fprintf(hopt->dalignOut, ":%d", hopt->port);
                    fprintf(hopt->dalignOut, " -r%d", hopt->RUN_ID);
                    for(j=0; j<hopt->MTOP; j++)
                       fprintf(hopt->dalignOut, " -m%s", hopt->MASK[j]); 
                    if (strlen(hopt->dbDir) > 1)
                      fprintf(hopt->dalignOut, " %s%s.%d", hopt->dbDir, hopt->dbName, i);
                    else
                      fprintf(hopt->dalignOut, " %s.%d", hopt->dbName, i);

                    for (j = hgh; j >= low; j--)
                      {
                        blub++;
                        if (strlen(hopt->dbDir) > 1)
                          fprintf(hopt->dalignOut, " %s%s.%d", hopt->dbDir, hopt->dbName, j);
                        else
                          fprintf(hopt->dalignOut, " %s.%d", hopt->dbName, j);
                      }

                    fprintf(hopt->dalignOut, "\n");
                  }
              }
            first = 0;
            test++;
          }
        if (hopt->dalignOut == stdout)
          fprintf(hopt->mergeOut, "# merge jobs(%d)\n", hopt->lblock - hopt->fblock + 1);
        char *dir;
        for (i = hopt->fblock; i <= hopt->lblock; i++)
          {
            dir = getDir(hopt->RUN_ID, i);
            fprintf(hopt->mergeOut, "LAmerge");
            if (hopt->VERBOSE)
              fprintf(hopt->mergeOut, " -v");
            if (hopt->CHECK)
              fprintf(hopt->mergeOut, " -c");
            if (hopt->KEEP)
              fprintf(hopt->mergeOut, " -k");
            fprintf(hopt->mergeOut, " -n %d", hopt->MERGE_JOBS);
            fprintf(hopt->mergeOut, " %s %s \n", hopt->db, dir);
          }
      }

    exit(0);
  }
