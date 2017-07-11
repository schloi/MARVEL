/*****************************************************************************************\
*                                                                                         *
*  This program improves a multi-alignment of DNA sequences                               *
*                                                                                         *
*  This is the driver that calls library routines to read, realign, and write a series    *
*    of contigs in vertical format                                                        *
*                                                                                         *
*  Author:  Gene Myers                                                                    *
*  Date  :  March 2007                                                                    *
*                                                                                         *
\*****************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#include "realigner.h"

#undef DEBUG

#define OPT_B_DEFAULT 8
#define OPT_R_DEFAULT 0
#define OPT_C_DEFAULT 0

extern char *optarg;
extern int optind, opterr, optopt;

static void usage()
{
    printf("[-b <int>] [-r] [-c]\n");
    printf("options: -b ... band width (%d)\n", OPT_B_DEFAULT);
    printf("         -r ... same rows (%d)\n", OPT_R_DEFAULT);
    printf("         -c ... comments (%d)\n", OPT_C_DEFAULT);
}

int main(int argc, char *argv[])
{ 

int bandwidth = OPT_B_DEFAULT;
  int samerows = OPT_R_DEFAULT;
  int comments = OPT_C_DEFAULT;
 
    int c;
    
    opterr = 0;
    
    while ((c = getopt(argc, argv, "crb:")) != -1)
    {
        switch (c)
        {
            case 'c':
                      comments = 1;
                      break;
        
            case 'b':
                      bandwidth = atoi(optarg);
                      break;

            case 'r':
                      samerows = 1;
                      break;
                      
            default:
                      usage();
                      exit(1);          
        }    
    }

  /* Read in each contig, realign, print, and free */

  while (1)
    { Re_Contig *ctg;

      if (comments)
        ctg = Re_Read_Contig(stdin,stdout);
      else
        ctg = Re_Read_Contig(stdin,NULL);

      if (ctg == NULL) break;
  
#ifdef DEBUG
      Re_Print_Structure(ctg,stdout);
#endif

      Re_Align_Contig(ctg,bandwidth);

#ifdef DEBUG
      Re_Print_Structure(ctg,stdout);
#endif

      Re_Print_Contig(ctg,stdout,samerows);

      Re_Free_Contig(ctg);
    }
}
