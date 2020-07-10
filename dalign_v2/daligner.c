
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <dirent.h>

#include <sys/param.h>
#if defined(BSD)
#include <sys/sysctl.h>
#endif

#include "db/DB.h"
#include "lib/tracks.h"
#include "radix.h"
#include "filter.h"
#include "ovlbuffer.h"

#define DAZZ_TRACK HITS_TRACK
// #define BLOCK_SYMBOL     '_'

static char *Usage[] =
  { "[-vaAI] [-k<int(16)>] [-w<int(6)>] [-h<int(50)>] [-t<int>] [-M<int>]",
    "        [-e<double(.75)] [-l<int(1500)>] [-s<int(100)>] [-H<int>]",
    "        [-T<int(4)>] [-P<dir(/tmp)>] [-m<track>]+",
    "        <subject:db|dam> <target:db|dam> ...",
  };

int     VERBOSE;   //   Globally visible to filter.c
int     MINOVER;
int     HGAP_MIN;
int     SYMMETRIC;
int     IDENTITY;
char   *SORT_PATH;

uint64_t  MEM_LIMIT;
uint64_t  MEM_PHYSICAL;

/*  Adapted from code by David Robert Nadeau (http://NadeauSoftware.com) licensed under
 *     "Creative Commons Attribution 3.0 Unported License"
 *          (http://creativecommons.org/licenses/by/3.0/deed.en_US)
 *
 *   I removed Windows options, reformated, and return int64 instead of size_t
 */

static int64 getMemorySize( )
{
#if defined(CTL_HW) && (defined(HW_MEMSIZE) || defined(HW_PHYSMEM64))

  // OSX, NetBSD, OpenBSD

  int     mib[2];
  size_t  size = 0;
  size_t  len = sizeof( size );

  mib[0] = CTL_HW;
#if defined(HW_MEMSIZE)
  mib[1] = HW_MEMSIZE;            // OSX
#elif defined(HW_PHYSMEM64)
  mib[1] = HW_PHYSMEM64;          // NetBSD, OpenBSD
#endif
  if (sysctl(mib,2,&size,&len,NULL,0) == 0)
    return ((size_t) size);
  return (0);

#elif defined(_SC_AIX_REALMEM)

  // AIX

  return ((size_t) sysconf( _SC_AIX_REALMEM ) * ((size_t) 1024L));

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)

  // FreeBSD, Linux, OpenBSD, & Solaris

  size_t  size = 0;

  size = (size_t) sysconf(_SC_PHYS_PAGES);
  return (size * ((size_t) sysconf(_SC_PAGESIZE)));

#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)

  // ? Legacy ?

  size_t  size = 0;

  size = (size_t) sysconf(_SC_PHYS_PAGES);
  return (size * ((size_t) sysconf(_SC_PAGE_SIZE)));

#elif defined(CTL_HW) && (defined(HW_PHYSMEM) || defined(HW_REALMEM))

  // DragonFly BSD, FreeBSD, NetBSD, OpenBSD, and OSX

  int          mib[2];
  unsigned int size = 0;
  size_t       len  = sizeof( size );

  mib[0] = CTL_HW;
#if defined(HW_REALMEM)
  mib[1] = HW_REALMEM;		// FreeBSD
#elif defined(HW_PYSMEM)
  mib[1] = HW_PHYSMEM;		// Others
#endif
  if (sysctl(mib,2,&size,&len,NULL,0) == 0)
    return (size_t)size;
  return (0);

#else

  return (0);

#endif
}

typedef struct
  { int *ano;
    int *end;
    int  idx;
    int  out;
  } Event;

static void reheap(int s, Event **heap, int hsize)
{ int      c, l, r;
  Event   *hs, *hr, *hl;

  c  = s;
  hs = heap[s];
  while ((l = 2*c) <= hsize)
    { r  = l+1;
      hl = heap[l];
      hr = heap[r];
      if (hr->idx > hl->idx)
        { if (hs->idx > hl->idx)
            { heap[c] = hl;
              c = l;
            }
          else
            break;
        }
      else
        { if (hs->idx > hr->idx)
            { heap[c] = hr;
              c = r;
            }
          else
            break;
        }
    }
  if (c != s)
    heap[c] = hs;
}

static int64 merge_size(DAZZ_DB *block, int mtop)
{ Event       ev[mtop+1];
  Event      *heap[mtop+2];
  int         r, mhalf;
  int64       nsize;

  { DAZZ_TRACK *track;
    int         i;

    track = block->tracks;
    for (i = 0; i < mtop; i++)
      { ev[i].ano = ((int *) (track->data)) + ((int64 *) (track->anno))[0];
        ev[i].out = 1;
        heap[i+1] = ev+i;
        track = track->next;
      }
    ev[mtop].idx = INT32_MAX;
    heap[mtop+1] = ev+mtop;
  }

  mhalf = mtop/2;

  nsize = 0;
  for (r = 0; r < block->nreads; r++)
    { int         i, level, hsize;
      DAZZ_TRACK *track;

      track = block->tracks;
      for (i = 0; i < mtop; i++)
        { ev[i].end = ((int *) (track->data)) + ((int64 *) (track->anno))[r+1];
          if (ev[i].ano < ev[i].end)
            ev[i].idx = *(ev[i].ano);
          else
            ev[i].idx = INT32_MAX;
          track = track->next;
        }
      hsize = mtop;

      for (i = mhalf; i > 1; i--)
        reheap(i,heap,hsize);

      level = 0;
      while (1)
        { Event *p;

          reheap(1,heap,hsize);

          p = heap[1];
          if (p->idx == INT32_MAX) break;

          p->out = 1-p->out;
          if (p->out)
            { level -= 1;
              if (level == 0)
                nsize += 1;
            }
          else
            { if (level == 0)
                nsize += 1;
              level += 1;
            }
          p->ano += 1;
          if (p->ano >= p->end)
            p->idx = INT32_MAX;
          else
            p->idx = *(p->ano);
        }
    }

  return (nsize);
}

static DAZZ_TRACK *merge_tracks(DAZZ_DB *block, int mtop, int64 nsize)
{ DAZZ_TRACK *ntrack;
  Event       ev[mtop+1];
  Event      *heap[mtop+2];
  int         r, mhalf;
  int64      *anno;
  int        *data;

  ntrack = (DAZZ_TRACK *) Malloc(sizeof(DAZZ_TRACK),"Allocating merged track");
  if (ntrack == NULL)
    Clean_Exit(1);
  ntrack->name = Strdup("merge","Allocating merged track");
  ntrack->anno = anno = (int64 *) Malloc(sizeof(int64)*(block->nreads+1),"Allocating merged track");
  ntrack->data = data = (int *) Malloc(sizeof(int)*nsize,"Allocating merged track");
  ntrack->size = sizeof(int);
  ntrack->next = NULL;
  // ntrack->loaded = 1;
  if (anno == NULL || data == NULL || ntrack->name == NULL)
    Clean_Exit(1);

  { DAZZ_TRACK *track;
    int         i;

    track = block->tracks;
    for (i = 0; i < mtop; i++)
      { ev[i].ano = ((int *) (track->data)) + ((int64 *) (track->anno))[0];
        ev[i].out = 1;
        heap[i+1] = ev+i;
        track = track->next;
      }
    ev[mtop].idx = INT32_MAX;
    heap[mtop+1] = ev+mtop;
  }

  mhalf = mtop/2;

  nsize = 0;
  for (r = 0; r < block->nreads; r++)
    { int         i, level, hsize;
      DAZZ_TRACK *track;

      anno[r] = nsize;

      track = block->tracks;
      for (i = 0; i < mtop; i++)
        { ev[i].end = ((int *) (track->data)) + ((int64 *) (track->anno))[r+1];
          if (ev[i].ano < ev[i].end)
            ev[i].idx = *(ev[i].ano);
          else
            ev[i].idx = INT32_MAX;
          track = track->next;
        }
      hsize = mtop;

      for (i = mhalf; i > 1; i--)
        reheap(i,heap,hsize);

      level = 0;
      while (1)
        { Event *p;

          reheap(1,heap,hsize);

          p = heap[1];
          if (p->idx == INT32_MAX) break;

          p->out = 1-p->out;
          if (p->out)
            { level -= 1;
              if (level == 0)
                data[nsize++] = p->idx;
            }
          else
            { if (level == 0)
                data[nsize++] = p->idx;
              level += 1;
            }
          p->ano += 1;
          if (p->ano >= p->end)
            p->idx = INT32_MAX;
          else
            p->idx = *(p->ano);
        }
    }
  anno[r] = nsize;

  return (ntrack);
}

static int read_DB(DAZZ_DB *block, char *name, char **mask, int *mstat, int mtop, int kmer)
{ int i, isdam, stop;

  isdam = Open_DB(name,block);
  if (isdam < 0)
    Clean_Exit(1);

    /*
  for (i = 0; i < mtop; i++)
    { status = Check_Track(block,mask[i]); // ,&kind);
      if (status >= 0)
        if (kind == MASK_TRACK)
          mstat[i] = 0;
        else
          { if (mstat[i] != 0)
              mstat[i] = -3;
          }
      else
        { if (mstat[i] == -2)
            mstat[i] = status;
        }
      if (status == 0 && kind == MASK_TRACK)
        Open_Track(block,mask[i]);
    }
    */

  // Trim_DB(block);

  stop = 0;
  for (i = 0; i < mtop; i++)
    { DAZZ_TRACK *track;
      int64      *anno;
      int         j;

    /*
      status = Check_Track(block,mask[i]); // ,&kind);
      if (status < 0 || kind != MASK_TRACK)
        continue;
        */

      stop += 1;

      if ( block->part >  0 )
      {
          track = track_load_block(block, mask[i]);
      }
      else
      {
          track = track_load(block, mask[i]);
      }

      if ( track == NULL )
      {
          printf("unable to load track %s\n", mask[i]);
          exit(1);
      }

      mstat[i] = 0;

      anno = (int64 *) (track->anno);
      for (j = 0; j <= block->nreads; j++)
        anno[j] /= sizeof(int);
    }

  if (stop > 1)
    { int64       nsize;
      DAZZ_TRACK *track;

      nsize = merge_size(block,stop);
      track = merge_tracks(block,stop,nsize);

      while (block->tracks != NULL)
        Close_Track(block,block->tracks->name);

      block->tracks = track;
    }

    for (i = 0; i < block->nreads; i++)
        if (block->reads[i].rlen < kmer)
          { fprintf(stderr,"%s: Block %s contains reads < %dbp long !  Run DBsplit.\n",
                           Prog_Name,name,kmer);
            Clean_Exit(1);
          }

  // Load_All_Reads(block,0);
  Read_All_Sequences(block, 0);

  return (isdam);
}

/*
static char *CommandBuffer(char *aname, char *bname, char *spath)
{ static char *cat = NULL;
  static int   max = -1;
  int len;

  len = 2*(strlen(aname) + strlen(bname) + strlen(spath)) + 200;
  if (len > max)
    { max = ((int) (1.2*len)) + 100;
      if ((cat = (char *) realloc(cat,max+1)) == NULL)
        { fprintf(stderr,"%s: Out of memory (Making path name)\n",Prog_Name);
          Clean_Exit(1);
        }
    }
  return (cat);
}
*/

void Clean_Exit(int val)
{
  /*
  char *command;

    printf("Clean_Exit... skipping\n");
    return ;

  command = CommandBuffer("","",SORT_PATH);
  sprintf(command,"rm -r %s",SORT_PATH);
  if (system(command) != 0)
    { fprintf(stderr,"%s: Command Failed:\n%*s      %s\n",
                     Prog_Name,(int) strlen(Prog_Name),"",command);
      exit (1);
    }
    */

  exit (val);
}

int main(int argc, char *argv[])
{ DAZZ_DB    _ablock, _bblock;
  DAZZ_DB    *ablock = &_ablock, *bblock = &_bblock;
  char       *afile,  *bfile;
  char       *apath,  *bpath;
  char       *aroot,  *broot;
  void       *aindex, *bindex;
  int         alen,    blen;
  Align_Spec *asettings;
  int         isdam;
  int         MMAX, MTOP, *MSTAT;
  char      **MASK;

  int    KMER_LEN;
  int    BIN_SHIFT;
  int    MAX_REPS;
  int    HIT_MIN;
  double AVE_ERROR;
  int    SPACING;
  int    NTHREADS;
  int    MAP_ORDER;

  { int    i, j, k;
    int    flags[128];
    char  *eptr;
    DIR   *dirp;

    ARG_INIT("daligner2.0")

    KMER_LEN  = 16;
    HIT_MIN   = 50;
    BIN_SHIFT = 6;
    MAX_REPS  = 0;
    HGAP_MIN  = 0;
    AVE_ERROR = .75;
    SPACING   = 100;
    MINOVER   = 1500;    //   Globally visible to filter.c
    NTHREADS  = 4;
    SORT_PATH = "/tmp";

    MEM_PHYSICAL = getMemorySize();
    MEM_LIMIT    = MEM_PHYSICAL;
    if (MEM_PHYSICAL == 0)
      { fprintf(stderr,"\nWarning: Could not get physical memory size\n");
        fflush(stderr);
      }

    MTOP  = 0;
    MMAX  = 10;
    MASK  = (char **) Malloc(MMAX*sizeof(char *),"Allocating mask track array");
    MSTAT = (int *) Malloc(MMAX*sizeof(int),"Allocating mask status array");
    if (MASK == NULL || MSTAT == NULL)
      exit (1);

    j    = 1;
    for (i = 1; i < argc; i++)
      if (argv[i][0] == '-')
        switch (argv[i][1])
        { default:
            ARG_FLAGS("vaAI")
            break;
          case 'k':
            ARG_POSITIVE(KMER_LEN,"K-mer length")
            if (KMER_LEN > 32)
              { fprintf(stderr,"%s: K-mer length must be 32 or less\n",Prog_Name);
                exit (1);
              }
            break;
          case 'w':
            ARG_POSITIVE(BIN_SHIFT,"Log of bin width")
            break;
          case 'h':
            ARG_POSITIVE(HIT_MIN,"Hit threshold (in bp.s)")
            break;
          case 't':
            ARG_POSITIVE(MAX_REPS,"Tuple supression frequency")
            break;
          case 'H':
            ARG_POSITIVE(HGAP_MIN,"HGAP threshold (in bp.s)")
            break;
          case 'e':
            ARG_REAL(AVE_ERROR)
            if (AVE_ERROR < .7 || AVE_ERROR >= 1.)
              { fprintf(stderr,"%s: Average correlation must be in [.7,1.) (%g)\n",
                               Prog_Name,AVE_ERROR);
                exit (1);
              }
            break;
          case 'l':
            ARG_POSITIVE(MINOVER,"Minimum alignment length")
            break;
          case 's':
            ARG_POSITIVE(SPACING,"Trace spacing")
            break;
          case 'M':
            { int limit;

              ARG_NON_NEGATIVE(limit,"Memory allocation (in Gb)")
              MEM_LIMIT = limit * 0x40000000ll;
              break;
            }
          case 'm':
            if (MTOP >= MMAX)
              { MMAX  = 1.2*MTOP + 10;
                MASK  = (char **) Realloc(MASK,MMAX*sizeof(char *),"Reallocating mask track array");
                MSTAT = (int *) Realloc(MSTAT,MMAX*sizeof(int),"Reallocating mask status array");
                if (MASK == NULL || MSTAT == NULL)
                  exit (1);
              }
            MASK[MTOP++] = argv[i]+2;
            break;
          case 'P':
            SORT_PATH = argv[i]+2;
            break;
          case 'T':
            ARG_POSITIVE(NTHREADS,"Number of threads")
            break;
        }
      else
        argv[j++] = argv[i];
    argc = j;

    VERBOSE   = flags['v'];   //  Globally declared in filter.h
    SYMMETRIC = 1-flags['A'];
    IDENTITY  = flags['I'];
    MAP_ORDER = flags['a'];

    if (argc <= 2)
      { fprintf(stderr,"Usage: %s %s\n",Prog_Name,Usage[0]);
        fprintf(stderr,"       %*s %s\n",(int) strlen(Prog_Name),"",Usage[1]);
        fprintf(stderr,"       %*s %s\n",(int) strlen(Prog_Name),"",Usage[2]);
        fprintf(stderr,"       %*s %s\n",(int) strlen(Prog_Name),"",Usage[3]);
        fprintf(stderr,"\n");
        fprintf(stderr,"      -k: k-mer size (must be <= 32).\n");
        fprintf(stderr,"      -w: Look for k-mers in averlapping bands of size 2^-w.\n");
        fprintf(stderr,"      -h: A seed hit if the k-mers in band cover >= -h bps in the");
        fprintf(stderr," targest read.\n");
        fprintf(stderr,"      -t: Ignore k-mers that occur >= -t times in a block.\n");
        fprintf(stderr,"      -M: Use only -M GB of memory by ignoring most frequent k-mers.\n");
        fprintf(stderr,"\n");
        fprintf(stderr,"      -e: Look for alignments with -e percent similarity.\n");
        fprintf(stderr,"      -l: Look for alignments of length >= -l.\n");
        fprintf(stderr,"      -s: The trace point spacing for encoding alignments.\n");
        fprintf(stderr,"      -H: HGAP option: align only target reads of length >= -H.\n");
        fprintf(stderr,"\n");
        fprintf(stderr,"      -T: Use -T threads.\n");
        fprintf(stderr,"      -P: Do block level sorts and merges in directory -P.\n");
        fprintf(stderr,"      -m: Soft mask the blocks with the specified mask.\n");
        fprintf(stderr,"\n");
        fprintf(stderr,"      -v: Verbose mode, output statistics as proceed.\n");
        fprintf(stderr,"      -a: sort .las by A-read,A-position pairs for map usecase\n");
        fprintf(stderr,"          off => sort .las by A,B-read pairs for overlap piles\n");
        fprintf(stderr,"      -A: Compare subjet to target, but not vice versa.\n");
        fprintf(stderr,"      -I: Compare reads to themselves\n");
        exit (1);
      }

    for (j = 0; j < MTOP; j++)
      MSTAT[j] = -2;
  }

  MINOVER *= 2;
  Set_Filter_Params(KMER_LEN,BIN_SHIFT,MAX_REPS,HIT_MIN,NTHREADS);
  Set_Radix_Params(NTHREADS,VERBOSE);

  // Create directory in SORT_PATH for file operations

  { char *newpath;

    newpath = (char *) Malloc(strlen(SORT_PATH)+30,"Allocating sort path");
    if (newpath == NULL)
      exit (1);
    // sprintf(newpath,"%s/daligner.%d",SORT_PATH,getpid());
    sprintf(newpath,"%s",SORT_PATH);

    if (mkdir(newpath,S_IRWXU) != 0 && errno != EEXIST)
      { fprintf(stderr,"%s: Could not create directory %s\n",Prog_Name,newpath);
        perror("failed");
        exit (1);
      }
    SORT_PATH = newpath;

  }

  // Read in the reads in A

  afile = argv[1];
  isdam = read_DB(ablock,afile,MASK,MSTAT,MTOP,KMER_LEN);
  if (isdam)
    aroot = Root(afile,".dam");
  else
    aroot = Root(afile,".db");
  apath = PathTo(afile);

  asettings = New_Align_Spec( AVE_ERROR, SPACING, ablock->freq, 1, NTHREADS);

  // Compare against reads in B in both orientations

  { int           i, j;
    // sBlock_Looper *parse;
    // char         *command;

    aindex = NULL;
    broot  = NULL;
    for (i = 2; i < argc; i++)
      { // parse = Parse_Block_DB_Arg(argv[i]);
        char* bfile = argv[i];

        // while (Advance_Block_Arg(parse))
          { broot = Root(bfile, ".db");
            bpath = PathTo(bfile);

            if (strcmp(bpath,apath) != 0 || strcmp(broot,aroot) != 0)
              { bfile = Strdup(Catenate(bpath,"/",broot,""),"Allocating path");
                read_DB(bblock,bfile,MASK,MSTAT,MTOP,KMER_LEN);
                free(bfile);
              }
            else
              { free(broot);
                broot = aroot;
              }
            free(bpath);

            if (i == 2)
              { for (j = 0; j < MTOP; j++)
                  { if (MSTAT[j] == -2)
                      printf("%s: Warning: -m%s option given but no track found.\n",
                              Prog_Name,MASK[j]);
                    else if (MSTAT[j] == -1)
                      printf("%s: Warning: %s track not sync'd with relevant db.\n",
                             Prog_Name,MASK[j]);
                    else if (MSTAT[j] == -3)
                      printf("%s: Warning: %s track is not a mask track.\n",Prog_Name,MASK[j]);
                  }

                if (VERBOSE)
                  printf("\nBuilding index for %s\n",aroot);
                aindex = Sort_Kmers(ablock,&alen);
              }

            if (aroot != broot)
              { if (VERBOSE)
                  printf("\nBuilding index for %s\n",broot);
                bindex = Sort_Kmers(bblock,&blen);
                Match_Filter(aroot,ablock,broot,bblock,aindex,alen,bindex,blen,asettings);

                int lastRead;
                if ( bblock->part < ablock->part )
                    lastRead = bblock->ufirst + bblock->nreads - 1;
                else
                    lastRead = ablock->ufirst + ablock->nreads - 1;

                Write_Overlap_Buffer( asettings, SORT_PATH, aroot, broot, lastRead );
                Reset_Overlap_Buffer( asettings );

                Close_DB(bblock);
              }
            else
            {
              Match_Filter(aroot,ablock,aroot,ablock,aindex,alen,aindex,alen,asettings);

              Write_Overlap_Buffer( asettings, SORT_PATH, aroot, aroot, ablock->ufirst + ablock->nreads - 1 );
              Reset_Overlap_Buffer( asettings );

            }

/*
            command = CommandBuffer(aroot,broot,SORT_PATH);

#define SYSTEM_CHECK(command)						\
 if (VERBOSE)								\
   printf("\n%s\n",command);

            sprintf(command,"LAsort %s %s %s/%s.%s.N%c",VERBOSE?"-v":"",
                            MAP_ORDER?"-a":"",SORT_PATH,aroot,broot,BLOCK_SYMBOL);
            SYSTEM_CHECK(command)

            sprintf(command,"LAmerge %s %s %s.%s.las %s/%s.%s.N%c.S",VERBOSE?"-v":"",
                            MAP_ORDER?"-a":"",aroot,broot,SORT_PATH,aroot,broot,BLOCK_SYMBOL);
            SYSTEM_CHECK(command)

            if (aroot != broot && SYMMETRIC)
              { sprintf(command,"LAsort %s %s %s/%s.%s.N%c",VERBOSE?"-v":"",
                                MAP_ORDER?"-a":"",SORT_PATH,broot,aroot,BLOCK_SYMBOL);
                SYSTEM_CHECK(command)

                sprintf(command,"LAmerge %s %s %s.%s.las %s/%s.%s.N%c.S",VERBOSE?"-v":"",
                                MAP_ORDER?"-a":"",broot,aroot,SORT_PATH,broot,aroot,BLOCK_SYMBOL);
                SYSTEM_CHECK(command)
              }
*/

            if (aroot != broot)
              free(broot);
          }

        // Free_Block_Arg(parse);
      }
  }

  free(aindex);
  free(apath);
  free(aroot);
  Clean_Exit(0);
}
