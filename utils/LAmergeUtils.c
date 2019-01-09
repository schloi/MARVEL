#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <ctype.h>

#include "LAmergeUtils.h"
#include "lib/oflags.h"

void printUsage( char* prog, FILE* out )
{
    fprintf( out, "usage: %s [-hksv] [-C [n|s|S|t|A]] [-n n] [-S string] [-f file] database output.las [input.directory | input.1.las ...]\n\n", prog );

    fprintf( out, "Merge (and sorts) multiple input las files into a single output file.\n\n" );

    fprintf( out, "options:\n" );

    fprintf( out, "  -h  prints this usage info\n" );
    fprintf( out, "  -v  verbose output\n" );
    fprintf( out, "  -s  sort content of the input las files prior to merging\n" );
    fprintf( out, "  -k  keep intermediate merge results\n" );
    fprintf( out, "  -C mode  perform sanity checks. Multiple options are possible.\n" );
    fprintf( out, "     n  file names must be consistent with database.\n" );
    fprintf( out, "     s  ensure ascending read id ordering\n" );
    fprintf( out, "     S  -s + ensure complement and alignment start position ordering\n" );
    fprintf( out, "     t  check alignment trace points\n" );
    fprintf( out, "     A  check all (equals -nSt)\n" );
    fprintf( out, "  -n n  number of input files that are merged simultaneously [2, 255], (Default: 8).\n" );
    fprintf( out, "  -S suffix  specify a file suffix, e.g. ovh, or rescued, (default: not set)\n" );
    fprintf( out, "  -f file  file that contains a list of las files to be merged. (One file per line)\n" );
}

int ends_with(const char *str, const char *suffix)
{
    if (!str || !suffix)
    {
        return 0;
    }

    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);

    if (lensuffix > lenstr)
    {
        return 0;
    }

    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

int SORT_OVL(const void *x, const void *y)
  {
    Overlap* l = (Overlap *) x;
    Overlap* r = (Overlap *) y;

    int al, ar;
    int bl, br;

    al = l->aread;
    bl = l->bread;

    ar = r->aread;
    br = r->bread;

    if (al != ar)
      return (al - ar);

    if (bl != br)
      return (bl - br);

    if (COMP(l->flags) > COMP(r->flags))
      return 1;

    if (COMP(l->flags) < COMP(r->flags))
      return -1;

    return (l->path.abpos - r->path.abpos);
  }

#define COMPARE(lp,rp)                          \
  if (lp->aread > rp->aread)                    \
    bigger = 1;                                 \
  else if (lp->aread < rp->aread)               \
    bigger = 0;                                 \
  else if (lp->bread > rp->bread)               \
    bigger = 1;                                 \
  else if (lp->bread < rp->bread)               \
    bigger = 0;                                 \
  else if (COMP(lp->flags) > COMP(rp->flags))   \
    bigger = 1;                                 \
  else if (COMP(lp->flags) < COMP(rp->flags))   \
    bigger = 0;                                 \
  else if (lp->path.abpos > rp->path.abpos)     \
    bigger = 1;                                 \
  else                                          \
    bigger = 0;

void reheap(int s, Overlap **heap, int hsize)
  {
    int c, l, r;
    int bigger;
    Overlap *hs, *hr, *hl;

    c = s;
    hs = heap[s];
    while ((l = 2 * c) <= hsize)
      {
        r = l + 1;
        hl = heap[l];
        if (r > hsize)
          bigger = 1;
        else
          {
            hr = heap[r];
            COMPARE(hr, hl)
          }
        if (bigger)
          {
            COMPARE(hs, hl)
            if (bigger)
              {
                heap[c] = hl;
                c = l;
              }
            else
              break;
          }
        else
          {
            COMPARE(hs, hr)
            if (bigger)
              {
                heap[c] = hr;
                c = r;
              }
            else
              break;
          }
      }
    if (c != s)
      heap[c] = hs;
  }

void ovl_reload(IO_block *in, int64 bsize)
  {
    int64 remains;

    remains = in->top - in->ptr;
    if (remains > 0)
      memcpy(in->block, in->ptr, remains);
    in->ptr = in->block;
    in->top = in->block + remains;
    in->top += fread(in->top, 1, bsize - remains, in->stream);
  }

void showheap(Overlap **heap, int hsize)
  {
    int i;
    printf("\n");
    for (i = 1; i <= hsize; i++)
      printf(" %3d: %5d, %5d\n", i, heap[i]->aread, heap[i]->bread);
  }

static void check_pre(PassContext* pctx, CheckContext* cctx)
  {
    cctx->twidth = pctx->twidth;
    cctx->prev_a = 0;
  }

static void check_post(PassContext* pctx, CheckContext* cctx, char *filename)
  {
    if (!cctx->error && pctx->novl != cctx->novl)
      {
        fprintf(stderr, "[ERROR] - LAmerge: In file: %s -> novl of %lld doesn't match actual overlap count of %lld\n", filename, pctx->novl, cctx->novl);
        cctx->error = 1;
      }
  }

#define CMP(a, b) cmp = (a) - (b); if (cmp != 0) return cmp;

inline static int compare_sort(Overlap* o1, Overlap* o2, int sort)
  {
    int cmp;

    if (sort == 0)
      return 0;

    if (sort == 1)
      {
        CMP(o1->aread, o2->aread);
        CMP(o1->bread, o2->bread);
      }
    else
      {
        CMP(o1->aread, o2->aread);
        CMP(o1->bread, o2->bread);
        CMP(o1->flags & OVL_COMP, o2->flags & OVL_COMP);
        CMP(o1->path.abpos, o2->path.abpos);
      }

    return cmp;
  }

static int check_process(void* _ctx, Overlap* ovl, int novl)
  {
    CheckContext* ctx = (CheckContext*) _ctx;

    int i, lena, lenb;

    for (i = 0; i < novl; i++)
      {
        ctx->novl++;

        if (i == 0)
          {
            if (ctx->check_sort && ctx->prev_a > ovl[i].aread)
              {
                fprintf(stderr, "overlap %lld: not sorted\n", ctx->novl);
                ctx->error = 1;
              }
          }
        else
          {
            int cmp = compare_sort(ovl + (i - 1), ovl + i, ctx->check_sort);

            if (cmp > 0 && ctx->check_sort)
              {
                printf("%d %d\n", ovl[i - 1].aread, ovl[i - 1].bread);

                fprintf(stderr, "overlap %lld: not sorted\n", ctx->novl);
                ctx->error = 1;
              }
          }

        lena = DB_READ_LEN(ctx->db, ovl[i].aread);
        lenb = DB_READ_LEN(ctx->db, ovl[i].bread);

        if (ovl[i].path.abpos < 0)
          {
            fprintf(stderr, "overlap %lld: abpos < 0\n", ctx->novl);
            ctx->error = 1;
          }

        if (ovl[i].path.bbpos < 0)
          {
            fprintf(stderr, "overlap %lld: bbpos < 0\n", ctx->novl);
            ctx->error = 1;
          }

        if (ovl[i].path.aepos > lena)
          {
            fprintf(stderr, "overlap %lld: aepos > lena\n", ctx->novl);
            ctx->error = 1;
          }

        if (ovl[i].path.bepos > lenb)
          {
            fprintf(stderr, "overlap %lld: bepos > lenb\n", ctx->novl);
            ctx->error = 1;
          }

        if (ovl[i].path.tlen < 0)
          {
            fprintf(stderr, "overlap %lld: invalid tlen %d\n", ctx->novl, ovl[i].path.tlen);
            ctx->error = 1;
          }

        if (ctx->check_ptp)
          {
            ovl_trace* trace = ovl[i].path.trace;

            int apos = ovl[i].path.abpos;
            int bpos = ovl[i].path.bbpos;

            int j;
            for (j = 0; j < ovl[i].path.tlen; j += 2)
              {
                apos += (apos / ctx->twidth + 1) * ctx->twidth;
                bpos += trace[j + 1];
              }

            if (bpos != ovl[i].path.bepos)
              {
                fprintf(stderr, "overlap %lld (%d x %d): pass-through points inconsistent be = %d (expected %d)\n", ctx->novl, ovl[i].aread, ovl[i].bread, bpos, ovl[i].path.bepos);
                ctx->error = 1;
              }
          }
      }

    ctx->prev_a = ovl->aread;

    return !ctx->error;
  }

int checkOverlapFile(MERGE_OPT *mopt, char *filename, int silent)
  {
    if (!silent || mopt->VERBOSE > 2)
      printf("Check file: %s\n", filename);

    // general sanity checks

    // check if file empty
      {
        struct stat st;
        stat(filename, &st);
        if (st.st_size == 0)
          {
            if (!silent || mopt->VERBOSE > 2)
              printf(" --> failed (Overlap file %s is empty)\n", filename);
            return 1;
          }
      }

    // check if file is accessible and has valid header
      {
        FILE *fileOvlIn;
        if ((fileOvlIn = fopen(filename, "r")) == NULL)
          {
            if (!silent || mopt->VERBOSE > 2)
              printf(" --> failed could not open '%s'\n", filename);

            return 1;
          }

        ovl_header_novl novl;
        ovl_header_twidth twidth;
        if (!ovl_header_read(fileOvlIn, &(novl), &(twidth)))
          {
            if (!silent || mopt->VERBOSE > 2)
              printf(" --> failed invalid header in file '%s'\n", filename);

            fclose(fileOvlIn);
            return 1;
          }
        fclose(fileOvlIn);
      }

    // check file name
    if (mopt->CHECK_NAME)
      {
        // check if filename has .las file extension
          {
            char * pch;
            pch = strrchr(filename, '.');
            if (pch == NULL)
              {
                if (mopt->VERBOSE > 2)
                  printf(" --> failed (No .las file extension)\n");
                else if (!silent)
                  fprintf(stderr, "[WARNING] - LAmerge: Overlap file %s does not have the proper .las file extension.\n", filename);

                return 1;
              }
            if (strcmp(pch, ".las") != 0)
              {
                if (mopt->VERBOSE > 2)
                  printf(" --> failed (No .las file extension)\n");
                else if (!silent)
                  fprintf(stderr, "[WARNING] - LAmerge: Overlap file %s does not have the proper .las file extension.\n", filename);

                return 1;
              }

            // check if file matches suffix (if present)
            if (strlen(mopt->suffix) > 0)
              {
                *pch = '\0';
                char * ppch;
                ppch = strrchr(filename, '.');
                if (ppch == NULL)
                  {
                    if (mopt->VERBOSE > 2)
                      printf(" --> failed (Overlap file %s.las does not match suffix pattern %s)\n", filename, mopt->suffix);
                    else if (!silent)
                      fprintf(stderr, "[WARNING] - LAmerge: Overlap file %s.las does not match suffix pattern %s\n", filename, mopt->suffix);

                    return 1;
                  }
                if (strcmp(ppch, mopt->suffix) != 0)
                  {
                    if (mopt->VERBOSE > 2)
                      printf(" --> failed (Overlap file %s.las does not match suffix pattern %s)\n", filename, mopt->suffix);
                    else if (!silent)
                      fprintf(stderr, "[WARNING] - LAmerge: Overlap file %s.las does not match suffix pattern %s\n", filename, mopt->suffix);

                    return 1;
                  }
                *pch = '.';
              }
          }

        // check if filename, belongs to database
          {
            char *root = Root(filename, ".las");

            if (strncmp(mopt->nameDB, root, strlen(mopt->nameDB)) != 0)
              {
                if (mopt->VERBOSE > 2)
                  printf(" --> failed (Overlap file name %s does not match database name %s)\n", root, mopt->nameDB);
                free(root);
                return 1;
              }
            free(root);
          }

        if (!silent || mopt->VERBOSE > 2)
          printf(" --> succeeded\n");
      }

    // check
    if (mopt->CHECK_SORT_ORDER || mopt->CHECK_TRACE_POINTS)
      {
          FILE* fileOvlIn;
          if ( ( fileOvlIn = fopen( filename, "r" ) ) == NULL )
          {
            if (!silent)
              fprintf(stderr, "could not open '%s'\n", filename);

            return 1;
          }

        PassContext* pctx;
        CheckContext cctx;

        bzero(&cctx, sizeof(CheckContext));
        cctx.db = mopt->db;

        cctx.check_ptp = mopt->CHECK_TRACE_POINTS;
        cctx.check_sort = mopt->CHECK_SORT_ORDER;

        pctx = pass_init(fileOvlIn, NULL);
        pctx->split_b = 0;
        pctx->load_trace = cctx.check_ptp;
        pctx->unpack_trace = cctx.check_ptp;
        pctx->data = &cctx;

        check_pre(pctx, &cctx);

        pass(pctx, check_process);

        check_post(pctx, &cctx, filename);

        pass_free(pctx);

        fclose(fileOvlIn);
        return cctx.error;
      }

    return 0;
  }

void addInputFile(MERGE_OPT *mopt, char *fileName)
  {
    if (mopt->numOfFilesToMerge == mopt->maxIFiles)
      {
        mopt->maxIFiles = (mopt->maxIFiles * 1.2 + 100);
        mopt->iFileNames = (char**) realloc(mopt->iFileNames, sizeof(char*) * mopt->maxIFiles);
        if (mopt->iFileNames == NULL)
          {
            fprintf(stderr, "[ERROR] - LAmerge: Unable to increase input file buffer!\n");
            exit(1);
          }
      }
    int len = strlen(fileName);

    mopt->iFileNames[mopt->numOfFilesToMerge] = malloc(len + 10);
    if (mopt->iFileNames[mopt->numOfFilesToMerge] == NULL)
      {
        fprintf(stderr, "[ERROR] - LAmerge: Unable to allocate file name buffer for %s\n", fileName);
        exit(1);
      }
    memcpy(mopt->iFileNames[mopt->numOfFilesToMerge], fileName, len + 1);
    if (mopt->VERBOSE)
      printf("Add input file: %s\n", fileName);
    mopt->numOfFilesToMerge++;
  }

static char *trimFileString(char *str)
  {
    char *end;

    // Trim leading space
    while (isspace(*str))
      str++;

    if (*str == 0)  // All spaces?
      return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace(*end))
      end--;

    // Write new null terminator
    *(end + 1) = 0;

    return str;
  }

void getFilesBySuffix(MERGE_OPT *mopt)
  {
    char *fileName;

    fileName = malloc(strlen(mopt->suffix) + strlen(mopt->pathDB) + 20);
    if (fileName == NULL)
      {
        fprintf(stderr, "[ERROR] - LAmerge : Cannot allocate file name buffer\n");
        exit(1);
      }

    char *dbPath = PathTo(mopt->pathDB);

    // look for files in the same directory where the database is located
    int i;
    int failed = 0;
    for (i = 1; i <= mopt->nBlocks; i++)
      {
        sprintf(fileName, "%s/%s.%d%s.las", dbPath, mopt->nameDB, i, mopt->suffix);
        if (checkOverlapFile(mopt, fileName, 1))
          {
            fprintf(stderr, "[ERROR] - LAmerge : Overlap file %s did not pass check!\n", fileName);
            failed = 1;
          }
      }
    if (failed)
      {
        fprintf(stderr, "[ERROR] - LAmerge : Some overlap files are not valid. Merging by suffix requires "
            "that all Overlaps files [1, numBlocks] to be valid! Stop here!\n");
        exit(1);
      }
    for (i = 1; i <= mopt->nBlocks; i++)
      {
        sprintf(fileName, "%s/%s.%d%s.las", dbPath, mopt->nameDB, i, mopt->suffix);
        addInputFile(mopt, fileName);
      }
    free(dbPath);
    free(fileName);
  }

void getFilesFromFile(MERGE_OPT* mopt)
  {
    if (mopt->inputFileList == NULL)
      return;

    FILE *in = fopen(mopt->inputFileList, "r");

    if (in == NULL)
      {
        fprintf(stderr, "[WARNING] - LAmerge : Cannot open input file %s\n.\n", mopt->inputFileList);
        return;
      }

    const size_t line_size = MAX_NAME;
    char* line = malloc(line_size);
    int failed = 0;
    while (fgets(line, line_size, in) != NULL)
      {
        // get rid of newline and blanks
        char *mline = trimFileString(line);
        if(checkOverlapFile(mopt, mline, 0))
          failed++;
        else
          addInputFile(mopt, mline);
      }
    free(line);
    fclose(in);

    if (failed)
      {
        fprintf(stderr, "[ERROR] - LAmerge : %d files do not pass check. Stop!\n", failed);
        exit(1);
      }
  }

void getFilesFromDir(MERGE_OPT *mopt, char *dirName)
  {
    DIR *dir;
    struct dirent *ent;
    char *fullPath;
    int fileNameLen;
    int dirNameLen = strlen(dirName);
    int fullPathLen = dirNameLen + 2 * strlen(mopt->nameDB) + 100;
    fullPath = (char*) malloc(fullPathLen);

    int failed = 0;
    if ((dir = opendir(dirName)) != NULL)
      {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL)
          {
            fileNameLen = strlen(ent->d_name);
            if (fileNameLen + 2 + dirNameLen > fullPathLen)
              {
                fullPathLen = (fileNameLen + 2 + dirNameLen) * 1.2;
                fullPath = (char*) realloc(fullPath, fullPathLen);
                if (fullPath == NULL)
                  {
                    fprintf(stderr, "[WARNING]: Cannot increase path buffer. Skip file: %s/%s.\n", dirName, ent->d_name);
                    continue;
                  }
              }

            if( ent->d_name[0] == '.' || !ends_with(ent->d_name, ".las") )
            {
              continue;
            }

            sprintf(fullPath, "%s/%s", dirName, ent->d_name);

            if(checkOverlapFile(mopt, fullPath, 1))
              failed++;
            else
              addInputFile(mopt, fullPath);
          }
        closedir(dir);
      }
    else
      fprintf(stderr, "[WARNING] - LAmerge: Could not open directory: %s\n", dirName);

    if (failed)
      {
        fprintf(stderr, "[ERROR]: %d input overlap files are corrupt. Stop!\n", failed);
        exit(1);
      }

    free(fullPath);
  }

MERGE_OPT* parseMergeOptions(int argc, char* argv[])
  {
    MERGE_OPT *mopt = (MERGE_OPT*) malloc(sizeof(MERGE_OPT));

    // set default values
    mopt->VERBOSE = 0;
    mopt->KEEP = 0;
    mopt->SORT = 0;
    mopt->fway = 8;
    mopt->CHECK_TRACE_POINTS = 0;
    mopt->CHECK_SORT_ORDER = 0;
    mopt->CHECK_NAME = 0;
    mopt->suffix = (char*) malloc(10);
    mopt->suffix[0] = '\0';
    mopt->dir = (char*) malloc(10);
    sprintf(mopt->dir, ".");
    mopt->numOfFilesToMerge = 0;
    mopt->oFile = NULL;
    mopt->maxIFiles = 100;
    mopt->iFileNames = (char**) malloc(sizeof(char*) * mopt->maxIFiles);
    mopt->inputFileList = NULL;
    int c;
    while (1)
      {
        static struct option long_options[] =
        {
        { "help", no_argument, 0, 'h' },
        { "keep", no_argument, 0, 'k' },
        { "sort", no_argument, 0, 's' },
        { "verbose", no_argument, 0, 'v' },
        { "nFiles", required_argument, 0, 'n' },
        { "in", required_argument, 0, 'f' },
        { "suffix", required_argument, 0, 'S' },
        { "check", required_argument, 0, 'C' } };

        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long(argc, argv, "hksvn:S:f:C:", long_options, &option_index);

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

          case 'h':
            printUsage(argv[0], stderr);
            exit(1);
          case 'v':
            mopt->VERBOSE++;
            break;
          case 'k':
            mopt->KEEP = 1;
            break;
          case 's':
            mopt->SORT = 1;
            break;
          case '?':
            printUsage(argv[0], stderr);
            exit(1);
          case 'n':
            {
              mopt->fway = (int) strtol(optarg, NULL, 10);
              if (errno)
                {
                  fprintf(stderr, "Cannot parse argument of numFiles (-j ARG)! Must be an integer in [2, 255]\n");
                  exit(1);
                }
              if (mopt->fway < 2 || mopt->fway > 255)
                {
                  fprintf(stderr, "Number of files to merge is not accpeted! Must be an integer in [2, 255]\n");
                  exit(1);
                }
            }
            break;
          case 'C':
            {
              int len = strlen(optarg);
              int i;
              for (i = 0; i < len; i++)
                {
                  switch (optarg[i])
                  {
                    case 'A':
                      mopt->CHECK_NAME = 1;
                      mopt->CHECK_TRACE_POINTS = 1;
                      mopt->CHECK_SORT_ORDER = 2;
                      break;
                    case 's':
                      if (mopt->CHECK_SORT_ORDER < 2)
                        mopt->CHECK_SORT_ORDER = 1;
                      break;
                    case 'S':
                      mopt->CHECK_SORT_ORDER = 2;
                      break;
                    case 't':
                      mopt->CHECK_TRACE_POINTS = 1;
                      break;
                    case 'n':
                      mopt->CHECK_NAME = 1;
                      break;
                    default:
                      fprintf(stderr, "[WARNING] - LAmerge: skip unknown check option: %c.\n", optarg[i]);
                      break;
                  }
                }
            }
            break;
          case 'S':
            {
              mopt->suffix = (char*) realloc(mopt->suffix, strlen(optarg) + 5);
              if (strlen(optarg) == 0)
                {
                  /* suffix remains empty, i.e. merge all overlap files DB.1.las ... DB.Nblocks.las */
                  mopt->suffix[0]='\0';
                }
              else if (optarg[0] == '.')
                sprintf(mopt->suffix, "%s", optarg);
              else
                sprintf(mopt->suffix, ".%s", optarg);
            }
            break;
          case 'f':
            mopt->inputFileList = optarg;
            break;
          default:
            printUsage(argv[0], stderr);
            exit(1);
        }
      }

    if (optind + 1 > argc)
      {
        fprintf(stderr, "At least a database is required!\n");
        printUsage(argv[0], stderr);
        exit(1);
      }

    // if -s option is used: reset CHECK_SORT to 0, the final overlap file is always checked for CHECK_SORT=1
    if(mopt->SORT)
      mopt->CHECK_SORT_ORDER=0;

    // parse database
      {
        int len = strlen(argv[optind]);
        mopt->pathDB = malloc(len + 10);
        memcpy(mopt->pathDB, argv[optind], len + 1);

        // add .db extension if not present
        if ((len < 3) || ((strcasecmp(argv[optind] + (len - 3), ".db") != 0)))
          {
            strncpy(mopt->pathDB + len, ".db", 3);
            mopt->pathDB[len + 3] = '\0';
          }

        // parse number of blocks
        mopt->nBlocks = DB_Blocks(mopt->pathDB);
        if (mopt->nBlocks < 0)
          {
            fprintf(stderr, "Cannot determine number of blocks of database file %s!\n", mopt->pathDB);
            exit(1);
          }

        // set name of db (without directory and without extension)
        mopt->nameDB = Root(mopt->pathDB, ".db");

        // open database
        if ( mopt->CHECK_SORT_ORDER || mopt->CHECK_TRACE_POINTS )
        {
          mopt->db = (HITS_DB*) malloc(sizeof(HITS_DB));
          Open_DB(mopt->pathDB, mopt->db);
          if (mopt->db == NULL)
            {
              fprintf(stderr, "[ERROR] - LAmerge : Cannot open database \'%s\'!\n", mopt->pathDB);
              exit(1);
            }
        }

        optind++;
      }

    if (optind == argc)
      {
        fprintf(stderr, "[ERROR] - LAmerge: The output file is required!\n");
        printUsage(argv[0], stderr);
        exit(1);
      }
    struct stat sb;
    // get output file name, and trim .las extension
      {
        int len = strlen(argv[optind]);
        mopt->oFile = (char*) malloc(len + 10);
        memcpy(mopt->oFile, argv[optind], len + 1);

        if ((len > 4) && ((strcmp(argv[optind] + (len - 4), ".las") == 0)))
          mopt->oFile[len - 4] = '\0';

        if (stat(argv[optind], &sb) != -1)
          {
            if (S_ISDIR(sb.st_mode))
              {
                fprintf(stderr, "[ERROR] - LAmerge: The output overlap file cannot be an existing directory: %s!\n", argv[optind]);
                exit(1);
              }

            if (S_ISREG(sb.st_mode))
              fprintf(stderr, "[WARNING] - LAmerge: The output overlap file %s will be overwritten!\n", argv[optind]);
          }
        optind++;
      }

    while (optind < argc)
      {
        if (stat(argv[optind], &sb) == -1)
          {
            fprintf(stderr, "[ERROR] - LAmerge: Cannot determine file type from argument: %s\n", argv[optind]);
            exit(1);
          }
        if (S_ISDIR(sb.st_mode))
          {
            getFilesFromDir(mopt, argv[optind]);
          }
        else if (S_ISREG(sb.st_mode) || S_ISLNK(sb.st_mode))
          {
            if (checkOverlapFile(mopt, argv[optind], 1))
              fprintf(stderr, "[WARNING] - LAmerge: skip file %s. File check fails!", argv[optind]);
            else
              addInputFile(mopt, argv[optind]);
          }
        else
          fprintf(stderr, "[WARNING] - LAmerge: Cannot handle file type from argument: %s\n", argv[optind]);
        optind++;
      }

    // parse file with list of overlap files (each line contains one overlap file)
    getFilesFromFile(mopt);

    // get input files solely by suffix
    if (mopt->numOfFilesToMerge == 0)
      {
        getFilesBySuffix(mopt);
      }

    return mopt;
  }

void printOptions(FILE* out, MERGE_OPT* mopt)
  {
    if (mopt && out)
      {
        fprintf(out, "############ merge options ###########\n");
        fprintf(out, "VERBOSE:     %d\n", mopt->VERBOSE);
        fprintf(out, "KEEP:        %d\n", mopt->KEEP);
        fprintf(out, "SORT:        %d\n", mopt->SORT);
        fprintf(out, "#DB BLOCKS:  %d\n", mopt->nBlocks);
        fprintf(out, "#FWAY MERGE: %d\n", mopt->fway);
        fprintf(out, "OUT:         %s.las\n", mopt->oFile);
        fprintf(out, "NUM:         %d\n", mopt->numOfFilesToMerge);
        int i;
        for (i = 0; i < mopt->numOfFilesToMerge; i++)
          fprintf(out, "IN [%3d]:    %s.las\n", i + 1, mopt->iFileNames[i]);
        fprintf(out, "DIR:         %s\n", mopt->dir);
        fprintf(out, "DB PATH:     %s\n", mopt->pathDB);
        fprintf(out, "DB NAME:     %s\n", mopt->nameDB);
      }
  }

void clearMergeOptions(MERGE_OPT* mopt)
  {
    if (!mopt)
      return;

    if (mopt->dir)
      free(mopt->dir);
    if (mopt->pathDB)
      free(mopt->pathDB);
    if (mopt->nameDB)
      free(mopt->nameDB);
    if (mopt->suffix)
      free(mopt->suffix);

    int i;
    if (mopt->oFile)
      free(mopt->oFile);
    if (mopt->iFileNames)
      {
        for (i = 0; i < mopt->numOfFilesToMerge; i++)
          if (mopt->iFileNames[i])
            free(mopt->iFileNames[i]);
      }
    free(mopt);
  }
