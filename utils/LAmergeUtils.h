#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <sys/stat.h>

#include "db/DB.h"
#include "lib/pass.h"
#include "dalign/align.h"
#include "dalign/filter.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define MEMORY 1000   // in Mb
#define MAX_FWAY_MERGE 255

// used in sortAndMerge(MERGE_OPT *mopt) to sort ovls
// according: aread, bread, COMP, abpos
int SORT_OVL(const void *x, const void *y);

//  Input block data structure and block fetcher
typedef struct {
	FILE *stream;
	char *block;
	char *ptr;
	char *top;
	int64 count;
} IO_block;

void reheap(int s, Overlap **heap, int hsize);

void ovl_reload(IO_block *in, int64 bsize);

void showheap(Overlap **heap, int hsize);

typedef struct
{
	// general parameter
	int VERBOSE;
	int KEEP;       // keep intermediate merge files (default: 0)
	int SORT;       // sort initial input files (default: 0)
	int CHECK_TRACE_POINTS;
	int CHECK_SORT_ORDER;
	int CHECK_NAME;

	char *pathDB;   // full path of database
	char *nameDB;   // Database name (without .db extension)
	HITS_DB *db;    // database, used for input file checks

	char *dir;      // all files within the directory will be merged, (if they belong to the given DB)
	char *suffix;   // file suffix, if present only those files that match the pattern DB.blockID.suffix will be merged

	int nBlocks;    // database blocks
	int fway;       // parallel merge

	char *inputFileList; // if -f option is given
	// merge on files
	int maxIFiles;
	int numOfFilesToMerge;
	char **iFileNames;  // input file names

	char *oFile;        // output file name
} MERGE_OPT;

typedef struct
{
    HITS_DB* db;
    ovl_header_twidth twidth;

    int error;              // file didn't pass check

    int check_ptp;          // pass through points
    int check_sort;         // sort order

    ovl_header_novl novl;   // Overlaps counted

    int prev_a;

} CheckContext;


void printUsage(char *prog, FILE* out);
MERGE_OPT* parseMergeOptions(int argc, char* argv[]);
void printOptions(FILE* out, MERGE_OPT* mopt);

void clearMergeOptions(MERGE_OPT* mopt);
int checkOverlapFile(MERGE_OPT *mopt, char *filename, int silent);
void addInputFile(MERGE_OPT *mopt, char *fileName);
void getFilesFromDir(MERGE_OPT *mopt, char *dirName);
void getFilesFromFile(MERGE_OPT *mopt);
void getFilesBySuffix(MERGE_OPT *mopt);
