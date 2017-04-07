#ifndef DEXTRACT_UTILS_H_
#define DEXTRACT_UTILS_H_

#include "stdio.h"
#include <math.h>
#include <sys/stat.h>
#include <ctype.h>
#include <hdf5.h>

typedef unsigned long long uint64;
typedef unsigned int       uint32;
typedef unsigned short     uint16;
typedef unsigned char      uint8;
typedef signed long long   int64;
typedef signed int         int32;
typedef signed short       int16;
typedef signed char        int8;

// Exception codes

#define CANNOT_OPEN_BAX_FILE        -1
#define BAX_BASECALL_ERR            -2
#define BAX_DELETIONQV_ERR          -3
#define BAX_DELETIONTAG_ERR         -4
#define BAX_INSERTIONQV_ERR         -5
#define BAX_MERGEQV_ERR             -6
#define BAX_SUBSTITUTIONQV_ERR      -7
#define BAX_QV_ERR                  -8
#define BAX_NR_EVENTS_ERR           -9
#define BAX_REGION_ERR              -10
#define BAX_HOLESTATUS_ERR          -11
#define BAX_WIDTHINFRAMES_ERR       -12
#define BAX_PREBASEFRAMES_ERR       -13
#define BAX_HQREGIONSTARTTIME_ERR   -14
#define BAX_HQREGIONENDTIME_ERR     -15
#define BAX_PAUSINESS_ERR 			-16
#define BAX_PRODUCTIVITY_ERR    	-17
#define BAX_READTYPE_ERR     	    -18
#define IGNORE_BAX                  -19

#define MAX_READ_LEN    100000
#define MAX_TIME_LIMIT  36000     			// 10 hours
#define FRAME_RATE      75.0001831055
#define MAX_SUBREADS    1000
#define PHRED_OFFSET 	33

#define NUC_COUNT   0
#define QV_SUM      1
#define DEL_SUM     2
#define INS_SUM     3
#define MER_SUM     4
#define SUB_SUM     5

//  Print an error message
void printBaxError(int errorCode);
#define  COMMA  ','
void Print_Number(FILE* out, int64 num, int width);

enum subreadSelection { best, longest, shortest, all};
enum holeStatus { SEQUENCING, ANTIHOLE, FIDUCIAL, SUSPECT, ANTIMIRROR, FDZMW, FBZMW, ANTIBEAMLET, OUTSIDEFOV, UNKNOWN };
enum bases { BASE_A=0, BASE_C=1, BASE_G=2, BASE_T=3, BASE_N=4};
typedef enum { prod_Empty = 0, prod_Productive = 1, prod_Other = 2, prod_NotDefined = 255} productivity;
typedef enum { type_Empty = 0, type_FullHqRead0 = 1, type_FullHqRead1 = 2, type_PartialHqRead0 = 3, type_PartialHqRead1 = 4, type_PartialHqRead2 = 5, type_Multiload = 6, type_Indeterminate = 7, type_NotDefined = 255} readType ;

typedef struct
{
	char *statOut;
	char *fastaOut;
	char *fastqOut;
	char *quivaOut;
	char *baxInFileName;    // argument -F
	char *wellNumbersInFileName;    // argumen -w

	int  *numWellNumbers;
	int  **wellNumbers;
	char **baxIn;
	int    nBax;
	int    nMaxBax;
	int    curBaxFile;

	FILE *statFile;
	FILE *fastaFile;
	FILE *fastqFile;
	FILE *quivaFile;

	int zmw_minNrOfSubReads;
	int MIN_LEN;
	int MAX_LEN;
	int MIN_QV;
	int CUMULATIVE;
	int READLEN_BIN_SIZE;
	int TIME_BIN_SIZE;
	int MIN_MOVIE_TIME;
	int MAX_MOVIE_TIME;
	int VERBOSE;

	enum subreadSelection subreadSel;
} BAX_OPT;

BAX_OPT* parseBaxOptions(int argc, char ** argv);
void initBaxOptions(BAX_OPT *bopt);
void freeBaxOptions(BAX_OPT *bopt);
void printBaxOptions(BAX_OPT *bopt);
void readInputBaxFromFile(BAX_OPT *bopt);
void readWellNumbersFromFile(BAX_OPT *bopt);

typedef struct
{
	char *fullName;      // full file path
	int  shortNameBeg;   // without path and file extension (used in header line)
	int  shortNameEnd;

	// streams that are parsed from each bax file (per base)
	unsigned char *baseCall;     // base calls (ACGT)
	unsigned char *delQV; 				// probability of a deletion error prior to the current base (PHRED QV)
	unsigned char *delTag; 				// Likely identity of the deleted base, if it exists.
	unsigned char *insQV; 				// probability that the current base is an insertion (PHRED QV)
	unsigned char *mergeQV; 			// probability of a merged-pulse error at the current base (PHRED QV)
	unsigned char *subQV; 				// probability of a substitution error  at the current base (PHRED QV)
	unsigned char *fastQV; 			// probability of a base calling error  at the current base (PHRED QV)

	unsigned short *widthInFrames; // duration of the base incorporation event, in frames
	unsigned short *preBaseFrames; // duration between start of the base and the end of the previous base, in frames

	char *holeStatus; 	// type of ZMW that produced the data (0 = SEQUENCING, 1 = ANTIHOLE, 2 = FIDUCIAL, 3 = SUSPECT,
											// 4 = ANTIMIRROR, 5 = FDZMW, 6 = FBZMW, 7 = ANTIBEAMLET, 8 = OUTSIDEFOV)
	int *numEvent;      // event counts per zmw on the cell
	int *region;        // Regions table (read annotation)

	int numZMW;         // number of ZMW (Holes)
	int numRegion;      // number of region rows
	int numBase;        // number of raw bases

	float *hqRegionBegTime, *hqRegionEndTime;   // Start/End time of the HQ (Sequencing) region, in seconds (per ZMW)
	float *pausiness;						    // Fraction of pause events over the HQ (sequencing) region
	unsigned char *productivity;			    // ZMW productivity classification --> UnitsOrEncoding = 0:Empty,1:Productive,2:Other,255:NotDefined
	unsigned char *readType; 				    // ZMW read type classification --> UnitsOrEncoding = 0:Empty,1:FullHqRead0,2:FullHqRead1,3:PartialHqRead0,4:PartialHqRead1,5:PartialHqRead2,6:Multiload,7:Indeterminate,255:NotDefined

	// information about the sequencing itself --> necessary for quiver
	char *sequencingKit;
	char *bindingKit;
	char *softwareVersion;
	char *sequencingChemistry; // optional

} BaxData;

void initBaxData(BaxData *b);
void ensureCapacity(BaxData *b, hsize_t numBaseCalls, hsize_t numHoles, hsize_t numHQReads);
void freeBaxData(BaxData *b);
void initBaxNames(BaxData *b, char *fname);

typedef struct
{
	int minLen;
	int minScore;
	int cumulative;
	int readLenBinSize;
	int timeLenBinSize;
	int minMovieTime;
	int maxMovieTime;

	int nLenBins;
	int nTimBins;

	int nFiles;
	uint64 nZMWs;
	uint64 readTypeHist[type_NotDefined+1];
	uint64 productiveHist[prod_NotDefined+1];
	uint64 stateHist[UNKNOWN+1];
	float cumPausiness;

	uint64 numSubreadBases;
	uint64 numSubreads;
	uint64 subreadHist[MAX_SUBREADS + 1];


	uint64 *readLengthHist, *readLengthBasesHist, *readLengthTimeHist; // count read lengths and corresponding number of bases
	uint64 **baseDistributionHist;       		// bins: 0 = A, 1 = C, 2 = G, 3 = T, 4 otherwise
	uint64 **cumTimeDepQVs;                 // accessible via above definition
	uint64 *cumSlowPolymeraseRegionLenHist, *nSlowPolymeraseRegionLenHist;
	uint64 *cumSlowPolymeraseRegionTimeHist;


} BaxStatistic;

void initBaxStatistic(BaxStatistic *s, BAX_OPT *bopt);
void resetBaxStatistic(BaxStatistic *s);
void freeBaxStatistic(BaxStatistic* s);

typedef struct
{
	int nRegions; 					// number of slow polymerase regions
	int segmentWidth;			  // number of bases that are grouped together
	int shift;							// number of bases a segment window is shifted
	int nmax;		  					// allocated memory
	int* beg;		  					// begin of "slow/bad" region
	int* end;		  					// end of "slow/bad" region
	int numSlowBases;				//
} slowPolymeraseRegions;

void initSlowPolymeraseRegions(slowPolymeraseRegions *spr, int segmentWidth, int shift);
void resetSlowPolymeraseRegions(slowPolymeraseRegions *spr);
void ensureSlowPolymeraseRegionsCapacity(slowPolymeraseRegions *spr);
void deleteSlowPolymeraseRegions(slowPolymeraseRegions *spr);
int isBaseInSlowPolymeraseRegion(slowPolymeraseRegions *spr, int baseIdx);

typedef struct
{
	int number;				// one smart cell is usually stored in 3 files --> hole numbers: 1-54493, 54494-105k, 105k-160k
	int index;				// keeps index of hole number for current file
	int regionRow;		// keeps row from region table for current file

	int hqBeg;				// begin of high quality region within ZMW (corresponds to 2nd column of Regions table)
	int hqEnd;				// end of high quality region within ZMW (corresponds to 3rd column of Regions table)
	int *insBeg;			// begin of insert region within ZMW, can be more the one if adapter are present (corresponds to 2nd column of Regions table)
	int *insEnd;			// end of insert region within ZMW, can be more the one if adapter are present (corresponds to 3rd column of Regions table)
	int *insTimeBeg;	// in number of frames
	int *insTimeEnd;	// in number of frames
	int regionScore;	// corresponds to complete high quality region

	enum holeStatus status;
	productivity prod;
	readType type;
	float pausiness;

	char *toReport; 	// [0|1] flags to specify which subreads from a ZMW should be reported

	int numFrag;
	int maxFrag;
	int roff;					// read offset

	unsigned char **fragSequ;								// keeps pointer on BaxData.basecall of current ZMW
	unsigned char **fragQual;								// keeps pointer on BaxData.fastQV of current ZMW
	unsigned short **widthInFrames; 				// keeps pointer on BaxData.widthInFrames of current ZMW
	unsigned short **preBaseFrames;					// keeps pointer on BaxData.preBaseFrames of current ZMW
	unsigned char **delQV;						 			// keeps pointer on BaxData.delQV of current ZMW
	unsigned char **delTag;						 			// keeps pointer on BaxData.delTag of current ZMW
	unsigned char **insQV; 									// keeps pointer on BaxData.insQV of current ZMW
	unsigned char **mergeQV; 								// keeps pointer on BaxData.mergeQV of current ZMW
	unsigned char **subQV; 									// keeps pointer on BaxData.subQV of current ZMW

	int   *len;
	float *avgQV;			// average qv per subread, based ion fragQV

	slowPolymeraseRegions *spr;
} ZMW;

void initZMW(ZMW *z);
void resetZMW(ZMW *z);
void ensureZMWCapacity(ZMW *z);
void deleteZMW(ZMW *z);
void printZMW(ZMW *z);

///////////////////////////////// SOME GENERAL STUFF
#define SQR(a) ((a)*(a))

// estimates based on 2 data series (used for widthinframes+prebaseframes)
void ln_estimate2(unsigned short* data1, unsigned short* data2, int beg, int end, double* mu, double* sig);
void n_estimate2(unsigned short* data1, unsigned short* data2, int beg, int end, double* mu, double* sig);
char *trimwhitespace(char *str);
int parse_ranges(char *line, int* _reps, int** _pts);
int cmp_range(const void* l, const void* r);

#endif /* DEXTRACT_UTILS_H_ */
