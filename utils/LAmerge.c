
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <math.h>

#include "lib/pass.h"
#include "dalign/align.h"
#include "dalign/filter.h"
#include "LAmergeUtils.h"

#undef DEBUG

// is only called once, and if the sort flag is enabled
void sortFile(char* fout, char* fin, int verbose)
{
	Overlap *allOvls;

	uint64 novls;

	int twidth;
	uint16 *traces;
	uint64 nTraces;
	uint64 tcur;

	uint64 j;
	FILE *inFile, *outFile;

	novls = 0;

	if ((inFile = fopen(fin, "r")) == NULL)
	{
		fprintf(stderr, "[ERROR] - LAmerge: Cannot open file for reading: %s\n",
				fin);
		exit(1);
	}
	if (fread(&novls, sizeof(uint64), 1, inFile) != 1)
	{
		fprintf(stderr,
				"[ERROR] - LAmerge: failed to read header novl of file %s\n",
				fin);
		fclose(inFile);
		exit(1);
	}

	if (fread(&twidth, sizeof(int), 1, inFile) != 1)
	{
		fprintf(stderr,
				"[ERROR] - LAmerge: failed to read header twidth of file %s\n",
				fin);
		fclose(inFile);
		exit(1);
	}

	if (verbose)
		printf("%s, novl: %llu\n", fin, novls);

	// open output file
	outFile = fopen(fout, "w");
	if (!outFile)
	{
		fprintf(stderr, "[ERROR] - LAmerge: Cannot open output file %s\n",
				fout);
		fclose(inFile);
		exit(1);
	}

	allOvls = (Overlap*) malloc(sizeof(Overlap) * novls);
	nTraces = novls * 60;
	traces = (uint16*) malloc(sizeof(uint16) * nTraces);

	// parse all overlaps
	tcur = 0;
	size_t tbytes = TBYTES(twidth);

	{
		for (j = 0; j < novls; j++)
		{
			if (Read_Overlap(inFile, allOvls + j))
				break;

			// check if trace buffer is still sufficient
			if (allOvls[j].path.tlen + tcur > nTraces)
			{
				nTraces = allOvls[j].path.tlen + tcur + 100;
				uint16* traces2 = realloc(traces, nTraces * sizeof(uint16));

				uint64 k;
				for (k = 0; k < j; k++)
					allOvls[k].path.trace = traces2
							+ ((uint16*) (allOvls[k].path.trace) - traces);

				traces = traces2;
			}

			allOvls[j].path.trace = traces + tcur;
			Read_Trace(inFile, allOvls + j, tbytes);

			tcur += allOvls[j].path.tlen;

			if (tbytes == sizeof(uint8))
				Decompress_TraceTo16(allOvls + j);
		}

		// check if j agrees with novls[i]
		assert(j <= novls);
	}

	qsort(allOvls, novls, sizeof(Overlap), SORT_OVL);

	if (verbose)
		fprintf(stdout, "sortFile %llu overlaps\n", j);

	// write header
	fwrite(&novls, sizeof(novls), 1, outFile);
	fwrite(&twidth, sizeof(twidth), 1, outFile);

	for (j = 0; j < novls; j++)
	{
		if (tbytes == sizeof(uint8))
			Compress_TraceTo8(allOvls + j);

		Write_Overlap(outFile, allOvls + j, tbytes);
	}

	// clean up
	fclose(outFile);
	fclose(inFile);

	free(allOvls);
	free(traces);
}

void sortAndMerge(char* fout, char** fin, int numF, int verbose)
{
	assert(fout != NULL);

	if (numF < 2)
	{
		fprintf(stderr, "ERROR: merge requires at least 2 input files!\n");
		exit(1);
	}

	if (numF > MAX_FWAY_MERGE)
	{
		fprintf(stderr, "ERROR: merge cannot merge more then %d files!\n",
				MAX_FWAY_MERGE);
		exit(1);
	}

	Overlap *allOvls;
	uint64 nAllOvls = 0;

	uint64 ovlIdx;

	FILE **inFiles;
	uint64 *novls;

	int twidth;
	uint16 *traces;
	uint64 nTraces;
	uint64 tcur;

	int i;
	uint64 j;

	inFiles = (FILE**) malloc(sizeof(FILE*) * numF);
	novls = (uint64*) malloc(sizeof(uint64) * numF);

	{
		int twidthTMP;
		// open files and parse header
		for (i = 0; i < numF; i++)
		{
			if ((inFiles[i] = fopen(fin[i], "r")) == NULL)
			{
				fprintf(stderr,
						"[ERROR] - LAmerge: cannot open file for reading: %s (%d)\n",
						fin[i], i);
				exit(0);
			}
			if (fread(novls + i, sizeof(novls[0]), 1, inFiles[i]) != 1)
			{
				fprintf(stderr,
						"[ERROR] - LAmerge: failed to read header novl of file %s\n",
						fin[i]);
				exit(1);
			}

			if (fread(&twidthTMP, sizeof(twidthTMP), 1, inFiles[i]) != 1)
			{
				fprintf(stderr,
						"[ERROR] - LAmerge: failed to read header twidth of file %s\n",
						fin[i]);
				exit(1);
			}

			nAllOvls += novls[i];

			if (i == 0)
				twidth = twidthTMP;
			else if (twidth != twidthTMP)
			{
				fprintf(stderr,
						"Cannot merge overlap files with different trace widths!!! (%d != %d)\n",
						twidth, twidthTMP);
				exit(1);
			}
			if (verbose)
				printf("%s, novl: %llu,sumOVLs: %llu\n", fin[i], novls[i],
						nAllOvls);
		}
	}
	allOvls = (Overlap*) malloc(sizeof(Overlap) * nAllOvls);
	nTraces = nAllOvls * 60;
	traces = (uint16*) malloc(sizeof(uint16) * nTraces);

	// try to open output file
	FILE* out = fopen(fout, "w");
	if (!out)
	{
		fprintf(stderr,
				"[ERROR] - LAmerge: cannot open output file %s for writing\n",
				fout);
		exit(1);
	}

	// parse all overlaps
	ovlIdx = 0;
	tcur = 0;
	size_t tbytes = TBYTES(twidth);
	for (i = 0; i < numF; i++)
	{
		FILE *f = inFiles[i];
		for (j = 0; j < novls[i]; j++)
		{
			if (ovlIdx >= nAllOvls) // should never happen, i.e. a header was broken
			{
				fprintf(stderr, "TERROR to many ovls!! %llu >= %llu\n", ovlIdx,
						nAllOvls);
				exit(1);
			}

			if (Read_Overlap(f, allOvls + ovlIdx))
				break;

			// check if trace buffer is still sufficient
			if (allOvls[ovlIdx].path.tlen + tcur > nTraces)
			{
				nTraces = allOvls[ovlIdx].path.tlen + tcur + 100;
				uint16* traces2 = realloc(traces, nTraces * sizeof(uint16));

				uint64 k;
				for (k = 0; k < ovlIdx; k++)
					allOvls[k].path.trace = traces2
							+ ((uint16*) (allOvls[k].path.trace) - traces);

				traces = traces2;
			}

			allOvls[ovlIdx].path.trace = traces + tcur;
			Read_Trace(f, allOvls + ovlIdx, tbytes);

			tcur += allOvls[ovlIdx].path.tlen;

			if (tbytes == sizeof(uint8))
				Decompress_TraceTo16(allOvls + ovlIdx);

			ovlIdx++;
		}

		// check if j agrees with novls[i]
		assert(j <= novls[i]);
	}

	qsort(allOvls, nAllOvls, sizeof(Overlap), SORT_OVL);

	if (verbose)
		fprintf(stdout, "SortAndMerged %llu overlaps\n", ovlIdx);

	// write header
	fwrite(&nAllOvls, sizeof(nAllOvls), 1, out);
	fwrite(&twidth, sizeof(twidth), 1, out);

	for (j = 0; j < nAllOvls; j++)
	{
		if (tbytes == sizeof(uint8))
			Compress_TraceTo8(allOvls + j);

		Write_Overlap(out, allOvls + j, tbytes);
	}

	// clean up
	fclose(out);

	for (i = 0; i < numF; i++)
		fclose(inFiles[i]);

	free(inFiles);
	free(novls);

	free(allOvls);
	free(traces);
}

void merge(char* fout, char** fin, int numInFiles, int verbose)
{
	assert(fout != NULL);

	if (numInFiles < 2)
	{
		fprintf(stderr, "ERROR: merge requires at least 2 input files!\n");
		exit(1);
	}
	if (numInFiles > MAX_FWAY_MERGE)
	{
		fprintf(stderr, "ERROR: merge cannot merge more then %d files!\n",
				MAX_FWAY_MERGE);
		exit(1);
	}

	IO_block * in;
	int64 bsize, osize, psize;
	char *block, *oblock;
	int i, fway;
	Overlap **heap;
	int hsize;
	Overlap *ovls;
	int64 totl;
	int tspace, tbytes;
	FILE *output;
	char *optr, *otop;

//  Open all the input files and initialize their buffers
	fway = numInFiles;
	psize = sizeof(void *);
	osize = sizeof(Overlap) - psize;
	bsize = (MEMORY * 1000000ll) / (fway + 1);
	block = (char *) Malloc(bsize * (fway + 1) + psize,
			"Allocating LAmerge blocks");
	in = (IO_block *) Malloc(sizeof(IO_block) * fway,
			"Allocating LAmerge IO-reacords");
	if (block == NULL || in == NULL)
		exit(1);
	block += psize;

	totl = 0;
	tbytes = 0;
	tspace = 0;
	for (i = 0; i < fway; i++)
	{
		int64 novl;
		int mspace;
		FILE *input;
		char *iblock;

		input = fopen(fin[i], "r");
		if (input == NULL)
		{
			fprintf(stderr,
					"[ERROR] - LAmerge: Cannot open file \"%s\" for reading\n",
					fin[i]);
			exit(1);
		}

		if (fread(&novl, sizeof(int64), 1, input) != 1)
			SYSTEM_ERROR
		totl += novl;
		if (fread(&mspace, sizeof(int), 1, input) != 1)
			SYSTEM_ERROR
		if (i == 0)
		{
			tspace = mspace;
			if (tspace <= TRACE_XOVR)
				tbytes = sizeof(uint8);
			else
				tbytes = sizeof(uint16);
		}
		else if (tspace != mspace)
		{
			fprintf(stderr, "%s: PT-point spacing conflict (%d vs %d)\n",
					Prog_Name, tspace, mspace);
			exit(1);
		}

		in[i].stream = input;
		in[i].block = iblock = block + i * bsize;
		in[i].ptr = iblock;
		in[i].top = iblock + fread(in[i].block, 1, bsize, input);
		in[i].count = 0;
	}

//  Open the output file buffer and write (novl,tspace) header

	{
		output = fopen(fout, "w");
		if (output == NULL)
		{
			fprintf(stderr, "[ERROR] - LAmerge: Cannot open file \"%s\" for writing\n", fout);
			exit(1);
		}

		fwrite(&totl, sizeof(int64), 1, output);
		fwrite(&tspace, sizeof(int), 1, output);

		oblock = block + fway * bsize;
		optr = oblock;
		otop = oblock + bsize;
	}

	if (verbose)
	{
		printf("Merging %d files totalling ", fway);
		Print_Number(totl, 0, stdout);
		printf(" records\n");
	}

//  Initialize the heap

	heap = (Overlap **) Malloc(sizeof(Overlap *) * (fway + 1),
			"Allocating heap");
	ovls = (Overlap *) Malloc(sizeof(Overlap) * fway, "Allocating heap");
	if (heap == NULL || ovls == NULL)
		exit(1);

	hsize = 0;
	for (i = 0; i < fway; i++)
	{
		if (in[i].ptr < in[i].top)
		{
			ovls[i] = *((Overlap *) (in[i].ptr - psize));
			in[i].ptr += osize;
			hsize += 1;
			heap[hsize] = ovls + i;
		}
	}

	if (hsize > 3)
		for (i = hsize / 2; i > 1; i--)
			reheap(i, heap, hsize);

//  While the heap is not empty do

	while (hsize > 0)
	{
		Overlap *ov;
		IO_block *src;
		int64 tsize, span;

		reheap(1, heap, hsize);

		ov = heap[1];
		src = in + (ov - ovls);

		src->count += 1;

		tsize = ov->path.tlen * tbytes;
		span = osize + tsize;
		if (src->ptr + span > src->top)
			ovl_reload(src, bsize);
		if (optr + span > otop)
		{
			fwrite(oblock, 1, optr - oblock, output);
			optr = oblock;
		}

		memcpy(optr, ((char *) ov) + psize, osize);
		optr += osize;
		memcpy(optr, src->ptr, tsize);
		optr += tsize;

		src->ptr += tsize;
		if (src->ptr < src->top)
		{
			*ov = *((Overlap *) (src->ptr - psize));
			src->ptr += osize;
		}
		else
		{
			heap[1] = heap[hsize];
			hsize -= 1;
		}
	}

//  Flush output buffer and wind up

	if (optr > oblock)
		fwrite(oblock, 1, optr - oblock, output);
	fclose(output);

	for (i = 0; i < fway; i++)
		fclose(in[i].stream);

	for (i = 0; i < fway; i++)
		totl -= in[i].count;
	if (totl != 0)
	{
		fprintf(stderr, "ERROR: Did not write all records (%lld)\n", totl);
		exit(1);
	}

	free(ovls);
	free(heap);
	free(in);
	free(block - psize);

}

static void doMergeAll(MERGE_OPT *mopt)
{
	int mergeRounds = 0;
	int tmp = mopt->numOfFilesToMerge;

	char *fout;
	fout = (char*) malloc(strlen(mopt->oFile) + 20);
	sprintf(fout, "%s.las", mopt->oFile);

	while (tmp > 1)
	{
		tmp = ceil(tmp / (double) mopt->fway);
		mergeRounds++;
	}

	if (mergeRounds < 2)
	{
		if (mopt->SORT)
			sortAndMerge(fout, mopt->iFileNames, mopt->numOfFilesToMerge,
					mopt->VERBOSE);
		else
			merge(fout, mopt->iFileNames, mopt->numOfFilesToMerge,
					mopt->VERBOSE);
	}
	else // merging in multiple rounds
	{
		char **tmpIN = (char**) malloc(
				sizeof(char*) * mopt->numOfFilesToMerge * 2);
		char **tmpOUT = tmpIN + mopt->numOfFilesToMerge;
		void *tmp;
		int i, j;
		for (i = 0; i < mopt->numOfFilesToMerge * 2; i++)
			tmpIN[i] = (char*) malloc(MAX_NAME);

		int currentMergeRound = 1;
		int numIn = mopt->numOfFilesToMerge;
		int numOut = 0;

		while (currentMergeRound < mergeRounds)
		{
			for (i = 0; i + mopt->fway < numIn; i += mopt->fway)
			{
				if (currentMergeRound == 1)
				{
#ifdef DEBUG
					printf("mergeIN:");
					for (j = i; j < i + mopt->fway; j++)
					{
						printf(" %s", mopt->iFileNames[j]);
					}
					printf("\nmergeOut: %s\n", tmpOUT[numOut]);
#endif
					sprintf(tmpOUT[numOut], "%s.L%d.%d.las", mopt->oFile,
							currentMergeRound, numOut);
					if (mopt->SORT)
						sortAndMerge(tmpOUT[numOut], (mopt->iFileNames + i),
								mopt->fway, mopt->VERBOSE);
					else
						merge(tmpOUT[numOut], (mopt->iFileNames + i),
								mopt->fway, mopt->VERBOSE);

					numOut++;
				}
				else
				{
#ifdef DEBUG
					printf("mergeIN:");
					for (j = i; j < i + mopt->fway; j++)
					{
						printf(" %s", tmpIN[j]);
					}
					printf("\nmergeOut: %s\n", tmpOUT[numOut]);
#endif
					sprintf(tmpOUT[numOut], "%s.L%d.%d.las", mopt->oFile,
							currentMergeRound, numOut);
					merge(tmpOUT[numOut], (tmpIN + i), mopt->fway,
							mopt->VERBOSE);
					numOut++;
				}
			}
			if (i < numIn)
			{
				// if only a single input file remains, than add it directly to the tmpOUT
				// if SORT is enabled, then sort the file first
				if (i + 1 == numIn)
				{
					if (currentMergeRound == 1)
					{
						if (mopt->SORT)
						{
#ifdef DEBUG
							printf("sortIN: %s", mopt->iFileNames[i]);
							printf("\nmergeOut: %s\n", tmpOUT[numOut]);
#endif
							sprintf(tmpOUT[numOut], "%s.L%d.%d.las",
									mopt->oFile, currentMergeRound, numOut);
							sortFile(tmpOUT[numOut], mopt->iFileNames[i],
									mopt->VERBOSE);
						}
						else
						{
							sprintf(tmpOUT[numOut], "%s", mopt->iFileNames[i]);
#ifdef DEBUG
							printf("Add to mergeOut: %s\n", tmpOUT[numOut]);
#endif
						}
						numOut++;
					}
					else
					{
						sprintf(tmpOUT[numOut], "%s", tmpIN[i]);
#ifdef DEBUG
						printf("Add to mergeOut: %s\n", tmpOUT[numOut]);
#endif
						// reduce number of numIn by 1, as this file is used in the next merge round, an should not be removed now!!!!
						numIn--;
						numOut++;
					}
				}
				else
				{
					if (currentMergeRound == 1)
					{
#ifdef DEBUG
						printf("mergeIN:");
						for (j = i; j < numIn; j++)
						{
							printf(" %s", mopt->iFileNames[j]);
						}
						printf("\nmergeOut: %s\n", tmpOUT[numOut]);
#endif
						sprintf(tmpOUT[numOut], "%s.L%d.%d.las", mopt->oFile,
								currentMergeRound, numOut);
						if (mopt->SORT)
							sortAndMerge(tmpOUT[numOut], (mopt->iFileNames + i),
									numIn - i, mopt->VERBOSE);
						else
							merge(tmpOUT[numOut], (mopt->iFileNames + i),
									numIn - i, mopt->VERBOSE);
						numOut++;
					}
					else
					{
#ifdef DEBUG
						printf("mergeIN:");
						for (j = i; j < numIn; j++)
						{
							printf(" %s", tmpIN[j]);
						}
						printf("\nmergeOut: %s\n", tmpOUT[numOut]);
#endif
						sprintf(tmpOUT[numOut], "%s.L%d.%d.las", mopt->oFile,
								currentMergeRound, numOut);
						merge(tmpOUT[numOut], (tmpIN + i), numIn - i,
								mopt->VERBOSE);
						numOut++;
					}

				}
			}
#if DEBUG
			// check all written files for correctness
			for (j = 0; j < numOut; j++)
			{
				if (checkFile(mopt->db, tmpOUT[j], 1, mopt->CHECK_TRACE_POINTS, 0, mopt->SORT))
				{
					fprintf(stderr, "[ERROR] : LAmerge - intermediate overlap file %s failed check! Stop here!\n", tmpOUT[j]);
					exit(1);
				}
			}
#endif
			// remove intermediate files
			if (!mopt->KEEP && currentMergeRound > 1)
			{
				for (j = 0; j < numIn; j++)
				{
					if (unlink(tmpIN[j]))
						fprintf(stderr,
								"WARNING - Cannot remove intermediate overlap file: %s\n",
								tmpIN[j]);

				}
			}

			tmp = tmpIN;
			tmpIN = tmpOUT;
			tmpOUT = (char**) tmp;
			numIn = numOut;
			numOut = 0;
			currentMergeRound++;
		}
		// last merge step
#ifdef DEBUG
		printf("LAST mergeIN:");
		for (j = 0; j < numIn; j++)
		{
			printf(" %s", tmpIN[j]);
		}
		printf("\nLAST mergeOut: %s.las\n", mopt->oFile);
#endif
		sprintf(tmpOUT[0], "%s.las", mopt->oFile);
		merge(tmpOUT[0], tmpIN, numIn, mopt->VERBOSE);
		// remove intermediate files
		if (!mopt->KEEP && currentMergeRound > 1)
		{
			for (j = 0; j + 1 < numIn; j++)
			{
				if (unlink(tmpIN[j]))
					fprintf(stderr,
							"WARNING - Cannot remove intermediate overlap file: %s\n",
							tmpIN[j]);
			}
			// remove last file, only if its not an initial input file
			if (strcmp(tmpIN[j], mopt->iFileNames[mopt->numOfFilesToMerge - 1])
					!= 0)
			{
				if (unlink(tmpIN[j]))
					fprintf(stderr,
							"WARNING - Cannot remove intermediate overlap file: %s\n",
							tmpIN[j]);
			}
		}
		//cleanup
		if (tmpIN < tmpOUT)
		{
			for (i = 0; i < mopt->numOfFilesToMerge * 2; i++)
				free(tmpIN[i]);
			free(tmpIN);
		}
		else
		{
			for (i = 0; i < mopt->numOfFilesToMerge * 2; i++)
				free(tmpOUT[i]);
			free(tmpOUT);

		}
	}

	if (checkOverlapFile(mopt, fout, 0))
	{
		fprintf(stderr,
				"[ERROR] : LAmerge - overlap file %s failed check! Use -s option to sort the input files!\n",
				fout);
		exit(1);
	}
	free(fout);
}

static void copyFile(char *in, char *out)
{
	FILE *from, *to;
	char ch;

	char *fout;
	fout = (char*) malloc(strlen(out) + 20);
	sprintf(fout, "%s.las", out);

	/* open source file */
	if ((from = fopen(in, "rb")) == NULL)
	{
		fprintf(stderr, "Cannot open source file: %s.\n", in);
		exit(1);
	}

	/* open destination file */
	if ((to = fopen(fout, "wb")) == NULL)
	{
		fprintf(stderr, "Cannot open destination file: %s.\n", fout);
		exit(1);
	}

	/* copy the file */
	while (!feof(from))
	{
		ch = fgetc(from);
		if (ferror(from))
		{
			fprintf(stderr, "Error reading source file.\n");
			exit(1);
		}
		if (!feof(from))
			fputc(ch, to);
		if (ferror(to))
		{
			fprintf(stderr, "Error writing destination file.\n");
			exit(1);
		}
	}

	if (fclose(from) == EOF)
	{
		fprintf(stderr, "Error closing source file.\n");
		exit(1);
	}

	if (fclose(to) == EOF)
	{
		fprintf(stderr, "Error closing destination file.\n");
		exit(1);
	}

	free(fout);
}

int main(int argc, char* argv[])
{
	MERGE_OPT* mopt = parseMergeOptions(argc, argv);

	if (!mopt)
	{
		fprintf(stderr, "[ERROR] - LAmerge: Unable to parse arguments\n");
		exit(1);
	}

	if (mopt->numOfFilesToMerge == 0)
	{
		fprintf(stderr,
				"[WARNING] - LAmerge: No input files. Nothing to do!\n");
		return 0;
	}

	if (mopt->numOfFilesToMerge == 1)
	{
		copyFile(mopt->iFileNames[0], mopt->oFile);
		return 0;
	}

	doMergeAll(mopt);

	//cleanup
	clearMergeOptions(mopt);

	return 0;
}
