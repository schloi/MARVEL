
#pragma once

#include "lib/pass.h"

#include "dalign/align.h"

typedef struct
{
    unsigned long counts[5];     // A C G T -
} msa_profile_entry;

typedef struct
{
    int* vector;
    int  vecmax;

    int* trace;
    int  ntrace;

    int*  Stop;          //  Ongoing stack of alignment indels

    char* Babs;          //  Absolute base of A and B sequences

    int** PVF;  // waves for NP alignment
    int** PHF;

    msa_profile_entry* Aabs;
} msa_alignment_ctx;

typedef struct
{
    msa_profile_entry* profile;
    int curprof;
    int maxprof;

    char** msa_seq;     // points to sequences
    int msa_max;        // mem for msa_seq and msa_smax allocated
    int* msa_smax;      // mem for msa_seq[i] allocated
    int* msa_lgaps;     // number of leading gaps in seq[i]
    int* msa_len;       // length of seq[i]
    int* msa_ids;       // ids for the added sequences

    int* track;
    int tmax;

    char* seq;
    int nseq;

    int alen;

    int added;

    int twidth;

    int* ptp;        // storage for pass through points
    int ptpmax;

    msa_alignment_ctx* aln_ctx;
} msa;

msa* msa_init();
void msa_free(msa* m);
void msa_reset(msa* m);

void msa_add(msa* m, char* seq, int pb, int pe, int sb, int se, ovl_trace* trace, int tlen, int id);

char* msa_consensus(msa* m, int dashes);

void msa_print(msa* m, FILE* fileOut, int b, int e);
void msa_print_v(msa* m, FILE* fileOut);
void msa_print_profile(msa* m, FILE* fileOut, int b, int e, int colorize);
void msa_print_simple( msa* m, FILE* filemsa, FILE* filerids, int b, int e );
