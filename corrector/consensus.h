
#pragma once

typedef struct
{
    unsigned char counts[5];     // A C G T -
} profile_entry;

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

    profile_entry* Aabs;
} v3_consensus_alignment_ctx;

typedef struct
{
    profile_entry* profile;
    int curprof;
    int maxprof;

    char* seq;
    int nseq;

    int alen;

    int added;

    v3_consensus_alignment_ctx* aln_ctx;
} consensus;

consensus* consensus_init();
void consensus_free(consensus* c);

void consensus_reset(consensus* c);

void consensus_add(consensus* c, char* seq, int sb, int se); // , int pb, int pe);

char* consensus_sequence(consensus* c, int dashes);

void consensus_print_profile(consensus* c, FILE* fileOut, int colorize);

int consensus_added(consensus* c);
