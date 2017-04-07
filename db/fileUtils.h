#ifndef _FASTA_UTILS

#define _FASTA_UTILS

#include <stdio.h>

// checks if fasta header has pacbio format
int isPacBioHeader(char* header);

typedef struct
  { int    argc;
    char **argv;
    FILE  *input;
    int    count;
    char  *name;
  } File_Iterator;

File_Iterator *init_file_iterator(int argc, char **argv, FILE *input, int first);

int next_file(File_Iterator *it);

#define MAX_BUFFER       10001

typedef struct
  { FILE  *input;
    int    lineno;
    int    read;
    int    beg;
    int    end;
  } Read_Iterator;

Read_Iterator *init_read_iterator(FILE *input);

int next_read(Read_Iterator *it);

#endif //_FASTA_UTILS
