
#pragma once

#include "db/DB.h"

typedef struct _Read_Loader Read_Loader;

struct _Read_Loader
{
    HITS_DB* db;
    size_t max_mem;
    
    char* reads;        // storage for loaded reads
    uint64 maxreads;      // size of reads
    
    char** index;       // pointers into reads, indexed by read id
    
    int* rid;
    int currid;
    int nrid;
};

Read_Loader* rl_init(HITS_DB* db, size_t max_mem);

void rl_add(Read_Loader* rl, int rid);

void rl_load(Read_Loader* rl, int* reads, int nreads);


void rl_load_added(Read_Loader* rl);

void rl_load_read(Read_Loader* rl, int rid, char* read, int ascii);

void rl_free(Read_Loader* rl);

