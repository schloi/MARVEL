
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "read_loader.h"

#define BLOCK_BUFFER (100*1024*1024)

#undef DEBUG_RL

Read_Loader* rl_init(HITS_DB* db, size_t max_mem)
{
    Read_Loader* rl = malloc(sizeof(Read_Loader));

    rl->db = db;
    rl->max_mem = max_mem;
    rl->index = (char**)malloc(sizeof(char*) * db->nreads);

    bzero(rl->index, sizeof(char*) * db->nreads);

    rl->reads = NULL;
    rl->maxreads = 0;

    rl->rid = NULL;
    rl->currid = 0;
    rl->nrid = 0;

    return rl;
}

void rl_add(Read_Loader* rl, int rid)
{
    if (rl->currid >= rl->nrid)
    {
        rl->nrid = rl->nrid * 1.2 + 1000;
        rl->rid = (int*)realloc(rl->rid, sizeof(int) * rl->nrid);
    }

    rl->rid[ rl->currid ] = rid;
    rl->currid++;
}

void rl_load_added(Read_Loader* rl)
{
    rl_load(rl, rl->rid, rl->currid);

    free(rl->rid);
    rl->currid = 0;
    rl->nrid = 0;
}

static int cmp_rids(const void* a, const void* b)
{
    return *((int*)a) - *((int*)b);
}

static int unique(int* a, int len)
{
    int i, j;
    j = 0;

    for (i = 1; i < len; i++)
    {
        if (a[i] != a[j])
        {
            j++;
            a[j] = a[i];
        }
    }

    return j+1;
}

void rl_load(Read_Loader* rl, int* rids, int nrids)
{
    HITS_DB* db = rl->db;

    FILE* bases = db->bases;
    uint64 nreads = db->nreads;
    HITS_READ* reads = db->reads;

    if (bases)
    {
        rewind(bases);
    }
    else
    {
        char* path = Catenate(db->path, "", "", ".bps");
        db->bases = bases = fopen(path, "r");

        if ( bases == NULL )
        {
            fprintf(stderr, "failed to open %s\n", path);
            exit(1);
        }
    }

    qsort(rids, nrids, sizeof(int), cmp_rids);
    nrids = unique(rids, nrids);

    int i;
    uint64 totallen = 0;
    for (i = 0; i < nrids; i++)
    {
        int rid = rids[i];
        totallen += reads[rid].rlen;
    }

#ifdef DEBUG_RL
    printf("%''llu bytes needed for %d reads\n", totallen, nrids);
#endif // DEBUG_RL

    if (totallen >= rl->maxreads)
    {
        rl->maxreads = totallen * 1.2 + 1000;
        rl->reads = (char*)realloc(rl->reads, rl->maxreads);

        if (  rl->reads == NULL )
        {
            fprintf(stderr, "failed to realloc\n");
            exit(1);
        }
    }

    size_t nbuf = BLOCK_BUFFER;
    char* buffer = malloc(nbuf);

    uint64 rb = 0;
    uint64 re = 0;
    size_t offb = 0;
    size_t offe = 0;
    int currid = 0;
    uint64 curreads = 0;

    while (re < nreads && currid < nrids)
    {
        while (offe - offb < nbuf && re < nreads)
        {
            offe = reads[re].boff;
            re++;
        }

        while (offe - offb > nbuf)
        {
            re--;
            offe = reads[re].boff;
        }

#ifdef DEBUG_RL
        printf("reading from read %'llu..%'llu byte %'zu..%'zu\n", rb, re, offb, offe);
#endif // DEBUG_RL

        fseeko(bases, offb, SEEK_SET);

        if (fread(buffer, offe - offb, 1, bases) != 1)
        {
            fprintf(stderr, "failed to read %ld\n", offe - offb);
            exit(1);
        }

        while ( currid < nrids && (uint64)rids[currid] < re )
        {
            int rid = rids[currid];

            int len = reads[rid].rlen;
            int clen = COMPRESSED_LEN(len);

#ifdef DEBUG_RL
            printf("  %'8d @ %'8lld %'8lld %lld %d\n",
                    rid, reads[rid].boff, reads[rid].boff - offb,
                    curreads, clen);
#endif // DEBUG_RL

            memcpy(rl->reads + curreads, buffer + (reads[rid].boff - offb), clen);
            rl->index[rid] = rl->reads + curreads;
            curreads += clen;

            currid++;
        }

        rb = re;
        offb = offe;
    }

    free(buffer);
}

void rl_load_read(Read_Loader* rl, int rid, char* read, int ascii)
{
    char* compressed = rl->index[rid];

    assert(compressed != NULL);

    HITS_READ* reads = rl->db->reads;
    int len = reads[rid].rlen;

    int clen = COMPRESSED_LEN(len);

    memcpy(read, compressed, clen);

    Uncompress_Read(len, read);

    if (ascii == 1)
    {
        Lower_Read(read);
        read[-1] = '\0';
    }
    else if (ascii == 2)
    {
        Upper_Read(read);
        read[-1] = '\0';
    }
    else
    {
        read[-1] = 4;
    }
}

void rl_free(Read_Loader* rl)
{
    free(rl->index);
    free(rl->reads);

    free(rl);
}

