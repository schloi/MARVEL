
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <limits.h>
#include <sys/param.h>
#include <unistd.h>


#include "lasidx.h"
#include "pass.h"

#define SIZE_IO_BUFFER ( 1 * 1024 * 1024 )

// #define SEEK_PAST_TRACE

static char* lasidx_filename(const char* pathLas)
{
    char* pathIdx = strdup(pathLas);
    char* pathDot = rindex(pathIdx, '.');

    if (pathDot == NULL || strlen(pathIdx) - (pathDot - pathIdx) != 4)
    {
        fprintf(stderr, "malformed filename %s\n", pathLas);
        free(pathIdx);

        return NULL;
    }

    strcpy(pathDot + 1, "idx");

    return pathIdx;
}

static time_t file_mtime(const char* path)
{
    struct stat attr;
    stat(path, &attr);

    return attr.st_mtime;
}


lasidx* lasidx_create(HITS_DB* db, const char* pathLas)
{
    FILE* fileLas;
    FILE* fileIdx;
    void* buffer;

    if ( (fileLas = fopen(pathLas, "r")) == NULL )
    {
        fprintf(stderr, "failed to open %s\n", pathLas);

        return NULL;
    }

    char* pathIdx = lasidx_filename(pathLas);

    if ( (fileIdx = fopen(pathIdx, "w")) == NULL)
    {
        fprintf(stderr, "failed to open %s\n", pathIdx);
        free(pathIdx);

        return NULL;
    }

    buffer = malloc( SIZE_IO_BUFFER );
    setvbuf(fileLas, buffer, _IOFBF, SIZE_IO_BUFFER);

    printf("indexing %s\n", pathLas);

    PassContext* pctx = pass_init(fileLas, NULL);

    int tbytes = TBYTES( pctx->twidth );
    int nreads = DB_NREADS(db);

    lasidx* lasIndex = malloc(sizeof(lasidx) * ( nreads + 1) );

    if ( lasIndex == NULL )
    {
        fprintf(stderr, "Cannot allocate index buffer if size: %d\n!", nreads+1);
        exit(1);
    }

    bzero(lasIndex, sizeof(lasidx) * ( nreads + 1) );

    Overlap ovl;
    int a = -1;

    /*
    {
        off_t size = pctx->sizeOvlIn - (sizeof(ovl_header_novl) + sizeof(ovl_header_twidth));

        void* buffer = malloc(size);
        fread(buffer, size, 1, fileLas);

        printf("START\n");

        void* cur = buffer;
        while ( (cur - buffer) < size )
        {
            Overlap* ovl = cur - sizeof(void*);
            // printf("%d\n", ovl->aread);
            cur += sizeof(Overlap) - sizeof(void*) + tbytes * ovl->path.tlen;
        }

        printf("END\n");

        free(buffer);
    }

    return NULL;
    */

#ifndef SEEK_PAST_TRACE
    size_t tmax = 1000;
    ovl_trace* trace = malloc(sizeof(ovl_trace) * tmax);
#endif

    while (!Read_Overlap(fileLas, &ovl))
    {
        if (a != ovl.aread)
        {
            off_t cur = ftello(fileLas) - ( sizeof(Overlap) - sizeof(void*) );

            lasIndex[ ovl.aread ] = cur;
            a = ovl.aread;
        }

#ifdef SEEK_PAST_TRACE
        fseek(fileLas, tbytes * ovl.path.tlen, SEEK_CUR);
#else

        if ( ovl.path.tlen > tmax )
        {
            tmax = ovl.path.tlen * 1.2 + 1000;
            trace = realloc( trace, sizeof(ovl_trace) * tmax );
        }

        ovl.path.trace = trace;

        Read_Trace(fileLas, &ovl, tbytes);

#endif

    }

#ifdef SEEK_PAST_TRACE
    free(trace);
#endif

    if (a != ovl.aread)
    {
        off_t cur = ftello(fileLas) - ( sizeof(Overlap) - sizeof(void*) );

        lasIndex[ ovl.aread ] = cur;
    }

    long mtime = file_mtime(pathLas);

    fwrite(&mtime, sizeof(long), 1, fileIdx);

    uint64 left = 0;
    uint64 right = nreads;

    while ( lasIndex[left] == 0 )
    {
        left++;
    }

    while ( lasIndex[right] == 0 )
    {
        right--;
    }

    fwrite(&left, sizeof(uint64), 1, fileIdx);
    fwrite(&right, sizeof(uint64), 1, fileIdx);

    fwrite(lasIndex + left, sizeof(lasidx), right - left + 1, fileIdx);

    fclose(fileIdx);
    fclose(fileLas);

    pass_free(pctx);
    free(pathIdx);

    free(buffer);

    return lasIndex;
}

lasidx* lasidx_load(HITS_DB* db, const char* pathLas, int create)
{
    FILE* fileIdx;
    char* pathIdx = lasidx_filename(pathLas);

    if ( (fileIdx = fopen(pathIdx, "r")) == NULL)
    {
        if (create)
        {
            return lasidx_create(db, pathLas);
        }

        fprintf(stderr, "failed to open %s\n", pathIdx);
        free(pathIdx);

        return NULL;
    }

    long mtime = 0;

    if ( fread(&mtime, sizeof(long), 1, fileIdx) != 1 )
    {
        fprintf(stderr, "failed to read index header\n");
        return NULL;
    }

    if ( mtime != file_mtime(pathLas) )
    {
        printf("las file has been modified after index creation\n");

        if (create)
        {
            return lasidx_create(db, pathLas);
        }

        return NULL;
    }

    uint64 left, right;

    if( fread(&left, sizeof(uint64), 1, fileIdx) != 1 )
    {
        fprintf(stderr, "ERROR: failed to read left offset\n");
        exit(1);
    }

    if ( fread(&right, sizeof(uint64), 1, fileIdx) != 1 )
    {
        fprintf(stderr, "ERROR: failed to read right offset\n");
        exit(1);
    }

    int nreads = DB_NREADS(db);
    lasidx* lasIndex = malloc(sizeof(lasidx) * ( nreads + 1) );
    bzero(lasIndex, sizeof(lasidx) * (nreads + 1));

    if ( fread(lasIndex + left, sizeof(lasidx), right - left + 1, fileIdx) != (size_t)( right - left + 1 ) )
    {
        fprintf(stderr, "index file too short\n");
        return NULL;
    }

    off_t end = ftello(fileIdx);
    fseeko(fileIdx, 0, SEEK_END);

    if (end != ftello(fileIdx))
    {
        fprintf(stderr, "index file too long\n");
        return NULL;
    }

    fclose(fileIdx);

    free(pathIdx);

    return lasIndex;
}

void lasidx_close(lasidx* idx)
{
    free(idx);
}

