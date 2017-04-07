
#include <zlib.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>

#include "utils.h"

#undef DEBUG_COMPRESSION

#define COMPRESS_MAX_CHUNK ( 8 * 1024 * 1024 )

void compress_chunks(void* ibuf, uint64_t ilen, void** _obuf, uint64_t* _olen)
{
    uLongf chunk = COMPRESS_MAX_CHUNK;

    uint64_t omax = ilen * 1.01 + 12 + sizeof(uint64_t) * ( ilen / chunk + 1 );
    void* obuf = malloc(omax);
    void* ocur = obuf;
    void* icur = ibuf;

#ifdef DEBUG_COMPRESSION
    printf("compress_chunks ilen %" PRIu64 " omax %" PRIu64 "\n", ilen, omax);
#endif

    while ( ilen != 0 )
    {
        if ( chunk > ilen )
        {
            chunk = ilen;
        }
        uLongf cchunk = omax - (ocur - obuf);

        compress(ocur + sizeof(uint64_t), &cchunk, icur, chunk);

#ifdef DEBUG_COMPRESSION
        printf("compressed chunk of %lu to %lu\n", chunk, cchunk);
#endif

        *((uint64_t*)ocur) = cchunk;
        ocur += cchunk + sizeof(uint64_t);

        icur += chunk;
        ilen -= chunk;
    }

    *_obuf = obuf;
    *_olen = ocur - obuf;

    // return ( ocur - obuf );
}

void uncompress_chunks(void* ibuf, uint64_t ilen, void* obuf, uint64_t olen)
{
    UNUSED(olen);

    void* icur = ibuf;
    void* ocur = obuf;

#ifdef DEBUG_COMPRESSION
    printf("uncompress_chunks ilen = %" PRIu64 "\n", ilen);
#endif

    while ( (uint64_t)(icur - ibuf) != ilen )
    {
        uLongf clen = *((uint64_t*)icur);

        // uLongf destlen = olen - (ocur - obuf);
        uLongf destlen = COMPRESS_MAX_CHUNK;

#ifdef DEBUG_COMPRESSION
        printf("uncompress %lu into %" PRIu64 " %ld", clen, olen, destlen);
#endif

        uncompress(ocur, &destlen, icur + sizeof(uint64_t), clen);

#ifdef DEBUG_COMPRESSION
        printf(" to %lu\n", destlen);
#endif

        icur += sizeof(uint64_t) + clen;
        ocur += destlen;
    }
}

#ifdef DEBUG_COMPRESSION
void test_chunks()
{
    /*   TESTING - BEGIN   */
    uint64_t bmax = 1024 * 1024 * 10;
    uint64_t* buf = malloc(sizeof(uint64_t) * bmax);
    uint64_t* cbuf = NULL;
    uint64_t clen = 0;

    bzero(buf, sizeof(uint64_t) * bmax);
    uint64_t i;
    for ( i = 0 ; i < bmax ; i++ )
    {
        buf[i] = i;
    }

    compress_chunks(buf, sizeof(uint64_t) * bmax, (void*)&cbuf, &clen);

    printf("clen = %" PRIu64 "\n", clen);
    printf("compressed to %d%%\n", (int)(100.0 * clen / (sizeof(uint64_t) * bmax)) );

    bzero(buf, sizeof(uint64_t) * bmax);
    uncompress_chunks(cbuf, clen, buf, sizeof(uint64_t) * bmax);

    for ( i = 0 ; i < bmax ; i++ )
    {
        if (buf[i] != i)
        {
            printf("buf[%" PRIu64 "] = %" PRIu64 "\n", i, buf[i]);
        }

    }
}
#endif
