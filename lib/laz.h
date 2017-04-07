
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "dalign/align.h"

#define LAZ_MAGIC 0x254c415a

typedef struct
{
    uint32_t    magic;

    uint16_t    version;
    uint16_t    twidth;

    uint64_t    novl;

    uint64_t    reserved1;
    uint64_t    reserved2;
    uint64_t    reserved3;
    uint64_t    reserved4;
} LAZ_HEADER;

typedef struct
{
    uint64_t    a_from;
    uint64_t    a_to;

    uint64_t    novl;

    uint64_t    next;
    uint64_t    data;

    uint64_t    reserved1;
    uint64_t    reserved2;
    uint64_t    reserved3;
    uint64_t    reserved4;
} LAZ_INDEX;

typedef struct
{
    FILE* file;
    uint16_t version;
    uint16_t twidth;
    uint64_t novl;

    Overlap* ovl;
    uint32_t on;
    uint32_t omax;
    uint32_t ocur;

    void* buf;
    uint32_t bmax;

} LAZ;

LAZ* laz_open(char* fpath, int create);
int  laz_close(LAZ* laz);

Overlap* laz_read(LAZ* laz);
int laz_write(LAZ* laz, Overlap* ovl);

