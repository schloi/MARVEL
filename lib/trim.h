
#pragma once

#include "db/DB.h"
#include "dalign/align.h"
#include "lib/tracks.h"
#include "lib/pass.h"
#include "lib/read_loader.h"

typedef struct
{
    HITS_DB* db;
    HITS_TRACK* track;

    Read_Loader* rl;

    ovl_header_twidth twidth;       // trace point spacing

    Work_Data* align_work;          // working storage for the alignment module
    Alignment align;                // alignment and path record for computing the
    Path path;                      // alignment in the gap region

    uint64 nOvls;
    uint64 nOvlBases;
    uint64 nTrimmedOvls;
    uint64 nTrimmedBases;

} TRIM;

TRIM* trim_init(HITS_DB* db, ovl_header_twidth twidth, HITS_TRACK *track, Read_Loader *rl);
void trim_overlap(TRIM* trim, Overlap* ovl);
void trim_close(TRIM* trim);
