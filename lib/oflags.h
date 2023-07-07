
#pragma once

#define OVL_COMP         (1 << 0)      // B on reverse strand
#define OVL_DISCARD      (1 << 1)      // overlap tagged for removal
#define OVL_REPEAT       (1 << 3)      // repeat induced overlap
#define OVL_LOCAL        (1 << 4)      // local alignment
#define OVL_DIFF         (1 << 5)      // too many differences
#define OVL_STITCH       (1 << 7)      // stitched to another overlap
#define OVL_SYMDISCARD   (1 << 8)      //
#define OVL_OLEN         (1 << 9)      // overlap length
#define OVL_RLEN         (1 << 10)      // read length
#define OVL_TEMP         (1 << 11)     // temporary flag, not written to disk
#define OVL_CONT         (1 << 15)     // containment
#define OVL_GAP          (1 << 16)     // gap
#define OVL_TRIM         (1 << 17)     // trimmed
#define OVL_MODULE       (1 << 18)     // overlap spans unique repeat modules junction
#define OVL_OPTIONAL     (1 << 19)     // optional (risky) overlaps for touring
#define OVL_MULTI        (1 << 20)     // has more than one alignment

#define OVL_FLAGS       16             // number of flags


typedef struct _OverlapFlag2Label OverlapFlag2Label;

struct _OverlapFlag2Label
{
    int mask;

    char* label;
    char indicator;
};


void flags2str(char* pc, int flags);
