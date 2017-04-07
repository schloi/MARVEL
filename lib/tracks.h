#pragma once

#include "db/DB.h"
#include <inttypes.h>

#define TRACK_ANNO          "anno"
#define TRACK_DUST          "dust"
#define TRACK_MASK_R        "maskr"
#define TRACK_MASK_C        "maskc"
#define TRACK_EDGES         "edges"
#define TRACK_FULL_EDGES    "fedges"
#define TRACK_Q             "q"
#define TRACK_REPEATS       "repeats"
#define TRACK_SOURCE        "source"
#define TRACK_TRIM          "trim"
#define TRACK_DDREPEATS     "ddrepeats"
#define TRACK_HREPEATS      "hrepeats"
#define TRACK_KREPEATS      "krepeats"
#define TRACK_KNOISE        "knoise"

#define TRACK_PACBIO_HEADER "pacbio"        // pacbio header (well, beg, end)
#define TRACK_PACBIO_RQ     "RQ"            // pacbio RQ value "read quality"
#define TRACK_PACBIO_CHEM   "chemistry"        // pacbio chemistry (BindingKit, SequencingKit, SoftwareVersion, [SequencingChemistry])
#define TRACK_SCAFFOLD      "scaffold"      // used in DAM, to keep track of putative N's (contigNumber, contigOffset)
#define TRACK_SEQID         "seqID"         // sequence identifier, keeps track of sequence in original fasta file
#define TRACK_SPR           "spr"           // slow polymerase region

#define TRACK_RPOINTS    "rpoints"
#define TRACK_JSOURCE    "jsource"

#define TRACK_VERSION_2   2

typedef int track_header_len;
typedef int track_header_size;
typedef uint64 track_header_offset;

typedef uint64 track_anno;
typedef int track_data;

typedef struct
{
  uint16_t version;
  uint16_t size;

  uint32_t pad1;

  uint64_t len;
  uint64_t clen;
  uint64_t cdlen;

  uint64_t reserved1;
  uint64_t reserved2;
  uint64_t reserved3;
  uint64_t reserved4;

} track_anno_header;

HITS_TRACK* track_load(HITS_DB *db, char* track);
void        track_close(HITS_TRACK* track);
int         track_delete(HITS_DB* db, const char* track);
void        track_write(HITS_DB* db, const char* track, int block, track_anno* anno, track_data* data, uint64_t dlen);


char* track_name(HITS_DB* db, const char* track, int block);

void write_track_trimmed(HITS_DB* db, const char* track, int block, track_anno* anno, track_data* data, uint64_t dlen);
void write_track_untrimmed(HITS_DB* db, const char* track, int block, track_anno* anno, track_data* data, uint64_t dlen);
