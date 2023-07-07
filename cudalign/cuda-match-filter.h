
#pragma once

#include "cuda/resource-manager.h"
#include <future>
extern "C"
{
#include "align.h"
#include "filter.h"
#include "ovlbuffer.h"
#include "radix.h"
#include "types.h"
#include "db/DB.h"
#include "lib/tracks.h"
}

void GPU_Match_Filter( char* aname,
                       HITS_DB* ablock,
                       char* bname,
                       HITS_DB* bblock,
                       KmerPos* atable,
                       int alen,
                       KmerPos** btable,
                       int blen,
                       Align_Spec* asettings,
                       uint32_t numberOfThreads,
                       int deviceId,
                       ResourceManager* resourceManager,
                       int numberOfStreams,
                       const std::string& sortPath,
                       const SequenceInfo* currentABlock,
                       const SequenceInfo* currentBBlock,
                       const SequenceInfo* currentBBlockComplement );
