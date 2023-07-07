
#pragma once

#include <stdint.h>
#include <stdlib.h>

typedef int64_t(*eldistfunc)(const void* a, const void* b);

size_t iseparator( void* pts, size_t npts, size_t elsz,
                   uint64_t maxdist,
                   uint64_t** intervals, size_t* maxintervals,
                   eldistfunc eldist );
