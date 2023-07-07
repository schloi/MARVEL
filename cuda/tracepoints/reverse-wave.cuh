#pragma once

#include "cuda/debug.cuh"
#include "definitions.h"
#include "matrix.cuh"

namespace CudaTracePoints
{
namespace ReverseWave
{

__device__ void computeReverseWave( const char* aSequence,
                                    const uint32_t aLength,
                                    Tracepoints* paths,
                                    const uint32_t aFirstTracePoint,
                                    const char* bSequence,
                                    const uint32_t bLength,
                                    marvl_float_t errorCorrelation,
                                    preliminary_Tracepoint* prelimTracepoints DEBUG_PARAMETERS LEVENSHTEIN_MATRIX_PARAMS )
{

#if defined( DEBUG ) || defined( _DEBUG )
    const uint32_t tid = threadIdx.x;

    if ( tid == 0 )
    {
        printf( "Reverse Wave:\n\tSequence A [%.5d -> %.2d]\n\tSequence B [%.5d]\n", (int)aLength, aFirstTracePoint, (int)bLength );
    }
#endif

    CudaTracePoints::Matrix::computeWave( aSequence,
                                          aLength,
                                          aFirstTracePoint,
                                          bSequence,
                                          bLength,
                                          paths,
                                          errorCorrelation,
                                          prelimTracepoints,
                                          REVERSE DEBUG_PARAMETERS_VALUES LEVENSHTEIN_MATRIX_PARAM_VALUES );
}

} // namespace ReverseWave

} // namespace CudaTracePoints
