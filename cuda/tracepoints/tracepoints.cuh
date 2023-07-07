#pragma once
#include "definitions.h"

namespace CudaTracePoints
{

#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
#define LEVENSHTEIN_PARAMS , matrix_int *aReverseMatrix, matrix_int *aForwardMatrix, matrix_int *bReverseMatrix, matrix_int *bForwardMatrix
#else
#define LEVENSHTEIN_PARAMS
#endif
/**
 * This function computes the tracepoints AxB and BxA
 *
 * @param aSequence A sequence
 * @param aLength A length
 * @param[out] paths Memory pointer for the tracepoints AxB (paths[0]) and BxA (paths[1])
 * @param bSequence B sequence
 * @param bLength B length
 * @param diagonal Diagonal of the seed
 * @param antiDiagonal Anti-diagonal of the seed
 * @param errorCorrelation Expected error correlation
 * @param complement true if B is a complement sequence, false otherwise
 * @param prelimTracepoint memory pointer to the per SM allocated memory to store information for the traceback
 *
 * Extra parameters defined in DEBUG_PARAMETERS and LEVENSHTEIN_PARAMS are only for debug purposes
 *
 * @return
 * The position in A where the alignment stopped
 */
__device__ uint32_t computeTracepoints( char* aSequence,
                                        uint32_t aLength,
                                        Tracepoints* paths,
                                        char* bSequence,
                                        uint32_t bLength,
                                        const int32_t diagonal,
                                        const int32_t antiDiagonal,
                                        marvl_float_t errorCorrelation,
                                        bool complement,
                                        preliminary_Tracepoint* prelimTracepoint DEBUG_PARAMETERS LEVENSHTEIN_PARAMS );
/**
 * Kernel to compute the alignment
 * @param inputs Inputs to be computed
 * @param inputIndexRanges Index ranges to be distributed among the blocks
 * @param startIndex the first index range to be considered
 * @param indicesToProcess the amount of index ranges to be processed
 * @param tracepointsChunkSize The amount of memory reserved for a block. The amount refers to `paths`
 * @param paths The pointer to the memory reserved for the kernel
 * @param prelimTracepoints The pointer to the memory reserved for the traceback information / per SM memory
 * @param prelimTracepointLocks The pointer to the locks of the traceback memory `prelimTracepoints`
 * @param tracePointVerticalChunkSize The amount of memory reserved for each block in the traceback memory `prelimTracepoints`
 */
__global__ void computeStreamedTracepointsBatch( const LocalAlignmentInput* __restrict__ inputs,
                                                 const size_t* __restrict__ inputIndexRanges,
                                                 uint64_t startIndex,
                                                 uint32_t indicesToProcess,
                                                 uint32_t tracepointsChunkSize,
                                                 Tracepoints* __restrict__ paths,
                                                 preliminary_Tracepoint* prelimTracepoints,
                                                 uint32_t* prelimTracepointLocks,
                                                 size_t tracePointVerticalChunkSize );

} // namespace CudaTracePoints
