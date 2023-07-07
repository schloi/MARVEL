#include "cuda/debug.cuh"
#include "forward-wave.cuh"
#include "reverse-wave.cuh"
#include "tracepoints.cuh"
#include <cuda_fp16.h>
#include <stdio.h>

namespace CudaTracePoints
{

__device__ static uint32_t rotateBitsRight(uint32_t val, uint32_t bits, uint32_t shift)
{
    const uint32_t mask = ( 1 << bits ) - 1;
    val &= mask;
    return (val >> shift) | ( val << (bits - shift) );
}

__device__ static uint32_t allocPrelimTracepointChunk(uint32_t* locks)
{
#ifdef DEBUG_ALLOC
    uint32_t attempts = 0;
    uint64_t start = clock64();
#endif
    uint32_t smid, chunk;
    if ( threadIdx.x == 0 )
    {
        asm( "mov.u32 %0, %%smid;" : "=r"( smid ) );
        uint32_t offset = ( blockIdx.x ^ ( blockIdx.x >> 4 ) ) % BLOCKS_PER_SM;

        while ( 1 )
        {
#ifdef DEBUG_ALLOC
            attempts++;
#endif
            uint32_t lockState0 = locks[ smid ];
            int firstSetBit     = __ffs( rotateBitsRight( lockState0, BLOCKS_PER_SM, offset ) );
            chunk = ( firstSetBit + offset - 1 ) % BLOCKS_PER_SM;
            uint32_t mask       = 1 << chunk;
            if ( chunk >= BLOCKS_PER_SM )
            {
#ifdef DEBUG_ALLOC
                printf( " Block %3d waiting for allocation on SM %3d: lockState0 %04x chunk %d\n", blockIdx.x, smid, lockState0 & 0xffff, chunk );
#endif
#if __CUDA_ARCH__ >= 700
                asm volatile("nanosleep.u32 100;");  // ease load on memory subsystem
#endif
                continue; // try again
            }
            uint32_t lockState1 = atomicAnd( &locks[ smid ], ~mask ); // clear "our" bit
            if ( ( lockState1 & mask ) == 0 )
            {
#ifdef DEBUG_ALLOC
                printf( " Block %3d unsuccessful allocation of chunk %2d mask %4x on SM %3d: lockState0 %04x lockState1 %04x attempt %u\n",
                        blockIdx.x, chunk, mask, smid, lockState0 & 0xffff, lockState1 & 0xffff, attempts);
#endif
                continue; // try again
            }
#ifdef DEBUG_ALLOC
            printf( " SM %03d chunk %02d successfully allocated for Block %03d mask %4x: lockState0 %04x lockState1 %04x attempt %u\n",
                     smid, chunk, blockIdx.x, mask, lockState0 & 0xffff, lockState1 & 0xffff, attempts);
#endif
            break; // allocation succesful
        }
#ifdef DEBUG_ALLOC
        uint64_t stop = clock64();
        printf( " SM %03d chunk %02d Block %03d allocation took %lu cycles (%u attempts)\n",
                smid, chunk, blockIdx.x, stop - start, attempts);
#endif
    }
    return __shfl_sync( FULL_WARP_MASK, smid * BLOCKS_PER_SM + chunk, 0);
}

__device__ static void freePrelimTracepointChunk(uint32_t* locks, int chunk)
{
    if ( threadIdx.x == 0 )
    {
        uint32_t smid = chunk / BLOCKS_PER_SM;
        chunk -= smid * BLOCKS_PER_SM;
        uint32_t mask = 1 << chunk;

        uint32_t lockState0 = atomicOr( &locks[ smid ], mask ); // set "our" bit
#ifdef ENABLE_SANITY_CHECK
        if ( lockState0 & mask )
        {
            printf( " double free of chunk %d on SM %d lockstate %4x\n", chunk, smid, lockState0 );
        }
#endif
#ifdef DEBUG_ALLOC
        printf( " SM %03d chunk %02d freed for Block %03d: lockState0 %04x\n",
                smid, chunk, blockIdx.x, lockState0 & 0xffff);
#endif
    }
}

__global__ void __launch_bounds__( BLOCK_SIZE, BLOCKS_PER_SM ) computeStreamedTracepointsBatch( const LocalAlignmentInput* __restrict__ inputs,
                                                                                                    const size_t* __restrict__ inputIndexRanges,
                                                                                                    uint64_t startIndex,
                                                                                                    uint32_t indicesToProcess,
                                                                                                    uint32_t tracepointsChunkSize,
                                                                                                    Tracepoints* __restrict__ paths,
                                                                                                    preliminary_Tracepoint* prelimTracepoints,
                                                                                                    uint32_t* prelimTracepointLocks,
                                                                                                    size_t tracePointVerticalChunkSize )
    {

    // Fixing the memory region for the blocks using: blockId * chunkSize
    // It is assuming that alignments is already fixed to the current running stream
    Tracepoints* blockPaths                  = paths + ( blockIdx.x * tracepointsChunkSize );
    Tracepoints* currentPath                 = blockPaths;
    const LocalAlignmentInput* previousInput = nullptr;
    tracepoint_int* tracepointsMemoryTop     = currentPath[ threadIdx.x == 1 ].tracepoints; // thread 0 stores currentPath[0] (aTracepointsMemoryTop),
                                                                                            // thread 1 stores currentPath[1] (bTracepointsMemoryTop),
    // all other threads we don't care (buth they do not read out of bounds)

    uint32_t chunk = allocPrelimTracepointChunk( prelimTracepointLocks );

    prelimTracepoints += chunk * tracePointVerticalChunkSize; // locate temporary storage allocated for our block

    for ( uint32_t i0 = blockIdx.x; i0 < indicesToProcess; i0 += gridDim.x )
    {

        uint32_t i         = i0 + startIndex;
        uint32_t aEndIndex = 0;
        for ( uint32_t inputIndex = inputIndexRanges[ i ]; inputIndex < inputIndexRanges[ i + 1 ]; inputIndex += 1 )
        {

#ifdef DEBUG
            if ( threadIdx.x == 0 )
            {
                printf( "Starting Input: %d\n", inputIndex );
                printf( "\taSequence:  [%d] (%d)\n", inputs[ inputIndex ].aSequence.sequenceLength, blockIdx.x );
                printf( "\tbSequence:  [%d] (%d)\n", inputs[ inputIndex ].bSequence.sequenceLength, blockIdx.x );
                printf( "APath: (%d)\n", blockIdx.x );
                printf( "BPath: (%d)\n", blockIdx.x );
                printf( "\tDiagonal: %d (%d)\n", inputs[ inputIndex ].diagonal, blockIdx.x );
                printf( "\tAntiDiagonal: %d (%d)\n", inputs[ inputIndex ].antiDiagonal, blockIdx.x );
            }

#endif

            if ( previousInput && previousInput->pair.unique == inputs[ inputIndex ].pair.unique &&
                 previousInput->diagonalBand == inputs[ inputIndex ].diagonalBand && aEndIndex >= inputs[ inputIndex ].aStartIndex )
            {
                if ( threadIdx.x == 0 )
                {
                    currentPath[ 0 ].skipped = true;
                    currentPath[ 1 ].skipped = true;
                }
                currentPath += 2;
#ifdef DEBUG
                if ( threadIdx.x == 0 )
                {
                    printf( "Skipped\n" );
                    printf( "End of Input: %d\n", inputIndex );
                    printf( "-----------------------------\n" );
                }
#endif
                continue;
            }

            previousInput = inputs + inputIndex;

            if ( threadIdx.x <= 1 )
            {
                currentPath[ threadIdx.x ].tracepoints = tracepointsMemoryTop;
            }
#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
#define LEVENSHTEIN_PARAM_VALUES                                                                                                                               \
    , inputs[ inputIndex ].aReverseMatrix, inputs[ inputIndex ].aForwardMatrix, inputs[ inputIndex ].bReverseMatrix, inputs[ inputIndex ].bForwardMatrix
#else
#define LEVENSHTEIN_PARAM_VALUES
#endif

            aEndIndex = CudaTracePoints::computeTracepoints( inputs[ inputIndex ].aSequence.deviceSequence,
                                                             inputs[ inputIndex ].aSequence.sequenceLength,
                                                             currentPath,
                                                             inputs[ inputIndex ].bSequence.deviceSequence,
                                                             inputs[ inputIndex ].bSequence.sequenceLength,
                                                             inputs[ inputIndex ].diagonal,
                                                             inputs[ inputIndex ].antiDiagonal,
                                                             FLOAT_TO_HALF( FLOAT_SUB( FLOAT_TO_HALF( 1.0F ), inputs[ inputIndex ].errorCorrelation ) ),
                                                             inputs[ inputIndex ].complement,
                                                             prelimTracepoints DEBUG_PARAMETERS_VALUES LEVENSHTEIN_PARAM_VALUES );

            safe_syncthreads();

#ifdef DEBUG
            if ( threadIdx.x == 0 )
            {

                printf( "APath A: %d -> %d [%d]\n", currentPath->aStartIndex, currentPath->aEndIndex, currentPath->aEndIndex - currentPath->aStartIndex );
                printf( "APath B: %d -> %d [%d]\n", currentPath->bStartIndex, currentPath->bEndIndex, currentPath->bEndIndex - currentPath->bStartIndex );
                printf( "APath Diffs: %d\n", currentPath->differences );
                printf( "APath Tracepoint Length: %d\n", currentPath->tracepointsLength );
                printf( "APath: " );
                uint32_t sum = 0;
                for ( uint32_t i = 0; i < currentPath->tracepointsLength; i += 2 )
                {
                    sum += currentPath[ 0 ].tracepoints[ i + 1 ];
                    printf( "(% 3d, %3d) ", currentPath->tracepoints[ i ], currentPath->tracepoints[ i + 1 ] );
                }
                printf( " (%d) \n\n", sum );

                printf( "BPath A: %d -> %d [%d]\n",
                        currentPath[ 1 ].aStartIndex,
                        currentPath[ 1 ].aEndIndex,
                        currentPath[ 1 ].aEndIndex - currentPath[ 1 ].aStartIndex );
                printf( "BPath B: %d -> %d [%d]\n",
                        currentPath[ 1 ].bStartIndex,
                        currentPath[ 1 ].bEndIndex,
                        currentPath[ 1 ].bEndIndex - currentPath[ 1 ].bStartIndex );
                printf( "BPath Diffs: %d\n", currentPath[ 1 ].differences );
                printf( "BPath Tracepoint Length: %d\n", currentPath[ 1 ].tracepointsLength );
                printf( "BPath:" );
                sum = 0;
                for ( uint32_t i = 0; i < currentPath[ 1 ].tracepointsLength; i += 2 )
                {
                    sum += currentPath[ 1 ].tracepoints[ i + 1 ];
                    printf( "(% 3d, %3d) ", currentPath[ 1 ].tracepoints[ i ], currentPath[ 1 ].tracepoints[ i + 1 ] );
                }
                printf( " (%d) \n", sum );
                printf( "Trace lengths: [%d x %d]\n", currentPath[ 0 ].tracepointsLength, currentPath[ 1 ].tracepointsLength );

                printf( "-----------------------------\n" );
            }
#endif

            if ( threadIdx.x <= 1 )
            {
                tracepointsMemoryTop = currentPath[ threadIdx.x ].tracepoints + currentPath[ threadIdx.x ].tracepointsLength;
            }
            currentPath += 2;
        }
    }

    freePrelimTracepointChunk( prelimTracepointLocks, chunk );
}

// B Path is always aPath + 1
__device__ uint32_t computeTracepoints( char* aSequence,
                                        uint32_t aLength,
                                        Tracepoints* paths,
                                        char* bSequence,
                                        uint32_t bLength,
                                        const int32_t diagonal,
                                        const int32_t antiDiagonal,
                                        marvl_float_t errorCorrelation,
                                        bool complement,
                                        preliminary_Tracepoint* prelimTracepoints DEBUG_PARAMETERS LEVENSHTEIN_PARAMS )
{

    uint32_t offset;
    uint32_t bFirstTracePoint;
    offset           = ( antiDiagonal - diagonal ) >> 1;
    bFirstTracePoint = complement ? LOCAL_ALIGNMENT_TRACE_SPACE - ( bLength - offset ) % LOCAL_ALIGNMENT_TRACE_SPACE : offset % LOCAL_ALIGNMENT_TRACE_SPACE;

    if ( bFirstTracePoint == LOCAL_ALIGNMENT_TRACE_SPACE )
    {
        bFirstTracePoint = 0;
    }
    // Initialization before reverse wave

    paths[ 0 ].tracepointsLength = 0;
    paths[ 1 ].tracepointsLength = 0;
    paths[ 0 ].skipped           = false;
    paths[ 1 ].skipped           = false;
    paths[ 0 ].differences       = 0;
    paths[ 1 ].differences       = 0;

    safe_syncthreads();

// Do reverse first to fill the initial trace points
#ifdef DEBUG
    if ( threadIdx.x == 0 )
    {
        printf( "\n-> Computing A <-\n" );
    }
#endif

#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
#define A_REVERSE_MATRIX , aReverseMatrix
#else
#define A_REVERSE_MATRIX
#endif

    ReverseWave::computeReverseWave( aSequence,
                                     offset + diagonal,
                                     paths,
                                     ( ( diagonal + offset ) % LOCAL_ALIGNMENT_TRACE_SPACE ),
                                     bSequence,
                                     offset,
                                     // ( offset % LOCAL_ALIGNMENT_TRACE_SPACE ),
                                     errorCorrelation,
                                     prelimTracepoints DEBUG_PARAMETERS_VALUES A_REVERSE_MATRIX );

#ifdef DEBUG
    if ( threadIdx.x == 0 )
    {
        printf( "\n-> Computing B <-\n" );
    }
#endif

#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
#define B_REVERSE_MATRIX , bReverseMatrix
#else
#define B_REVERSE_MATRIX
#endif

    ReverseWave::computeReverseWave( bSequence,
                                     offset,
                                     paths + 1,
                                     bFirstTracePoint,
                                     aSequence,
                                     offset + diagonal,
                                     errorCorrelation,
                                     prelimTracepoints DEBUG_PARAMETERS_VALUES B_REVERSE_MATRIX );

    safe_syncthreads();

    if ( threadIdx.x < WARP_SIZE )
    {
        //        const int lane_id = threadIdx.x;
        for ( int warp_id = 0; warp_id < 2; warp_id++ ) // used to be different warps, now just loop iterations
        {
            paths[ warp_id ].aStartIndex = paths[ warp_id ].aEndIndex;
            paths[ warp_id ].bStartIndex = paths[ warp_id ].bEndIndex;

            //            struct __builtin_align__( 2 * sizeof( tracepoint_int ) ) tracepoint_int_pair
            //            {
            //                tracepoint_int bEditDistance;
            //                tracepoint_int bTracePoint;
            //            };
            //            struct tracepoint_int_pair* tracepoint_pairs = reinterpret_cast<struct tracepoint_int_pair*>( paths[ warp_id ].tracepoints );
            //            int pair_size                                = paths[ warp_id ].tracepointsLength / 2;
            //
            //
            //            for ( int i = lane_id; i < pair_size / 2; i += WARP_SIZE )
            //            {
            //                struct tracepoint_int_pair tmp        = tracepoint_pairs[ i ];
            //                tracepoint_pairs[ i ]                 = tracepoint_pairs[ pair_size - 1 - i ];
            //                tracepoint_pairs[ pair_size - 1 - i ] = tmp;
            //            }
        }
    }

    safe_syncthreads();
#ifdef DEBUG
    if ( threadIdx.x == 0 )
    {
        printf( "\n-> Computing A <-\n" );
    }
#endif
#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
#define A_FORWARD_MATRIX , aForwardMatrix
#else
#define A_FORWARD_MATRIX
#endif

    // Forward should fill the last trace points in the array
    ForwardWave::computeForwardWave( aSequence + ( offset + diagonal ),
                                     aLength - offset - diagonal,
                                     paths,
                                     LOCAL_ALIGNMENT_TRACE_SPACE - ( ( diagonal + offset ) % LOCAL_ALIGNMENT_TRACE_SPACE ),
                                     bSequence + offset,
                                     bLength - offset,
                                     //         LOCAL_ALIGNMENT_TRACE_SPACE - ( offset % LOCAL_ALIGNMENT_TRACE_SPACE ),
                                     errorCorrelation,
                                     prelimTracepoints DEBUG_PARAMETERS_VALUES A_FORWARD_MATRIX );
#ifdef DEBUG
    if ( threadIdx.x == 0 )
    {
        printf( "\n-> Computing B <-\n" );
    }
#endif
#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
#define B_FORWARD_MATRIX , bForwardMatrix
#else
#define B_FORWARD_MATRIX
#endif

    ForwardWave::computeForwardWave( bSequence + offset,
                                     bLength - offset,
                                     paths + 1,
                                     LOCAL_ALIGNMENT_TRACE_SPACE - bFirstTracePoint,
                                     aSequence + ( offset + diagonal ),
                                     aLength - offset - diagonal,
                                     errorCorrelation,
                                     prelimTracepoints DEBUG_PARAMETERS_VALUES B_FORWARD_MATRIX );

    safe_syncthreads();

    if ( threadIdx.x == 0 )
    {

        paths[ 0 ].aEndIndex += diagonal + offset;
        paths[ 0 ].bEndIndex += offset;

        paths[ 1 ].aEndIndex += offset;
        paths[ 1 ].bEndIndex += diagonal + offset;

        paths[ 0 ].aStartIndex = diagonal + offset - paths[ 0 ].aStartIndex;
        paths[ 0 ].bStartIndex = offset - paths[ 0 ].bStartIndex;

        paths[ 1 ].aStartIndex = offset - paths[ 1 ].aStartIndex;
        paths[ 1 ].bStartIndex = diagonal + offset - paths[ 1 ].bStartIndex;
    }
    safe_syncthreads();
    return paths[ 0 ].aEndIndex;
}
} // namespace CudaTracePoints
