#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda/utils.cuh"
#include "definitions.h"
#include "editDistanceTuple.cuh"
#include "packed-bytes-helper.cuh"
#include "warp-helper.cuh"

namespace CudaTracePoints
{
namespace Matrix
{

__device__ packed_bytes reverseSequenceReader( const char* sequence, const int32_t length, const int32_t index )
{
    if ( index < -4 || index >= length + 3 )
    {
        return { .w = 0xfdfdfdfd };
    }
    // Read an unaligned 32 bit value via two aligned 32 bit reads.
    // From a Stack Overflow answer by Tim Schmielau
    // https://stackoverflow.com/questions/40194012/reading-from-an-unaligned-uint8-t-recast-as-a-uint32-t-array-not-getting-all-v
    // WARNING! Reads past ptr!
    packed_bytes result;
    asm( "{\n\t"
         "   .reg .b64    aligned_ptr;\n\t"
         "   .reg .b32    low, high, alignment;\n\t"
         "   and.b64      aligned_ptr, %1, 0xfffffffffffffffc;\n\t" // %1 : ptr
         "   ld.u32       low, [aligned_ptr+-4];\n\t"
         "   ld.u32       high, [aligned_ptr];\n\t"
         "   cvt.u32.u64  alignment, %1;\n\t"
         "   prmt.b32.b4e %0, high, low, alignment;\n\t" // %0 : result
         "}"
         : "=r"( result.w )
         : "l"( sequence + length - 1 - index ) );
    for ( int i = 0; i < 4; i++ )
    {
        if ( (uint32_t)index + i >= length ) // check against either >= length or < 0
        {
            result.b[ i ] = ( index + i == -1 ) ? 0x40 : 0xfd;
        }
    }
#ifdef ENABLE_SANITY_CHECK
    packed_bytes reference;
    for ( int i = 0; i < 4; i++ )
    {
        if ( (uint32_t)index + i < length ) // check against either >= length or < 0
        {
            reference.b[ i ] = sequence[ length - 1 - index - i ];
        }
        else
        {
            reference.b[ i ] = ( index + i == -1 ) ? 0x40 : 0xfd;
        }
    }
    if ( reference.w != result.w )
    {
        printf( "reverseSequenceReader disagreement %08x != %08x ( index %d, length %d )\n", result.w, reference.w, index, length );
    }
#endif
    return result; // only works on low endian architectures
}

__device__ packed_bytes forwardSequenceReader( const char* sequence, const int32_t length, const int32_t index )
{
    if ( index < -4 || index >= length + 3 )
    {
        return { .w = 0xfefefefe };
    }
    // Read an unaligned 32 bit value via two aligned 32 bit reads.
    // From a Stack Overflow answer by Tim Schmielau
    // https://stackoverflow.com/questions/40194012/reading-from-an-unaligned-uint8-t-recast-as-a-uint32-t-array-not-getting-all-v
    // WARNING! Reads past ptr!
    packed_bytes result;
    asm( "{\n\t"
         "   .reg .b64    aligned_ptr;\n\t"
         "   .reg .b32    low, high, alignment;\n\t"
         "   and.b64      aligned_ptr, %1, 0xfffffffffffffffc;\n\t" // %1 : ptr
         "   ld.u32       low, [aligned_ptr];\n\t"
         "   ld.u32       high, [aligned_ptr+4];\n\t"
         "   cvt.u32.u64  alignment, %1;\n\t"
         "   prmt.b32.f4e %0, low, high, alignment;\n\t" // %0 : result
         "}"
         : "=r"( result.w )
         : "l"( sequence + index ) );
    for ( int i = 0; i < 4; i++ )
    {
        if ( (uint32_t)(index + i) >= (uint32_t) length ) // check against either >= length or < 0
        {
            result.b[ i ] = ( index + i == -1 ) ? 0x40 : 0xfe;
        }
    }
#ifdef ENABLE_SANITY_CHECK
    packed_bytes reference;
    for ( int i = 0; i < 4; i++ )
    {
        if ( (uint32_t)(index + i) < (uint32_t) length ) // check against either >= length or < 0
        {
            reference.b[ i ] = sequence[ index + i ];
        }
        else
        {
            reference.b[ i ] = ( index + i == -1 ) ? 0x40 : 0xfe;
        }
    }
    if ( reference.w != result.w )
    {
        printf( "forwardSequenceReader disagreement %08x != %08x ( index %d, length %d )\n", result.w, reference.w, index, length );
    }
#endif
    return result; // only works on low endian architectures
}

__device__
void readSequenceData( const char* aSequence,
                       const char* bSequence,
                       const int32_t aIndex00,
                       const int32_t bIndex00,
                       const int32_t aLength,
                       const int32_t bLength,
                       const WaveDirection& direction,
                       packed_bytes& aData,
                       packed_bytes& bData,
                       packed_bytes& nextAData,
                       packed_bytes& nextBData )
{
    {
        // load enough data for processing up to the next tracepoint (and a bit beyond)
        packed_bytes bReverseData, nextBReverseData;
#ifdef ENABLE_SANITY_CHECK
        bReverseData.w = 0x7d7d7d7d;
        aData.w = bData.w = 0x7e7e7e7e;
#endif
        switch ( direction )
        {
            case FORWARD:
                aData        = forwardSequenceReader( aSequence, aLength, aIndex00 - 1 );
                nextAData    = forwardSequenceReader( aSequence, aLength, aIndex00 - 1 + BASEPAIRS_PER_WARP / 2 + threadIdx.x ); // use all 4 bytes per thread
                bReverseData = forwardSequenceReader( bSequence, bLength, bIndex00 - BASEPAIRS_PER_THREAD / 2 );
                nextBReverseData =
                    forwardSequenceReader( bSequence, bLength, bIndex00 - BASEPAIRS_PER_THREAD / 2 + BASEPAIRS_PER_WARP / 2 + WARP_SIZE - 1 - threadIdx.x - 1 );
                break;
            case REVERSE:
            default:
                aData        = reverseSequenceReader( aSequence, aLength, aIndex00 - 1 );
                nextAData    = reverseSequenceReader( aSequence, aLength, aIndex00 - 1 + BASEPAIRS_PER_WARP / 2 + threadIdx.x ); // use all 4 bytes per thread
                bReverseData = reverseSequenceReader( bSequence, bLength, bIndex00 - BASEPAIRS_PER_THREAD / 2 );
                nextBReverseData =
                    reverseSequenceReader( bSequence, bLength, bIndex00 - BASEPAIRS_PER_THREAD / 2 + BASEPAIRS_PER_WARP / 2 + WARP_SIZE - 1 - threadIdx.x - 1 );
                break;
        }
        asm( "prmt.b32 %0, %1, 0, 0x4012;" : "=r"( bData.w ) : "r"( bReverseData.w ) );         // reverse lowest 3 bytes in bData
        asm( "prmt.b32 %0, %1, 0, 0x0123;" : "=r"( nextBData.w ) : "r"( nextBReverseData.w ) ); // reverse        4 bytes in nextBData
#ifdef ENABLE_SANITY_CHECK
        packed_bytes bDataRef;
        bDataRef.b[ 0 ] = bReverseData.b[ 2 ];
        bDataRef.b[ 1 ] = bReverseData.b[ 1 ];
        bDataRef.b[ 2 ] = bReverseData.b[ 0 ];
        bDataRef.b[ 3 ] = 0;
        if ( bDataRef.w != bData.w )
        {
            printf( "byte reversal disagreement %x != %x", bData.w, bDataRef.w );
        }
#endif
    }
}


__device__ EditDistanceTuple computeValue( const int32_t aIndex,
                                           const int32_t bIndex,
                                           EditDistanceTuple leftValue,
                                           EditDistanceTuple diagonalValue,
                                           EditDistanceTuple upperValue,
                                           const bool notSameChar,
                                           const bool preferDiagonalOrLeft,
                                           const bool preferLeft )
{
    // ToDo: Remove commented code ?
    //    EditDistanceTuple result = minEditDistance( leftValue, minEditDistance( diagonalValue + -!notSameChar, upperValue, aIndex < bIndex ), aIndex <= bIndex
    //    ) + 1;
    EditDistanceTuple result =
        minEditDistance( upperValue + 1, minEditDistance( diagonalValue + notSameChar, leftValue + 1, preferLeft ), preferDiagonalOrLeft );
#ifdef ENABLE_SANITY_CHECK
    {
    const bool myPreferDiagonalOrLeft = ( aIndex >= bIndex );
    const bool myPreferLeft           = ( aIndex > bIndex );
    if ( preferDiagonalOrLeft != myPreferDiagonalOrLeft || preferLeft != myPreferLeft )
    {
        printf( "[%2d]ERROR: preferDiagonal %d != myPreferDiagonal %d || preferUpper %d != myPreferUpper %d @ (%d, %d)\n",
                threadIdx.x,
                preferDiagonalOrLeft,
                myPreferDiagonalOrLeft,
                preferLeft,
                myPreferLeft,
                aIndex,
                bIndex );
    }
    }
#endif

#ifdef ENABLE_SANITY_CHECK
    if ( result.editDistance > MAX_EDIT_DISTANCE_OFFSET )
    {
        printf( "[%d]editDistance reserved range ( >= %u ) @ (%d, %d) : {%d, %d}\n"
                " left {%d, %d}, diagonal {%d, %d}, upper {%d, %d}, notsame %d\n",
                threadIdx.x,
                MAX_EDIT_DISTANCE_OFFSET,
                aIndex,
                bIndex,
                result.editDistance,
                result.prevIndex,
                leftValue.editDistance,
                leftValue.prevIndex,
                diagonalValue.editDistance,
                diagonalValue.prevIndex,
                upperValue.editDistance,
                upperValue.prevIndex,
                notSameChar );
    }
#endif
    return result;
}

__device__ PackedEditDistanceTuple computeValue( const int32_t aIndex0,
                                                 const int32_t bIndex0,
                                                 PackedEditDistanceTuple packedLeftValue,
                                                 PackedEditDistanceTuple packedDiagonalValue,
                                                 PackedEditDistanceTuple packedUpperValue,
                                                 const packed_mask packedNotSameChar,
                                                 const packed_mask packedPreferDiagonalOrLeft,
                                                 const packed_mask packedPreferLeft )
{
    // ToDo: Do we need a constant for 0x010101 ?
    PackedEditDistanceTuple result =
        minEditDistance( packedUpperValue + makePackedBytes( 0x010101 ),
                         minEditDistance( packedDiagonalValue + packedNotSameChar, packedLeftValue + makePackedBytes( 0x010101 ), packedPreferLeft ),
                                                      packedPreferDiagonalOrLeft );
#ifdef ENABLE_SANITY_CHECK
    {
        // ToDo: Do we need a constant for makePackedBytes( 0 ) ?
    PackedEditDistanceTuple check = { .editDistance = makePackedBytes( 0 ), .prevIndex = makePackedBytes( 0 ) };
    for ( int i = 0; i < BASEPAIRS_PER_THREAD / 2; i++ )
    {
        const int32_t aIndex             = aIndex0 + i;
        const int32_t bIndex             = bIndex0 - i;
        EditDistanceTuple leftValue      = { .editDistance = packedLeftValue.editDistance.b[ i ], .prevIndex = packedLeftValue.prevIndex.b[ i ] };
        EditDistanceTuple diagonalValue  = { .editDistance = packedDiagonalValue.editDistance.b[ i ], .prevIndex = packedDiagonalValue.prevIndex.b[ i ] };
        EditDistanceTuple upperValue     = { .editDistance = packedUpperValue.editDistance.b[ i ], .prevIndex = packedUpperValue.prevIndex.b[ i ] };
        const bool notSameChar           = packedNotSameChar.b[ i ];
        const bool preferDiagonalOrLeft  = packedPreferDiagonalOrLeft.b[ i ];
        const bool preferLeft            = packedPreferLeft.b[ i ];
        EditDistanceTuple oneCheck       = computeValue( aIndex, bIndex, leftValue, diagonalValue, upperValue, notSameChar, preferDiagonalOrLeft, preferLeft );
        check.editDistance.b[ i ]        = oneCheck.editDistance;
        check.prevIndex.b[ i ]           = oneCheck.prevIndex;
    }

        // ToDo: should we introduce a #define for '& 0xffffff' ?
        if ( ( result.editDistance.w & 0xffffff ) != ( check.editDistance.w & 0xffffff ) ||
             ( result.prevIndex.w & 0xffffff ) != ( check.prevIndex.w & 0xffffff ) )
    {
            printf(
                "ERROR in computeValue( @(%d,%d), left %08x'%08x, diag %08x'%08x, up %08x'%08x, noSame %08x, preferDorL %08x, preferL %08x ) == %08x'%08x != "
                "%08x'%08x\n",
                aIndex0,
                bIndex0,
                packedLeftValue.editDistance.w,
                packedLeftValue.prevIndex.w,
                packedDiagonalValue.editDistance.w,
                packedDiagonalValue.prevIndex.w,
                packedUpperValue.editDistance.w,
                packedUpperValue.prevIndex.w,
                packedNotSameChar.w,
                packedPreferDiagonalOrLeft.w,
                packedPreferLeft.w,
                result.editDistance.w,
                result.prevIndex.w,
                check.editDistance.w,
                check.prevIndex.w );
    }
    };
#endif
    return result;
}
/**
 * Block reduce and carray other parameters. The block reduce uses the operator <= on the subject parameter
 * ToDo: Rename to something that can lead to the comparison operator... e.g. leBlockReduceAndCarry
 * Todo: Do we need templates still
 * Todo: Can this function be inlined? Would that be useful?
 *
 * @tparam SUBJECT_TYPE Type of the subject of the comparison
 * @tparam PARAMETER_1_TYPE Type of the first carry parameter
 * @tparam PARAMETER_2_TYPE Type of the second carry parameter
 * @tparam PARAMETER_3_TYPE Type of the third carry parameter
 * @param[in,out] subject The subject of the comparison and the result of the block reduce
 * @param[in,out] parameter1 first carry parameter returned by the reduction
 * @param[in,out] parameter2 second carry parameter returned by the reduction
 * @param[in,out] parameter3 third carry parameter returned by the reduction
 * @return the wining lane
 */
template <typename SUBJECT_TYPE, typename PARAMETER_1_TYPE, typename PARAMETER_2_TYPE, typename PARAMETER_3_TYPE>
__device__ unsigned int blockReduceAndCarry( SUBJECT_TYPE& subject, PARAMETER_1_TYPE& parameter1, PARAMETER_2_TYPE& parameter2, PARAMETER_3_TYPE& parameter3 )
{
    // Warp Reduction, remembering source lane
    unsigned int lane = threadIdx.x;
    for ( int offset = 1; offset < WARP_SIZE; offset *= 2 )
    {
        // Now prefer values from higher lanes, which originate closer to the warp centre
        SUBJECT_TYPE otherObjective      = __shfl_down_sync( FULL_WARP_MASK, subject, offset );
        PARAMETER_1_TYPE otherParameter1 = __shfl_down_sync( FULL_WARP_MASK, parameter1, offset );
        unsigned int otherLane           = __shfl_down_sync( FULL_WARP_MASK, lane, offset );

        if ( otherObjective <= subject || ( otherObjective == subject && abs( (int)otherParameter1 ) <= abs( (int)parameter1 ) ) )
        {
            subject = otherObjective;
            lane      = otherLane;
        }
    }

    // broadcast winning lane to all threads
    lane = __shfl_sync( FULL_WARP_MASK, lane, 0 );
    // and get result from there
    subject    = __shfl_sync( FULL_WARP_MASK, subject, lane );
    parameter1 = __shfl_sync( FULL_WARP_MASK, parameter1, lane );
    parameter2 = __shfl_sync( FULL_WARP_MASK, parameter2, lane );
    parameter3 = __shfl_sync( FULL_WARP_MASK, parameter3, lane );

    return lane;
}

/**
 * Select and returns the minimum edit distance among the threads
 *
 * @param[in] editDistance The edit distance tuple values. Each thread will provide its own values
 * @param[in] diagonal0 the diagonal 0 for the current thread
 * @param[out] minDiagonal the diagonal corresponding to the smallest edit distance
 * @param[out] minEditDistance the smallest edit distance tuple among all thread
 */
__device__ void
findMinimumEditDistance( PackedEditDistanceTuple editDistance[ EVEN_ODD ], int32_t diagonal0, int32_t& minDiagonal, EditDistanceTuple& minEditDistance )
{

    // Find index and edit distance of value with minimum error rate:

    minEditDistance = editDistance[ 0 ][ 0] ;
    minDiagonal     = diagonal0;

    int32_t minAbsCurrDiagonal = abs( minDiagonal );

#ifdef DEBUG
    int selectedIndex = 0;
#endif
    for ( int valIndex = 1; valIndex < BASEPAIRS_PER_THREAD ; valIndex++ )
    {
        int32_t absCurrDiagonal = abs( diagonal0 + valIndex );
        if ( !( lessOrEqual( minEditDistance, editDistance[ valIndex % 2][ valIndex / 2], minAbsCurrDiagonal, absCurrDiagonal ) ) )
        {
            minEditDistance = editDistance[ valIndex % 2][ valIndex / 2];
            minDiagonal     = diagonal0 + valIndex;
#ifdef DEBUG
            selectedIndex = valIndex;
#endif
        }
    }

#ifdef DEBUG
#define INT_LANE_EQUAL int lane =
#else
#define INT_LANE_EQUAL
#endif
    INT_LANE_EQUAL blockReduceAndCarry<uint8_t, int32_t, uint8_t, int32_t>(
        minEditDistance.editDistance, minAbsCurrDiagonal, minEditDistance.prevIndex, minDiagonal );

#ifdef DEBUG
    int index = __shfl_sync( FULL_WARP_MASK, selectedIndex, lane );
    if ( threadIdx.x == 0 )
    {
        printf( " selected lane %d valIndex %d (index %d)\n", lane, index, lane * BASEPAIRS_PER_THREAD + index );
    }
#endif
}

/**
 * Checks if any thread has still the error rate below the error correlation. This functions does not compute the minimum error rate, but
 * allows all thread to vote if they are above or below the threshold. The error is computed using one or more trancepoints. This is define
 * by #define ERROR_RATE_WINDOW_SIZE (definitions.h::63)
 *
 * @param prelimTracepoint  The tracepoint verticals stored so far. This is used to compute the delta edit distances
 * @param errorCorrelation  The max expected error rate in a window
 * @param preliminaryTracepointIndex Current tracepoint verticals index
 * @param editDistanceBase Base edit distace to compute the absolute values
 * @param tracePointEditDistance The edit distances of the current thread
 * @param tracepointPosition Current tracepoint position in A
 * @param diagonal0 Diagonal 0 for the current thread
 * @return True if any thread has the error rate below the threshold
 */
__device__ bool checkErrorRateOk( preliminary_Tracepoint* prelimTracepoint,
                                  // ToDo: Consistently rename errorCorrelation to something else like errorRate, errorRateThreshold
                                  const marvl_float_t errorCorrelation,

                                  int32_t preliminaryTracepointIndex,

                                  uint32_t editDistanceBase,
                                  PackedEditDistanceTuple tracePointEditDistance[ EVEN_ODD ],
                                  int32_t tracepointPosition,
                                  int32_t diagonal0 DEBUG_PARAMETERS )
{

    // Find minimum edit distance within each thread:
    int minIndex;
    EditDistanceTuple minEditDistance = minWithIndex( tracePointEditDistance[0], tracePointEditDistance[1], minIndex );

    int32_t minBindex = tracepointPosition - diagonal0 - minIndex;

    bool isValid                 = ( minEditDistance.prevIndex != INVALID_TRACE);
    uint32_t currentEditDistance = minEditDistance.editDistance + editDistanceBase;
    diagonal_int prevIndex       = minEditDistance.prevIndex;

#ifdef DEBUG_ERROR_RATE
    if ( threadIdx.x == 16 )
        printf( " tid %d Bidx %d ED %d\n", threadIdx.x, minBindex, minEditDistance.editDistance );
#endif

    for ( int i = 1; i < ERROR_RATE_WINDOW_SIZE; i++ )
    {
        bool hasPrevious = ( preliminaryTracepointIndex >= i && prevIndex < BASEPAIRS_PER_WARP );
        if ( hasPrevious )
        {
#ifdef DEBUG_ERROR_RATE
            if ( threadIdx.x == 16 )
                printf( " tid %d Bidx %d ED %d = %d + %d prevIndex  %d  preliminaryTracepointIndex - i = %d\n",
                        threadIdx.x,
                        prelimTracepoint[ preliminaryTracepointIndex - i ].bIndex - prevIndex,
                        prelimTracepoint[ preliminaryTracepointIndex - i ].editDistanceBase +
                            prelimTracepoint[ preliminaryTracepointIndex - i ].editDistance[ prevIndex ].editDistance,
                        prelimTracepoint[ preliminaryTracepointIndex - i ].editDistanceBase,
                        prelimTracepoint[ preliminaryTracepointIndex - i ].editDistance[ prevIndex ].editDistance,
                        prevIndex,
                        preliminaryTracepointIndex - i );
#endif
            prevIndex = prelimTracepoint[ preliminaryTracepointIndex - i ].editDistance[ prevIndex ].prevIndex;
        }
    }

    bool hasPrevious = ( preliminaryTracepointIndex >= ERROR_RATE_WINDOW_SIZE && prevIndex < BASEPAIRS_PER_WARP );
    matrix_int previousEditDistance =
        hasPrevious ? prelimTracepoint[ preliminaryTracepointIndex - ( ERROR_RATE_WINDOW_SIZE ) ].editDistanceBase +
                          prelimTracepoint[ preliminaryTracepointIndex - ( ERROR_RATE_WINDOW_SIZE ) ].editDistance[ prevIndex ].editDistance
                    : 0;
    matrix_int previousIndex = hasPrevious ? prelimTracepoint[ preliminaryTracepointIndex - ( ERROR_RATE_WINDOW_SIZE ) ].bIndex - prevIndex : 0;
#ifdef DEBUG_ERROR_RATE
    if ( threadIdx.x == 16 )
        printf( " tid %d Bidx %d ED %d prevIndex  %d  hasPrevious %d\n", threadIdx.x, previousIndex, previousEditDistance, prevIndex, hasPrevious );
#endif

    marvl_error_t windowErrorRate = ( marvl_error_t )( currentEditDistance - previousEditDistance ) / ( marvl_error_t )( minBindex - previousIndex + 1 );

    bool thisThreadMayContinue = isValid && ( preliminaryTracepointIndex <= ERROR_RATE_WINDOW_SIZE || windowErrorRate <= errorCorrelation );

#ifdef DEBUG_ERROR_RATE
    //    if ( preliminaryTracepointIndex <= 4 )
    if ( isValid )
    {
        printf( "Thread:%d  MinEditDistace:%d MinBIndex:%d PreviousEditDistace:%d PreviousIndex:%d WindowErrorRate:%f @ %d with result %d || %d == %d\n",
                threadIdx.x,
                currentEditDistance,
                minBindex,
                previousEditDistance,
                previousIndex,
                windowErrorRate,
                preliminaryTracepointIndex,
                thisThreadMayContinue,
                windowErrorRate <= errorCorrelation,
                preliminaryTracepointIndex <= ERROR_RATE_WINDOW_SIZE || windowErrorRate <= errorCorrelation );
    }
#endif

    return __any_sync( FULL_WARP_MASK, thisThreadMayContinue );
}

/**
 * Warp reduce to find minimum
 * @param objective
 * @return the minimum value
 */
__device__ uint32_t warpMinimum( uint32_t objective )
{
    // Warp Reduction
    for ( int offset = 1; offset < WARP_SIZE; offset *= 2 )
    {
        // Now prefer values from higher lanes, which originate closer to the warp centre
        uint32_t otherObjective = __shfl_xor_sync( FULL_WARP_MASK, objective, offset );

        if ( otherObjective <= objective )
        {
            objective = otherObjective;
        }
    }

    return objective;
}

/**
 * Stores all computed edit distances at the current trancepoint position into prelimTrancepoit to compute the traceback.
 *
 * @param prelimTracepoint pointer to where the prelimirary tracepoints should be saved
 * @param tracepointPosition current tracepoint position
 * @param preliminaryTracepointIndex current index where the prelimirary tracepoints should be saved to
 * @param editDistanceBase0 edit distance base to be stored
 * @param tracePointEditDistance the edit distance tuples to be stored
 * @param diagonal0 the diagonal 0 for the current thread
 */
__device__ void storeTracePoint( preliminary_Tracepoint * prelimTracepoint,

                                 int32_t tracepointPosition,

                                 int32_t preliminaryTracepointIndex,
                                 uint32_t editDistanceBase0,
                                 PackedEditDistanceTuple tracePointEditDistance[ EVEN_ODD ],
                                 int32_t diagonal0 )
{
#ifdef DEBUG
    if ( threadIdx.x == 0 )
    {
        printf( "storing prelimTracepoint %d\n", preliminaryTracepointIndex );
    }
#endif

    uint32_t editDistanceBase = min( min3( tracePointEditDistance[0].editDistance ), min3( tracePointEditDistance[1].editDistance ) );
    editDistanceBase = warpMinimum( editDistanceBase );

#ifdef ENABLE_SANITY_CHECK
    if ( editDistanceBase != __shfl_sync( FULL_WARP_MASK, editDistanceBase, 0 ) )
    {
        printf( "ERROR: editDistanceBase disagreement.\n" );
    }
#endif
    if ( editDistanceBase > MAX_EDIT_DISTANCE_OFFSET )
    {
#ifdef DEBUG
        if ( threadIdx.x == 0 )
        {
            printf( "invalid editDistanceBase in storeTracePoint() %d (ignored)\n", editDistanceBase );
        }
#endif
        editDistanceBase = 0;
    }

    if ( threadIdx.x == 0 )
    {
        //        printf( "Preliminary Tracepoint Index: %d\n", preliminaryTracepointIndex );
#ifdef DEBUG
        printf( "editDistanceBase %d = %d + %d\n", editDistanceBase0 + editDistanceBase, editDistanceBase0, editDistanceBase );
        prelimTracepoint[ preliminaryTracepointIndex ].aIndex = tracepointPosition;
#endif
        prelimTracepoint[ preliminaryTracepointIndex ].editDistanceBase = editDistanceBase0 + editDistanceBase;
    }

    for ( int byteIndex = 0; byteIndex < BASEPAIRS_PER_THREAD / 2; byteIndex++ )
    {
        for ( int odd = 0; odd < EVEN_ODD; odd++ )
        {
            uint8_t editDistance = tracePointEditDistance[ odd ].editDistance.b[ byteIndex ];
            uint8_t prevIndex    = tracePointEditDistance[ odd ].prevIndex.b[ byteIndex ];
            if ( prevIndex != INVALID_TRACE )
            {
                editDistance -= editDistanceBase;
            }
            prelimTracepoint[ preliminaryTracepointIndex ].editDistance[ threadIdx.x * BASEPAIRS_PER_THREAD + byteIndex * EVEN_ODD + odd ] = {
                .editDistance = editDistance, .prevIndex = prevIndex };
            if ( prevIndex != INVALID_TRACE )
            {
                auto computedB = tracepointPosition - diagonal0 - byteIndex;

                // All thread write the same value to the same memory position. If it is guaranteed that the value is the same,
                // it is safe to proceed like this, which will result in a single write operation. The next sanity check verifies
                // if the assumption 'computedB + threadIdx.x * BASEPAIRS_PER_THREAD + byteIndex' is equal for all threads.
                prelimTracepoint[ preliminaryTracepointIndex ].bIndex = computedB + threadIdx.x * BASEPAIRS_PER_THREAD + byteIndex;
            }
        }
    }
#ifdef ENABLE_SANITY_CHECK
    {
    __syncwarp( FULL_WARP_MASK );
    for ( int byteIndex = 0; byteIndex < BASEPAIRS_PER_THREAD / 2; byteIndex++ )
    {
        for ( int odd = 0; odd < EVEN_ODD; odd++ )
        {
            uint8_t prevIndex = tracePointEditDistance[ odd ][ byteIndex ].prevIndex;
            if ( prevIndex != INVALID_TRACE )
            {
                auto computedB = tracepointPosition - diagonal0 - byteIndex;
                if ( prelimTracepoint[ preliminaryTracepointIndex ].bIndex != computedB + threadIdx.x * BASEPAIRS_PER_THREAD + byteIndex )
                {
                    printf( "ERROR: bIndex0 disagreement.\n" );
                }
            }
        }
    }
    }
#endif

#ifdef D_DEBUG
    for ( int tid = 0; tid < BLOCK_SIZE; tid++ )
    {
        for ( int valIndex = 0; valIndex < BASEPAIRS_PER_THREAD; valIndex++ )
        {
            if ( tid == threadIdx.x )
            {
                printf( "prelimTracepoint[%d] thread %d val %d idx %d: editDistance %d prev %d\n",
                        preliminaryTracepointIndex,
                        threadIdx.x,
                        valIndex,
                        threadIdx.x * BASEPAIRS_PER_THREAD + valIndex,
                        prelimTracepoint[ preliminaryTracepointIndex ].editDistanceBase +
                        prelimTracepoint[ preliminaryTracepointIndex ].editDistance[ threadIdx.x * BASEPAIRS_PER_THREAD + valIndex ].editDistance,
                        prelimTracepoint[ preliminaryTracepointIndex ].editDistance[ threadIdx.x * BASEPAIRS_PER_THREAD + valIndex ].prevIndex );
            }
        }
    }
#endif

    // Reseting variables for calculating next tracepoint
    for ( int i = 0; i < EVEN_ODD; i++ )
    {
        tracePointEditDistance[ i ] = { .editDistance = { .b = { INVALID_EDIT_DISTANCE, INVALID_EDIT_DISTANCE, INVALID_EDIT_DISTANCE, INVALID_EDIT_DISTANCE } },
            .prevIndex    = { .b = { INVALID_TRACE, INVALID_TRACE, INVALID_TRACE, INVALID_TRACE } } };
    }
    __syncwarp( FULL_WARP_MASK );
}

#ifdef DEBUG
__device__ void dumpPreliminaryTracepoints(
    int32_t preliminaryTracepointIndex, preliminary_Tracepoint* prelimTracepoints, EditDistanceTuple minEditDistance, int32_t minAindex, int32_t minBindex )
{
    printf( "!!!+++===+++!!!\n" );
    printf( "A:%d B:%d ED:%d I:%d PI:%d\n", minAindex, minBindex, minEditDistance.editDistance, 0, minEditDistance.prevIndex );
    while ( preliminaryTracepointIndex > 0 )
    {
        printf( "---\n" );

        preliminaryTracepointIndex--;
        for ( int i = 0; i < WARP_SIZE * BASEPAIRS_PER_THREAD; i++ )
        {

            auto a  = prelimTracepoints[ preliminaryTracepointIndex ].aIndex;
            auto b  = prelimTracepoints[ preliminaryTracepointIndex ].bIndex - i;
            auto ed = prelimTracepoints[ preliminaryTracepointIndex ].editDistanceBase +
                      prelimTracepoints[ preliminaryTracepointIndex ].editDistance[ i ].editDistance;
            auto pi = prelimTracepoints[ preliminaryTracepointIndex ].editDistance[ i ].prevIndex;

            printf( "A:%d B:%d ED:%d I:%d PI:%d\n", a, b, ed, i, pi );
        }
    }
    printf( "!!!+++===+++!!!\n" );
}
#endif

/**
 * Computes the trace back based on the stored prelim tracepoins and store the resulting trace points in the output memory in path
 *
 * @param path Output memory pointer
 * @param prelimTracepoints Stored preliminary tracepoints
 * @param minAindex A position of the smallest edit distance
 * @param minBindex B position of the smallest edit distance
 * @param minLastTracepointIndex The latest stored preliminary trancepoint index of the smallest edit distance
 * @param minEditDistance smallest edit distance
 * @param editDistanceBase base value of the edit distance to be able to compute the absolut values
 * @param firstTracePointPosition The first tracepoint posisiton
 * @param direction Direction of the computation

 */
__device__ void traceBackTrace( Tracepoints * path,
                                preliminary_Tracepoint * prelimTracepoints,
                                int32_t minAindex,
                                int32_t minBindex,
                                int32_t minLastTracepointIndex,
                                EditDistanceTuple minEditDistance,
                                uint32_t editDistanceBase,
                                int32_t firstTracePointPosition,
#ifdef DEBUG
                                int32_t preliminaryTracepointIndex,
#endif
                                WaveDirection direction DEBUG_PARAMETERS )

{
    if ( threadIdx.x == 0 )
    {

        int selectedIndex = minEditDistance.prevIndex;
#ifdef DEBUG
        printf( "preliminaryTracepointIndex: %d, minLastTracepointIndex: %d\n\tStarting at -> BIndex =  %d - %d = "
                "%d | Ed = (%d + %d) - %d = %d\n",
                preliminaryTracepointIndex,
                minLastTracepointIndex,
                minBindex,
                prelimTracepoints[ minLastTracepointIndex ].bIndex - selectedIndex,
                minBindex - ( prelimTracepoints[ minLastTracepointIndex ].bIndex - selectedIndex ),
                minEditDistance.editDistance,
                prelimTracepoints[ minLastTracepointIndex ].editDistanceBase,
                prelimTracepoints[ minLastTracepointIndex ].editDistance[ selectedIndex ].editDistance,
                minEditDistance.editDistance - ( prelimTracepoints[ minLastTracepointIndex ].editDistanceBase +
                                                 prelimTracepoints[ minLastTracepointIndex ].editDistance[ selectedIndex ].editDistance ) );

        int32_t aIndex = minAindex;

        printf( "Selected index: %d editDistance %d\n", minEditDistance.prevIndex, minEditDistance.editDistance + editDistanceBase );
#endif

        int32_t tracepointCounter = minLastTracepointIndex + 1;

#ifdef DEBUG_TRACEBACK_AND_MATRIX
        dumpPreliminaryTracepoints( tracepointCounter, prelimTracepoints, minEditDistance, minAindex, minBindex );
#endif

        path->tracepointsLength +=
            2 * ( tracepointCounter -
                  ( direction == WaveDirection::FORWARD &&
                    firstTracePointPosition !=
                        LOCAL_ALIGNMENT_TRACE_SPACE ) ); // firstTracePointPosition != LOCAL_ALIGNMENT_TRACE_SPACE  if true tracepoints must be added

        uint32_t tracepointIndex = ( direction == FORWARD ) ? path->tracepointsLength / 2 - 1 : 0;

        path->aEndIndex = minAindex;
        path->bEndIndex = minBindex;

        path->differences += minEditDistance.editDistance + editDistanceBase;

        matrix_int selectedEditDistance = minEditDistance.editDistance + editDistanceBase;

        for ( int index = tracepointCounter - 1; index >= 0; index--, tracepointIndex -= direction )
        {

#ifdef ENABLE_SANITY_CHECK
            if ( index != 0 && selectedIndex == TRACE_STARTS_HERE )
            {
                printf( "WARNING: Selected index is marked as `TRACE_STARTS_HERE` and current index (%d) is not 0. Input index %d. ", index, inputIndex );
                }
#endif

            matrix_int previousEditDistance =
                index == 0 ? 0 : prelimTracepoints[ index - 1 ].editDistanceBase + prelimTracepoints[ index - 1 ].editDistance[ selectedIndex ].editDistance;
            matrix_int previousBindex = index == 0 ? 0 : prelimTracepoints[ index - 1 ].bIndex - selectedIndex;

#ifdef ENABLE_SANITY_CHECK
            {
                if ( selectedEditDistance - previousEditDistance > UINT8_MAX )
                {
                    printf( "editDistance delta %d = %d - %d does not fit into uint8_t for inputIndex %d\n",
                            selectedEditDistance - previousEditDistance,
                            selectedEditDistance,
                            previousEditDistance,
                            inputIndex );
                }
            }
#endif

#ifdef DEBUG
            {

                if ( index == 0 && selectedIndex != TRACE_STARTS_HERE )
                {
                    printf( "*** Input index %d is a test case for `if ( index == 0 && selectedIndex != TRACE_STARTS_HERE )`\n", inputIndex );
                    //                break;
                }

                int32_t previousAIndex = index == 0 ? 0 : prelimTracepoints[ index - 1 ].aIndex;

                printf( "[%d] Traceppoint [%03d,%03d] Backtrace in direction %d ind %03d @ (%d, %d): %03d-%03d = %03d / %05d-%05d = %03d / %f%s\n",
                        threadIdx.x,
                        index,
                        tracepointIndex,
                        direction,
                        selectedIndex,
                        aIndex,
                        minBindex,
                        minBindex,
                        previousBindex,
                        minBindex - previousBindex,
                        selectedEditDistance,
                        previousEditDistance,
                        selectedEditDistance - previousEditDistance,
                        (float)( selectedEditDistance - previousEditDistance ) / (float)( minBindex - previousBindex ),
                        selectedEditDistance - previousEditDistance < abs( (int)( ( aIndex - previousAIndex ) - ( minBindex - previousBindex ) ) )
                            ? "!!! WARNING !!!"
                            : "" );

                aIndex = previousAIndex;
            }
#endif

            if ( direction == FORWARD && index == 0 && firstTracePointPosition != LOCAL_ALIGNMENT_TRACE_SPACE )
            {
                path->tracepoints[ tracepointIndex * 2 ] += selectedEditDistance - previousEditDistance;
                path->tracepoints[ tracepointIndex * 2 + 1 ] += minBindex - previousBindex;
            }
            else
            {
                path->tracepoints[ tracepointIndex * 2 ]     = selectedEditDistance - previousEditDistance;
                path->tracepoints[ tracepointIndex * 2 + 1 ] = minBindex - previousBindex;
            }
            minBindex            = previousBindex;
            selectedEditDistance = previousEditDistance;

            if ( index > 0 )
            {
                selectedIndex = prelimTracepoints[ index - 1 ].editDistance[ selectedIndex ].prevIndex;
            }
        }
    }
}
/**
 *
 * This function computes how many threads and to which direction the diagonal band of the matrix should be shifted to. Shift ensures that the 'valley' of edit
 * distance is always centered in the diagonal band. This is done by comparing the edit distance of the current thread to the opposite end of the warp. The
 * amount of thread which has bigger values then the opposite thread, is the amount that needs to be shifted.
 *
 * If the amount points to both direction, the function returns 0. Otherwise it returns the half of thread count which has bigger edit distance then the
 * opposite end thread.
 *
 * \note
 * This function can only be called on a even anti diagonal. Even index are hard-coded
 *
 *
 * @param currentDiagonalShift The accumulated shift amount so far
 * @param diagonal0 The diagonal 0 of the current thread
 * @param resultValueDiagonal the current computed matrix cell for the thread
 * @return The amount to be shifted
 *         * Negative amounts means the diagonal band will move down along the anti diagonal
 *         * Positive amounts means the diagonal band will move up along the anti diagonal
 */
__device__ int32_t shiftAmount( int32_t& currentDiagonalShift, int32_t& diagonal0, PackedEditDistanceTuple resultValueDiagonal[ EVEN_ODD ] DEBUG_PARAMETERS )
{
    //
    // Ensure "interesting" region is centered within the antidiagonal:
    //

    const int otherEndThreadIndex = ( threadIdx.x < WARP_SIZE / 2 ) ? WARP_SIZE - 1 : 0;
    // compare against value at the end of the "even" antidiagonal:
    const matrix_int endEditDistance =
        ( threadIdx.x < WARP_SIZE / 2 ) ? resultValueDiagonal[ 0 ][ 0 ].editDistance : resultValueDiagonal[ 0 ][ BASEPAIRS_PER_THREAD / 2 - 1 ].editDistance;
    const matrix_int otherEndEditDistance = __shfl_sync( FULL_WARP_MASK, endEditDistance, otherEndThreadIndex );

    // ToDo: the non-SIMD code had a bug here so slight differences in results are possible. Fix non-SIMD code and double-check.
    bool shouldShift =
        otherEndEditDistance <= MAX_EDIT_DISTANCE_OFFSET && resultValueDiagonal[ 0 ].editDistance.b[ BASEPAIRS_PER_THREAD / 4 ] > otherEndEditDistance;

    uint32_t shouldShiftPattern = __ballot_sync( FULL_WARP_MASK, shouldShift );

    int canShiftLeftBy  = WARP_SIZE - __clz( shouldShiftPattern & ( FULL_WARP_MASK >> WARP_SIZE / 2 ) );           // __cls  : count leading zeros
    int canShiftRightBy = WARP_SIZE - __clz( __brev( shouldShiftPattern ) & ( FULL_WARP_MASK >> WARP_SIZE / 2 ) ); // __brev : bit reversal

////    if (threadIdx.x == 0 )
//        printf("[%2d] canShiftLeftBy %2d, canShiftRightBy %2d, pattern %08x, endDist %3d, otherEndDist %3d\n",
//            threadIdx.x, canShiftLeftBy, canShiftRightBy, shouldShiftPattern,
//            endEditDistance, otherEndEditDistance );

    if ( canShiftLeftBy && canShiftRightBy )
    {
        // If it's unclear which way to shift, don't shift at all (because we lose some information of each shift)
        return 0;
    }

    // centre the interesting region, i.e. shift by half the possible amount:
    int32_t nextShift = ( canShiftLeftBy - canShiftRightBy ) / 2;

#ifdef ENABLE_SANITY_CHECK
    if ( abs( nextShift ) > 100 / BASEPAIRS_PER_THREAD )
    {
        printf( "Huge shift from %d to %d in input %d!\n", currentDiagonalShift, nextShift * BASEPAIRS_PER_THREAD, inputIndex );
    }
#endif

    return nextShift;
}

__device__ void shiftDiagonals( int32_t nextShift,
                                int32_t currentDiagonalShift,
                                int32_t diagonal0,
#ifdef DEBUG
                                int32_t antiDiagonal0,
#endif
                                PackedEditDistanceTuple * resultValueDiagonal,
                                PackedEditDistanceTuple * tracePointEditDistance,
                                packed_mask& tracepointMask DEBUG_PARAMETERS )
{
    if ( unlikely( nextShift != 0 ) )
    {
#ifdef ENABLE_SANITY_CHECK
        if ( tracepointMask.w != 0 )
        {
            printf ( "[%2d] nonzero tracepointMask %08x\n", threadIdx.x, tracepointMask.w );
        }
#endif
#ifdef DEBUG
        if ( threadIdx.x == WARP_SIZE / 2 )
        {
            printf( "Diagonal Shift from %d to %d at antidiagonal %d near %d\n",
                    currentDiagonalShift,
                    currentDiagonalShift + nextShift * BASEPAIRS_PER_THREAD,
                    antiDiagonal0,
                    ( antiDiagonal0 + diagonal0 ) / 2 );
        }
#endif

        if ( nextShift > 0 )
        {
            for ( int odd = 0; odd < EVEN_ODD; odd++ )
            {
                resultValueDiagonal[ odd ] = warpShuffleDown( resultValueDiagonal[ odd ], nextShift );
            }
            if ( threadIdx.x == WARP_SIZE - nextShift )
            {
                // ToDo: Comment or #define
                tracepointMask.w = 0xff0000;
            }
        }
        else if ( nextShift < 0 )
        {
            for ( int odd = 0; odd < EVEN_ODD; odd++ )
            {
                resultValueDiagonal[ odd ] = warpShuffleUp( resultValueDiagonal[  odd ], -nextShift );
            }
            if ( threadIdx.x == -nextShift - 1 )
            {
                // ToDo: Comment or #define
                tracepointMask.w = 0xff00;
            }
        }
    }
}

__device__ uint32_t editDistanceBaseUpdate( PackedEditDistanceTuple * resultValueDiagonal )
{
    // find resultValueDiagonal[] minimum
    uint32_t update = min( min3( resultValueDiagonal[0].editDistance ), min3( resultValueDiagonal[1].editDistance ) );

    update = warpMinimum( update );
    if ( update > MAX_EDIT_DISTANCE_OFFSET )
    {
        update = 0;
#ifdef DEBUG
        if ( threadIdx.x == 0 )
        {
            printf( "ERROR: invalid editDistanceBaseUpdate %d (ignored)\n", update );
        }
#endif
    }

    // and subtract from all threads
    for ( int odd = 0; odd < EVEN_ODD; odd++ )
    {
        for ( int i = 0; i < BASEPAIRS_PER_THREAD / 2; i++ )
        if ( resultValueDiagonal[ odd ].prevIndex.b[i] != INVALID_TRACE )
            resultValueDiagonal[ odd ].editDistance.b[i] -= update;
    }
    return update;
}

__device__ void computeThreadValues( packed_mask inMatrixMask,
                                     packed_mask tracepointMask,      // 0xff if on a tracepoint
                                     packed_mask afterTracepointMask, // 0xff on th ecolumn just after a tracepoint
                                     const int32_t tracepointPosition,
                                     const int32_t aIndex0,
                                     const int32_t bIndex0,
                                     PackedEditDistanceTuple leftValue,
                                     PackedEditDistanceTuple diagonalValue,
                                     PackedEditDistanceTuple upperValue,
                                     PackedEditDistanceTuple& resultValueDiagonal,
                                     packed_bytes notSameChar,
                                     packed_bytes preferDiagonalOrLeft,
                                     packed_bytes preferLeft,
                                     int odd,
                                     PackedEditDistanceTuple& tracePointEditDistance,
                                     int32_t preliminaryTracepointIndex DEBUG_PARAMETERS LEVENSHTEIN_MATRIX_PARAMS )
{
    leftValue.prevIndex     = selectBytes( generateByteIndex( odd-1 ), leftValue.prevIndex, afterTracepointMask );
    diagonalValue.prevIndex = selectBytes( generateByteIndex( odd ), diagonalValue.prevIndex, afterTracepointMask );
    resultValueDiagonal     = selectED( computeValue( aIndex0, bIndex0, leftValue, diagonalValue, upperValue, notSameChar, preferDiagonalOrLeft, preferLeft ),
                                    resultValueDiagonal,
                                    inMatrixMask );

#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
    // ToDo: adapt for packed SIMD
//        if ( levenshteinMatrix[ aIndex * ( bLength + 1 ) + bIndex ] != resultValueDiagonal.editDistance )
//        {
//            if ( aIndex < 20 && bIndex < 20 )
//                printf( "[%d] Thread %d valIndex %d has different edit distances compared to the levenshtein matrix. (%d, %d) = %d  != %d\n",
//                        threadIdx.x,
//                        threadIdx.x,
//                        valIndex,
//                        aIndex,
//                        bIndex,
//                        levenshteinMatrix[ aIndex * ( bLength + 1 ) + bIndex ],
//                        resultValueDiagonal.editDistance );
//        }
#endif

    // store if at tracepoint:
    tracePointEditDistance = { .editDistance = selectBytes( resultValueDiagonal.editDistance, tracePointEditDistance.editDistance, tracepointMask ),
                               .prevIndex    = selectBytes( resultValueDiagonal.prevIndex, tracePointEditDistance.prevIndex, tracepointMask ) };

#ifdef TRACEPOINT_DEBUG_START
    for ( int byteIndex = 0; byteIndex < BASEPAIRS_PER_THREAD / 2 ; byteIndex++ )
    {
        if ( tracepointMask.b[ byteIndex ] && preliminaryTracepointIndex >= TRACEPOINT_DEBUG_START && preliminaryTracepointIndex <= TRACEPOINT_DEBUG_END )
        {
            printf( "saving prelimTracepoint @ (%3d,%3d) thread %d val %d idx %d: editDistance %d prev %d mask %08x\n",
                    aIndex0 + byteIndex,
                    bIndex0 - byteIndex,
                    threadIdx.x,
                    byteIndex * 2 + odd,
                    threadIdx.x * BASEPAIRS_PER_THREAD + byteIndex * 2 + odd,
                    tracePointEditDistance[ byteIndex ].editDistance,
                    tracePointEditDistance[ byteIndex ].prevIndex,
                    tracepointMask.w );
        }
    }
#endif

#ifdef DEBUG_TRACEBACK_AND_MATRIX
    for ( int i=0; i<BASEPAIRS_PER_THREAD/2; i++)
    {
        if ( // inMatrixMask.b[i] &&
             (uint32_t)aIndex0 + i <= 400 && (uint32_t)bIndex0 - i <= 400 )
            printf( "# T%d V%d A%d B%d E%d P%d N%d S0\n",
                    threadIdx.x,
                    i*2+odd,
                    aIndex0+i,
                    bIndex0-i,
                    resultValueDiagonal[i].editDistance,
                    resultValueDiagonal[i].prevIndex,
                    preliminaryTracepointIndex ); // +1?
    }
#endif
}

__device__ __noinline__ void computeWave( const char* aSequence,
                                          const int32_t aLength,
                                          const int32_t aFirstTracePointPosition,
                                          const char* bSequence,
                                          const int32_t bLength,
                                          Tracepoints* paths,
                                          marvl_float_t errorCorrelation,
                                          preliminary_Tracepoint* prelimTracepoints,
                                          WaveDirection direction DEBUG_PARAMETERS LEVENSHTEIN_MATRIX_PARAMS )
{

    PackedEditDistanceTuple tracePointEditDistance[ EVEN_ODD ];

    // Diagonals which this thread is responsible for
    int32_t diagonal0 = ( threadIdx.x - BLOCK_SIZE / 2 ) * BASEPAIRS_PER_THREAD;
    //    int32_t diagonal1 = diagonal0 + 1;

    // Index of a and b and also index of the matrix
    // It must be signed because some threads will still be outside the matrix at the beginning
    //    int32_t aIndex; // Formally j
    //    int32_t bIndex; // Formally i

    // Anti diagonals: First, and last anti diagonal of each for loop
    int32_t antiDiagonal0 = 0;

    // Baseline edit distance. All other editDistance values are offsets against this base, in order to keep them within the range of
    // an uint8_t. Will be updated every LOCAL_ALIGNMENT_TRACE_SPACE staps, just after calling storeTracePoint() and shifting.
    int32_t editDistanceBase = 0;

    // Edit distance since original seed, and index into array written at latest tracepoint, allowing to trace back the best alignment
    // These values must be shuffled during diagonal shift
    PackedEditDistanceTuple resultValueDiagonal[ EVEN_ODD ];

    // Basepair data for A and B (3 basepairs per thread)
    packed_bytes aData, bData;
    // Basepair data for A and B beyond aData/bData
    // (4 basepairs per thread, so we can store more than 100 basepairs to get us past the next tracepoint without reloading)
    packed_bytes nextAData, nextBData;

    // Masks determining values above or on the main diagonal, for even antidiagonals
    packed_bytes aboveOrOnDiagonal, aboveDiagonal0;

    const packed_bytes initialOffdiagEditDistance = { .b = { INITIAL_INVALID_EDIT_DISTANCE, INITIAL_INVALID_EDIT_DISTANCE, INITIAL_INVALID_EDIT_DISTANCE, INITIAL_INVALID_EDIT_DISTANCE } };
    const packed_bytes initialDiagEditDistance = { .b = { 0, INITIAL_INVALID_EDIT_DISTANCE, INITIAL_INVALID_EDIT_DISTANCE, INITIAL_INVALID_EDIT_DISTANCE } };
    const packed_bytes initialDiagPrevIndex = { .b = { TRACE_STARTS_HERE, INVALID_TRACE, INVALID_TRACE, INVALID_TRACE } };
    const packed_bytes initialOffdiagPrevIndex = { .b = { INVALID_TRACE, INVALID_TRACE, INVALID_TRACE, INVALID_TRACE } };

    resultValueDiagonal[ 0 ].editDistance = ( diagonal0 == 0 ) ? initialDiagEditDistance : initialOffdiagEditDistance;
    resultValueDiagonal[ 1 ].editDistance = initialOffdiagEditDistance;
    resultValueDiagonal[ 0 ].prevIndex    = ( diagonal0 == 0 ) ? initialDiagPrevIndex : initialOffdiagPrevIndex;
    resultValueDiagonal[ 1 ].prevIndex    = initialOffdiagPrevIndex;

    {
        const int32_t bIndex00 = ( antiDiagonal0 - diagonal0 - 0 ) / 2;
        const int32_t aIndex00 = antiDiagonal0 - bIndex00;

        readSequenceData( aSequence, bSequence, aIndex00, bIndex00, aLength, bLength, direction, aData, bData, nextAData, nextBData );
    }

    // Variables for calculating B tracepoints
    // This values must be shuffled during diagonal shift

    int32_t preliminaryTracepointIndex = 0;

    // Current tracepoint position beeing calculated
    int32_t tracepointPosition = aFirstTracePointPosition;
    if ( tracepointPosition == 0 )
    {
        tracepointPosition = LOCAL_ALIGNMENT_TRACE_SPACE;
    }
    // The same but as a mask
//    packed_mask tracepointMask = { .b = { ( threadIdx.x * (BASEPAIRS_PER_THREAD / 2)     == tracepointPosition ) ? 0xff : 0,
//                                          ( threadIdx.x * (BASEPAIRS_PER_THREAD / 2) + 1 == tracepointPosition ) ? 0xff : 0,
//                                          ( threadIdx.x * (BASEPAIRS_PER_THREAD / 2) + 2 == tracepointPosition ) ? 0xff : 0,
//                                          0 } };
    packed_mask tracepointMask = makePackedBytes( 0 );
    // Mask identifying the column after a tracepoint - will become tracepointMask on the next iteration
    packed_mask afterTracepointMask = makePackedBytes( 0 );
    {
        // init tracepointMask:
        const int32_t bIndex00 = ( antiDiagonal0 - diagonal0 - 0 ) / 2;
        const int32_t aIndex00 = antiDiagonal0 - bIndex00;
        aboveOrOnDiagonal = makePackedBytes( ( aIndex00 >= bIndex00 ) ? 0x01010101 : 0 );
        aboveDiagonal0    = ( aIndex00 == bIndex00 ) ? makePackedBytes( 0x01010100 ) : aboveOrOnDiagonal;
        for ( int i = 0; i < BASEPAIRS_PER_THREAD / 2; i++ )
        {
            if (aIndex00 + i == tracepointPosition || aIndex00 + i == tracepointPosition - LOCAL_ALIGNMENT_TRACE_SPACE )
            {
                tracepointMask.b[i] = 0xff;
            }
        }
    }

    // Computed values for diagonal shifting
    int32_t currentDiagonalShift = 0;
    int32_t nextShift            = 0;

    for ( int i = 0; i < EVEN_ODD; i++ )
    {
        tracePointEditDistance[ i ] = { .editDistance = { .b = { INVALID_EDIT_DISTANCE, INVALID_EDIT_DISTANCE, INVALID_EDIT_DISTANCE, INVALID_EDIT_DISTANCE } },
            .prevIndex    = { .b = { INVALID_TRACE, INVALID_TRACE, INVALID_TRACE, INVALID_TRACE } } };
    }

    /**
     * Initialization
     */

    int32_t secondLimit = 2 * min( aLength, bLength ) - 4 * LOCAL_ALIGNMENT_TRACE_SPACE; // Factor 2 for even/odd
#ifdef DEBUG
    if ( threadIdx.x == 0 )
    {
        printf( "Unified for loop from %d til %d\n", antiDiagonal0, secondLimit );
    }
#endif
        // TODO: this variable helps computing latest a and b... needs revisinting
    bool errorRateCondition = false;
    // TODO: this variable helps computing latest a and b... needs revisinting
    int32_t latestShiftedThread = 0;    // Each loop iteration computes 2 anti diagonals (even/odd)
    for ( ;; antiDiagonal0 += 2 )
    {
        const int32_t bIndex00 = ( antiDiagonal0 - diagonal0 - 0 ) / 2;
        const int32_t aIndex00 = antiDiagonal0 - bIndex00;

#ifdef ENABLE_SANITY_CHECK
        {
            packed_bytes aDataCheck, bDataCheck;
            packed_bytes nextADataDummy, nextBDataDummy;
            readSequenceData( aSequence, bSequence, aIndex00, bIndex00, aLength, bLength, direction, aDataCheck, bDataCheck, nextADataDummy, nextBDataDummy );
            if (    ( aData.w & 0xffffff ) != ( aDataCheck.w & 0xffffff )
                 || ( bData.w & 0xffffff ) != ( bDataCheck.w & 0xffffff ) )
        {
                printf( "[%2d] sequenceData FAIL @ (%d, %d): %08x != %08x || %08x != %08x\n",
                        threadIdx.x, aIndex00, bIndex00, aData.w, aDataCheck.w, bData.w, bDataCheck.w );
        }
        }
#endif
        bool thisThreadHasComputedSomething = false;
        /**
         * Loop over even/odd thread diagonals
         */
#pragma unroll
        for ( int odd = 0; odd < 2; odd++ )
        {
            const int antiDiagonal = antiDiagonal0 + odd;

//#pragma unroll
//            for ( int valIndex = 0; valIndex < BASEPAIRS_PER_THREAD / 2; valIndex++ )
            {
                // Now increments in steps of one rather than two.
                // ToDo: Make sure all uses get updated accordingly
                const int32_t bIndex0 = ( antiDiagonal0 - diagonal0 ) / 2; // bIndex for bytePos 0
                const int32_t aIndex0 = antiDiagonal - bIndex0;            // aIndex for bytePos 0
#ifdef ENABLE_SANITY_CHECK
                for ( int i = 0; i < BASEPAIRS_PER_THREAD/2; i++)
                {
                    if ( aIndex0 + i >= 0 && bIndex0 + i >= 0 &&
                         !!( tracepointMask.b[ i ] ) !=
                             ( aIndex0 + i == tracepointPosition || aIndex0 + i == tracepointPosition - LOCAL_ALIGNMENT_TRACE_SPACE ) )
                    {
                        printf( "[%2d] tracepointMask FAIL aIndex %d bIndex %d bytePos %d tracepointPosition %d  %02x != %d\n",
                                threadIdx.x,
                                aIndex0 + i,
                                bIndex0 + i,
                                i,
                                tracepointPosition,
                                tracepointMask.b[ i ],
                                ( aIndex0 + i == tracepointPosition || aIndex0 + i == tracepointPosition - LOCAL_ALIGNMENT_TRACE_SPACE));
                    }
                }
#endif
#ifdef ENABLE_SANITY_CHECK
                if ( antiDiagonal0 <= secondLimit - abs( currentDiagonalShift ) && ( bIndex0 > bLength || aIndex0 > aLength ) )
                {
                    printf( "Unified loop index calculation is wrong at index %d: [%d + %d] ( %d > %d || %d > %d) /= %d \\= %d >>= %d\n",
                            inputIndex,
                            threadIdx.x,
                            odd,
                            bIndex0,
                            bLength,
                            aIndex0,
                            aLength,
                            antiDiagonal0,
                            diagonal0,
                            currentDiagonalShift );
                }
#endif
                const PackedEditDistanceTuple leftValue  = odd ? resultValueDiagonal[ 1 - odd ] : byteShuffleUp( resultValueDiagonal[ 1 - odd ] );
                const PackedEditDistanceTuple upperValue = odd ? byteShuffleDown( resultValueDiagonal[ 1 - odd ] ) : resultValueDiagonal[ 1 - odd ];

                const packed_bytes notSameChar =
                    makePackedBytes( ( ( aData.w ^ bData.w ) | ( ( aData.w ^ bData.w ) >> 1 ) ) & 0x010101 ); // do the basepairs differ?
                const packed_mask  inMatrixMask      = makePackedMask( ~ ( aData.w | bData.w ) ); // are both A and B inside the matrix?
                const packed_bytes aboveDiagonal     = ( odd ? aboveOrOnDiagonal : aboveDiagonal0 );
#ifdef ENABLE_SANITY_CHECK
                const packed_bytes aboveDiagonalCheck = ( !odd && aIndex0 == bIndex0 ) ? makePackedBytes( 0x01010100 ) : aboveOrOnDiagonal;
                if ( aboveDiagonal.w != aboveDiagonalCheck.w )
                {
                    printf( "[%2d] aboveDiagonal FAIL @(%d, %d) %08x != %08x\n",
                            threadIdx.x, aIndex0, bIndex0, aboveDiagonal.w, aboveDiagonalCheck.w );
                }
                for ( int bytePos = 0; bytePos < BASEPAIRS_PER_THREAD / 2; bytePos++ )
                {
                    const int32_t aIndex = aIndex0 + bytePos, bIndex = bIndex0 - bytePos;
                    const bool inMatrixCheck = ( bIndex >= 0 && bIndex <= bLength && aIndex >= 0 && aIndex <= aLength );
                    if ( !!inMatrixMask.b[ bytePos ] != inMatrixCheck )
                    {
                        printf( " inMatrix FAIL dir %d aIndex %d bIndex %d bytePos %d aData %08x bData %08x : %d != %d\n",
                                direction,
                                aIndex,
                                bIndex,
                                bytePos,
                                aData.w,
                                bData.w,
                                inMatrixMask.b[ bytePos ],
                                inMatrixCheck );
//                        continue; // limit the noise from other checks that will also fail as a consequence
                    }
                    if ( !inMatrixCheck )
                        continue; // we don't care if data outside matrix is different

                    char aCallData = aData.b[ bytePos ];
                    char bCallData = bData.b[ bytePos ];
                    char aCheckData, bCheckData;
                    switch ( direction )
                    {
                        case FORWARD:
                            aCheckData = (char)forwardSequenceReader( aSequence, aLength, aIndex - 1 ).w;
                            bCheckData = (char)forwardSequenceReader( bSequence, bLength, bIndex - 1 ).w;
                            break;
                        case REVERSE:
                        default:
                            aCheckData = (char)reverseSequenceReader( aSequence, aLength, aIndex - 1 ).w;
                            bCheckData = (char)reverseSequenceReader( bSequence, bLength, bIndex - 1 ).w;
                            break;
                    }
                    if ( aIndex != 0 && aCallData != aCheckData )
                    {
                        printf( "aData FAIL dir %d aIndex %d bIndex %d bytePos %d aData %08x shift %d : %02x != %02x\n",
                                direction,
                                aIndex,
                                bIndex,
                                bytePos,
                                aData.w,
                                bytePos,
                                aCallData,
                                aCheckData );
                    }
                    if ( bIndex != 0 && bCallData != bCheckData )
                    {
                        printf( "bData FAIL dir %d aIndex %d bIndex %d bytePos %d bData %08x shift %d : %02x != %02x\n",
                                direction,
                                aIndex,
                                bIndex,
                                bytePos,
                                bData.w,
                                bytePos,
                                bCallData,
                                bCheckData );
                    }
                    bool notSameCheck = ( ( aCallData & 0x3f ) != ( bCallData & 0x3f ) );
                    if ( !!notSameChar.b[ bytePos ] != notSameCheck )
                    {
                        printf( " notSameChar FAIL dir %d aIndex %d bIndex %d bytepos %d aData %08x bData %08x aCallData %02x bCallData %02x : %d != %d\n",
                                direction,
                                aIndex,
                                bIndex,
                                bytePos,
                                aData.w,
                                bData.w,
                                aCallData,
                                bCallData,
                                notSameChar.b[ bytePos ],
                                notSameCheck );
                    }
                }
#endif
//                if (aIndex0 >= 0 && bIndex0 >= 0)
//                    printf( "[%2d] (%3d,%3d) left %08x'%08x diag %08x'%08x upper %08x'%08x tracepoint %08x'%08x\n",
//                            threadIdx.x, aIndex0, bIndex0,
//                            leftValue.editDistance.w, leftValue.prevIndex.w,
//                            resultValueDiagonal[ odd ].editDistance.w, resultValueDiagonal[ odd ].prevIndex.w,
//                            upperValue.editDistance.w, upperValue.prevIndex.w,
//                            tracepointMask.w, afterTracepointMask.w );
                computeThreadValues( inMatrixMask,
                                     tracepointMask,
                                     afterTracepointMask,
                                     tracepointPosition,
                                     aIndex0,
                                     bIndex0,
                                     leftValue,
                                     resultValueDiagonal[ odd ],
                                     upperValue,
                                     resultValueDiagonal[ odd ],
                                     notSameChar,
                                     aboveOrOnDiagonal,
                                     aboveDiagonal,
                                     odd,
                                     tracePointEditDistance[ odd ],
                                     preliminaryTracepointIndex DEBUG_PARAMETERS_VALUES LEVENSHTEIN_MATRIX_PARAM_VALUES );
                if ( inMatrixMask.w & 0xffffff )
                {
                    thisThreadHasComputedSomething = true;
}
                }

            if ( odd )
            {
                bData = byteShuffleUp( bData, makePackedBytes( __shfl_sync( FULL_WARP_MASK, nextBData.w, WARP_SIZE - 1 ) ) );
                nextBData = byteShuffleUp( nextBData, makePackedBytes( 0xfbfbfbfb ), true );
            }
            else
            {
                aData = byteShuffleDown( aData, makePackedBytes( __shfl_sync( FULL_WARP_MASK, nextAData.w, 0 ) ) );
                nextAData = byteShuffleDown( nextAData, makePackedBytes( 0xfafafafa ), true );

                afterTracepointMask = tracepointMask;
                tracepointMask =
                    byteShuffleDown( tracepointMask, makePackedBytes( ( aIndex00 - BASEPAIRS_PER_WARP / 2 - 1 == tracepointPosition - 100 ) ? 0xff : 0 ) );
            }
            }

        if ( __all_sync( FULL_WARP_MASK, !thisThreadHasComputedSomething ) )
        {
#ifdef DEBUG
            if ( threadIdx.x == 0 )
            {
                printf( "Exited at unified because no threads has computed anything %d\n", antiDiagonal0 );
            }
#endif
            goto compute_matrix_edge;
        }
        if ( unlikely( antiDiagonal0 == ( tracepointPosition + 1 ) * 2 + BLOCK_SIZE * BASEPAIRS_PER_THREAD / 2 - currentDiagonalShift ) &&
             tracepointPosition < aLength )
        {

            if ( !checkErrorRateOk( prelimTracepoints,
                                    errorCorrelation,
                                    preliminaryTracepointIndex,
                                    editDistanceBase,
                                    tracePointEditDistance,
                                    tracepointPosition,
                                    diagonal0 DEBUG_PARAMETERS_VALUES ) )
            {
                // Restore state at last call to storeTracePoint
#ifdef DEBUG
                if ( threadIdx.x == 0 )
                    printf( "restore to state at preliminaryTracepointIndex %d\n", preliminaryTracepointIndex - 1 );
#endif
                if ( preliminaryTracepointIndex <= 0)
                {
                    // High error rate even before the first tracepoint - nothing to store
                    return;
                }
                for ( int odd = 0; odd < EVEN_ODD; odd++ )
                {
                    for ( int valIndex = odd; valIndex < BASEPAIRS_PER_THREAD; valIndex += 2 )
                    {
                        EditDistanceTuple ed =
                            prelimTracepoints[ preliminaryTracepointIndex - 1 ].editDistance[ threadIdx.x * BASEPAIRS_PER_THREAD + valIndex ];
                        resultValueDiagonal[ odd ].editDistance.b[ valIndex / 2 ] = ed.editDistance;
                        resultValueDiagonal[ odd ].prevIndex.b[ valIndex / 2 ]    = ed.prevIndex;
                    }
                }
                editDistanceBase = prelimTracepoints[ preliminaryTracepointIndex - 1 ].editDistanceBase;
#ifdef DEBUG
                if ( threadIdx.x == 0 )
                {
                    printf( "Exited at unified loop first check at %d\n", antiDiagonal0 );
                }
#endif

                errorRateCondition = true;
                goto compute_traceback;
            }
            storeTracePoint( prelimTracepoints, tracepointPosition, preliminaryTracepointIndex, editDistanceBase, tracePointEditDistance, diagonal0 );
            preliminaryTracepointIndex++;
            tracepointPosition += LOCAL_ALIGNMENT_TRACE_SPACE;

            nextShift = shiftAmount( currentDiagonalShift, diagonal0, resultValueDiagonal DEBUG_PARAMETERS_VALUES );
            shiftDiagonals( nextShift,
                            currentDiagonalShift,
                            diagonal0,
#ifdef DEBUG
                            antiDiagonal0,
#endif
                            resultValueDiagonal,
                            tracePointEditDistance,
                            tracepointMask DEBUG_PARAMETERS_VALUES );
            diagonal0 += nextShift * BASEPAIRS_PER_THREAD;
            currentDiagonalShift += nextShift * BASEPAIRS_PER_THREAD;
            latestShiftedThread = nextShift;
            nextShift = 0;

            editDistanceBase += editDistanceBaseUpdate( resultValueDiagonal );
#ifdef DEBUG
            if ( threadIdx.x == 0 )
            {
                printf( "updating editDistanceBase to %d\n", editDistanceBase );
            }
#endif
            // re-read data for next antidiagonal taking into account new shift
            const int32_t bIndex0 = ( antiDiagonal0 + 2 - diagonal0 ) / 2; // bIndex for bytePos 0
            const int32_t aIndex0 = antiDiagonal0 + 2 - bIndex0;           // aIndex for bytePos 0
            readSequenceData( aSequence, bSequence, aIndex0, bIndex0, aLength, bLength, direction, aData, bData, nextAData, nextBData );
            // re-compute above / on main diagonal masks
            aboveOrOnDiagonal = makePackedBytes( ( aIndex0 >= bIndex0 ) ? 0x01010101 : 0 );
            aboveDiagonal0    = ( aIndex0 == bIndex0 ) ? makePackedBytes( 0x01010100 ) : aboveOrOnDiagonal;

        }
        if ( antiDiagonal0 == 2 * LOCAL_ALIGNMENT_TRACE_SPACE )
        {
            // Ensure we do not run out of data in the larger gap before the first tracepoint is processed.
            const int32_t bIndex0 = ( antiDiagonal0 + 2 - diagonal0 ) / 2; // bIndex for bytePos 0
            const int32_t aIndex0 = antiDiagonal0 + 2 - bIndex0;           // aIndex for bytePos 0
            readSequenceData( aSequence, bSequence, aIndex0, bIndex0, aLength, bLength, direction, aData, bData, nextAData, nextBData );
        }
    }

compute_matrix_edge:

{

#ifdef DEBUG
    if ( threadIdx.x == 0 )
    {
        printf( "Storing tracepoint\n" );
    }
#endif
    storeTracePoint( prelimTracepoints, tracepointPosition, preliminaryTracepointIndex, editDistanceBase, tracePointEditDistance, diagonal0 );
    preliminaryTracepointIndex++;
    tracepointPosition += LOCAL_ALIGNMENT_TRACE_SPACE;
}
#ifdef D_DEBUG
    {
        for ( int tid = 0; tid < BLOCK_SIZE; tid++ )
        {
            for ( int valIndex = 0; valIndex < BASEPAIRS_PER_THREAD; valIndex++ )
            {
                if ( tid == threadIdx.x )
                {

                    printf( "thread %2d valIndex %d => (A: %d, B: %d [%d]) = %d and Prev %d Invalid: %d | Prev (%d , %d) = %d PPrev %d || %d\n",
                            threadIdx.x,
                            valIndex,
                            latestAindex[ valIndex ],
                            latestBindex[ valIndex ],
                            lastTracepointIndex[ valIndex ],
                            resultValueDiagonal[ valIndex ].editDistance,
                            resultValueDiagonal[ valIndex ].prevIndex,
                            lastTracepointIndex[ valIndex ],
                            prelimTracepoints[ preliminaryTracepointIndex - 1 ].aIndex + resultValueDiagonal[ valIndex ].prevIndex,
                            prelimTracepoints[ preliminaryTracepointIndex - 1 ].bIndex - resultValueDiagonal[ valIndex ].prevIndex,
                            prelimTracepoints[ preliminaryTracepointIndex - 1 ].editDistanceBase +
                                prelimTracepoints[ preliminaryTracepointIndex - 1 ].editDistanceDelta[ resultValueDiagonal[ valIndex ].prevIndex ],
                            tracepointPosition,
                            preliminaryTracepointIndex );
                }
            }
        }
    }
#endif

compute_traceback:
    EditDistanceTuple minEditDistance;

    int32_t minDiagonal;
    findMinimumEditDistance( resultValueDiagonal, diagonal0, minDiagonal, minEditDistance );

    int32_t antiDiagonal = antiDiagonal0 - ( minDiagonal % 2 );
    int32_t minBIndex    = min( ( antiDiagonal - minDiagonal ) / 2, bLength );
            int32_t minAIndex    = min( antiDiagonal - ( antiDiagonal - minDiagonal ) / 2, aLength );
            int32_t minLastTracepointIndex;
            if ( errorRateCondition )
    {
        minAIndex = tracepointPosition - LOCAL_ALIGNMENT_TRACE_SPACE;
        minBIndex = minAIndex - minDiagonal + latestShiftedThread * BASEPAIRS_PER_THREAD;
    }
    else
    {

        if ( minAIndex == aLength && minAIndex - minDiagonal <= bLength )
        {
            minBIndex = minAIndex - minDiagonal;
        }
        else if ( minBIndex == bLength && minBIndex + minDiagonal <= aLength )
        {
            minAIndex = minBIndex + minDiagonal;
        }
        else
        {
#ifdef ENABLE_DEBUG_PARAMETERS
            printf( "It should not happen... look at input index %d\n", inputIndex );
#else
            printf( "It should not happen... if it does -DENABLE_DEBUT_PARAMETERS and debug this one\n" );
            #endif
        }
                }

    minLastTracepointIndex =
        ( ( minAIndex - ( ( aFirstTracePointPosition == 0 ) ? LOCAL_ALIGNMENT_TRACE_SPACE : aFirstTracePointPosition ) + LOCAL_ALIGNMENT_TRACE_SPACE - 1 ) /
          LOCAL_ALIGNMENT_TRACE_SPACE );

            traceBackTrace( paths,
                            prelimTracepoints,
                            minAIndex,
                            minBIndex,
                            minLastTracepointIndex,
                            minEditDistance,
                            editDistanceBase,
                    aFirstTracePointPosition,
#ifdef DEBUG
                            preliminaryTracepointIndex,
#endif
                    direction DEBUG_PARAMETERS_VALUES );
}

} // namespace Matrix
} // namespace CudaTracePoints
