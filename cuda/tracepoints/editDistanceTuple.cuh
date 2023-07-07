#pragma once

#include "definitions.h"
#include "packed-bytes-helper.cuh"

namespace CudaTracePoints
{
namespace Matrix
{

// NOTE: when adding a field to this struct, computeValue() and blockReduceAndCarry() in matrix.cuh need to be updated AS WELL AS ALL function in this header
// file!

#define NIL_EDIT_DISTANCE                                                                                                        \
    {                                                                                                                            \
        .editDistance = INITIAL_INVALID_EDIT_DISTANCE, .prevIndex = INVALID_TRACE                                                                                      \
    }

#define PACKED_NIL_EDIT_DISTANCE                                                                                                 \
    {                                                                                                                            \
        .editDistance = { .b = { INITIAL_INVALID_EDIT_DISTANCE, INITIAL_INVALID_EDIT_DISTANCE,                                   \
                                 INITIAL_INVALID_EDIT_DISTANCE, INITIAL_INVALID_EDIT_DISTANCE } },                               \
        .prevIndex    = { .b = { INVALID_TRACE, INVALID_TRACE, INVALID_TRACE, INVALID_TRACE } }                                  \
    }

__device__ inline EditDistanceTuple makeEditDistanceTuple( uint8_t editDistance, uint8_t prevIndex )
{
    return { .editDistance = editDistance, .prevIndex = prevIndex };
}

__device__ inline bool operator!=( EditDistanceTuple a, EditDistanceTuple b )
{
    return a.editDistance != b.editDistance || a.prevIndex != b.prevIndex;
}

__device__ inline bool operator!=( PackedEditDistanceTuple a, PackedEditDistanceTuple b )
{
    return ( a.editDistance.w & 0xffffff ) != ( b.editDistance.w & 0xffffff ) ||
           ( a.prevIndex.w    & 0xffffff ) != ( b.prevIndex.w    & 0xffffff );
}

__device__ inline bool lessOrEqual( EditDistanceTuple a, EditDistanceTuple b, int32_t absCurrDiagonalA, int32_t absCurrDiagonalB )
{
    return ( (uint64_t) a.editDistance << 32 ) + absCurrDiagonalA <= ( (uint64_t) b.editDistance << 32 ) + absCurrDiagonalB;
}

__device__ inline bool operator<=( EditDistanceTuple a, EditDistanceTuple b )
{
    return a.editDistance <= b.editDistance;
}

__device__ inline bool operator<( EditDistanceTuple a, EditDistanceTuple b )
{
    return a.editDistance < b.editDistance;
}

__device__ inline PackedEditDistanceTuple selectED( PackedEditDistanceTuple trueValue, PackedEditDistanceTuple falseValue, packed_mask mask )
{
    PackedEditDistanceTuple result;
    result.editDistance = selectBytes( trueValue.editDistance, falseValue.editDistance, mask );
    result.prevIndex    = selectBytes( trueValue.prevIndex,    falseValue.prevIndex,    mask );
    return result;
}

__device__ inline EditDistanceTuple minEditDistance( EditDistanceTuple a, EditDistanceTuple b, bool preferSecond )
{
    if ( a.editDistance + preferSecond <= b.editDistance )
    {
        return a;
    }
    else
    {
        return b;
    }
}

__device__ inline PackedEditDistanceTuple minEditDistance( PackedEditDistanceTuple a, PackedEditDistanceTuple b, packed_bytes preferSecond )
{
    packed_mask mask = ( a.editDistance + preferSecond <= b.editDistance );
    PackedEditDistanceTuple result = { .editDistance = selectBytes( a.editDistance, b.editDistance, mask ),
                                       .prevIndex    = selectBytes( a.prevIndex,    b.prevIndex,    mask ) };
    return result;
}

__device__ inline EditDistanceTuple min3( PackedEditDistanceTuple x )
{
    EditDistanceTuple result = makeEditDistanceTuple( x.editDistance.b[0], x.prevIndex.b[0] );
    result = minEditDistance( result, makeEditDistanceTuple( x.editDistance.b[1], x.prevIndex.b[1] ), false );
    result = minEditDistance( result, makeEditDistanceTuple( x.editDistance.b[2], x.prevIndex.b[2] ), false );
    return result;
}

__device__ inline PackedEditDistanceTuple minWithMask( PackedEditDistanceTuple a, PackedEditDistanceTuple b, packed_mask& mask )
{
    PackedEditDistanceTuple result;
    uint32_t min128 = b.editDistance.w - a.editDistance.w + 0x80808080; // only works if less than 128 apart
    uint32_t min256 = min128 ^ b.editDistance.w ^ a.editDistance.w; // when it doesn't work, flip the top bit
    mask = makePackedMask( min256 );
    result.editDistance = selectBytes( a.editDistance, b.editDistance, mask );
    result.prevIndex    = selectBytes( a.prevIndex,    b.prevIndex,    mask );

//    printf( "[%2d] minWithMask( %08x, %08x ) = %08x, min128 %08x, min256 %08x, mask %08x\n",
//            threadIdx.x, a.editDistance.w, b.editDistance.w, result.editDistance.w, min128, min256, mask.w );
    return result;
}

__device__ EditDistanceTuple minWithIndex( PackedEditDistanceTuple low, PackedEditDistanceTuple high, int& index )
{
    packed_mask mask;
    PackedEditDistanceTuple minPackedEditDistance = minWithMask( high, low, mask );
    packed_bytes minPackedIndex = selectBytes( makePackedBytes( 0x07050301 ), makePackedBytes( 0x06040200 ), mask );

    EditDistanceTuple minEditDistance = minPackedEditDistance[0];
    int minIndex = minPackedIndex.b[0];
    for (int i=1; i < BASEPAIRS_PER_THREAD / 2; i++ )
    {
        if ( minPackedEditDistance[i] < minEditDistance )
        {
            minEditDistance = minPackedEditDistance[i];
            minIndex = minPackedIndex.b[i];
        }
    }
    index = minIndex;
    return minEditDistance;
}

__device__ inline EditDistanceTuple operator+( EditDistanceTuple x, matrix_int d )
{
    EditDistanceTuple result = x;
    result.editDistance += d;
    return result;
}

__device__ inline PackedEditDistanceTuple operator+( PackedEditDistanceTuple x, packed_bytes d )
{
    PackedEditDistanceTuple result = x;
    result.editDistance.w += d.w;
    return result;
}

__device__ PackedEditDistanceTuple warpShuffleDown( PackedEditDistanceTuple value, unsigned int offset = 1 )
{
    PackedEditDistanceTuple result;
    result.editDistance.w        = __shfl_down_sync( FULL_WARP_MASK, value.editDistance.w, offset );
    result.prevIndex.w           = __shfl_down_sync( FULL_WARP_MASK, value.prevIndex.w, offset );
    if ( threadIdx.x >= WARP_SIZE - offset )
    {
        result = PACKED_NIL_EDIT_DISTANCE;
    }

    return result;
}

__device__ PackedEditDistanceTuple warpShuffleUp( PackedEditDistanceTuple value, unsigned int offset = 1 )
{
    PackedEditDistanceTuple result;
    result.editDistance.w        = __shfl_up_sync( FULL_WARP_MASK, value.editDistance.w, offset );
    result.prevIndex.w           = __shfl_up_sync( FULL_WARP_MASK, value.prevIndex.w, offset );
    if ( threadIdx.x < offset )
    {
        result = PACKED_NIL_EDIT_DISTANCE;
    }

    return result;
}

__device__ packed_bytes byteShuffleDown( packed_bytes value, packed_bytes boundaryValue, bool dense = false )
{
    uint32_t nextValue = __shfl_down_sync( FULL_WARP_MASK, value.w, 1 );
    if ( threadIdx.x >= WARP_SIZE - 1 )
    {
        nextValue = boundaryValue.w;
    }
    if ( dense )
    {
        // shuffle through all 4 bytes per thread
        value.w = ( value.w >> 8 | nextValue << 24 );
    }
    else
    {
        // shuffle only through the 3 bytes representing a basepair
        value.w = value.w >> 8 | ( nextValue & 0xff ) << 16;
    }
    return value;
};

__device__ packed_bytes byteShuffleUp( packed_bytes value, packed_bytes boundaryValue, bool dense = false )
{
    uint32_t nextValue = __shfl_up_sync( FULL_WARP_MASK, value.w, 1 );
    if ( threadIdx.x < 1 )
    {
        nextValue = boundaryValue.w;
    }
    if ( dense )
    {
        // shuffle through all 4 bytes per thread
        value.w = ( value.w << 8 | nextValue >> 24 );
    } else
    {
        // shuffle only through the 3 bytes representing a basepair
        value.w = value.w << 8 | ( nextValue & 0xff0000 ) >> 16;
    }
    return value;
};

__device__ PackedEditDistanceTuple byteShuffleDown( PackedEditDistanceTuple value )
{
    uint32_t nextEditDistance   = __shfl_down_sync( FULL_WARP_MASK, value.editDistance.w, 1 );
    uint32_t nextPrevIndex      = __shfl_down_sync( FULL_WARP_MASK, value.prevIndex.w, 1 );
    if ( threadIdx.x >= WARP_SIZE - 1 )
    {
        nextEditDistance += BASEPAIRS_PER_THREAD; // prevent selection of this value
        nextPrevIndex     = INVALID_TRACE;
    }
    PackedEditDistanceTuple result;
    asm( "prmt.b32 %0, %1, %2, 0x7421;" : "=r"( result.editDistance.w ) : "r"( value.editDistance.w ), "r"( nextEditDistance ) );
    asm( "prmt.b32 %0, %1, %2, 0x7421;" : "=r"( result.prevIndex.w    ) : "r"( value.prevIndex.w ),    "r"( nextPrevIndex    ) );
#ifdef ENABLE_SANITY_CHECK
    PackedEditDistanceTuple check = { .editDistance = { .w = value.editDistance.w >> 8 & 0xffff| nextEditDistance << 16 & 0xff0000 },
                                      .prevIndex    = { .w = value.prevIndex.w    >> 8 & 0xffff| nextPrevIndex    << 16 & 0xff0000 } };
    if (result != check )
    {
        printf( "[%2d] byteShuffleDown(%08x, %08x) FAIL (%08x, %08x) != (%08x, %08x)\n",
                threadIdx.x,
                value.editDistance.w,  value.prevIndex.w,
                result.editDistance.w, result.prevIndex.w,
                check.editDistance.w,  check.prevIndex.w);
    }
#endif
    return result;
};

__device__ PackedEditDistanceTuple byteShuffleUp( PackedEditDistanceTuple value )
{
    uint32_t nextEditDistance   = __shfl_up_sync( FULL_WARP_MASK, value.editDistance.w, 1 );
    uint32_t nextPrevIndex      = __shfl_up_sync( FULL_WARP_MASK, value.prevIndex.w, 1 );
    if ( threadIdx.x < 1 )
    {
        nextEditDistance += BASEPAIRS_PER_THREAD << 16; // prevent selection of this value
        nextPrevIndex     = INVALID_TRACE << 16;
    }
    PackedEditDistanceTuple result;
    asm( "prmt.b32 %0, %1, %2, 0x7106;" : "=r"( result.editDistance.w ) : "r"( value.editDistance.w ), "r"( nextEditDistance ) );
    asm( "prmt.b32 %0, %1, %2, 0x7106;" : "=r"( result.prevIndex.w    ) : "r"( value.prevIndex.w ),    "r"( nextPrevIndex    ) );
#ifdef ENABLE_SANITY_CHECK
    PackedEditDistanceTuple check = { .editDistance = { .w = value.editDistance.w << 8 & 0xffff00 | nextEditDistance >> 16 & 0xff },
                                      .prevIndex    = { .w = value.prevIndex.w    << 8 & 0xffff00 | nextPrevIndex    >> 16 & 0xff } };
    if (result != check )
    {
        printf( "[%2d] byteShuffleDown(%08x, %08x) FAIL (%08x, %08x) != (%08x, %08x)\n",
                threadIdx.x,
                value.editDistance.w,  value.prevIndex.w,
                result.editDistance.w, result.prevIndex.w,
                check.editDistance.w,  check.prevIndex.w);
    }
#endif
    return result;
};

__device__ EditDistanceTuple warpShuffle( unsigned int activeMask, EditDistanceTuple value, unsigned int lane )
{
    EditDistanceTuple result;
    result.editDistance        = __shfl_sync( activeMask, value.editDistance, lane );
    result.prevIndex           = __shfl_sync( activeMask, value.prevIndex, lane );

    return result;
}

} // namespace Matrix
} // namespace CudaTracePoints
