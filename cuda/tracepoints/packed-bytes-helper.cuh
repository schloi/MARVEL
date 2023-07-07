#pragma once

#include "definitions.h"

namespace CudaTracePoints
{
namespace Matrix
{

__device__ packed_bytes generateByteIndex( int odd )
{
    packed_bytes result;

    // ToDo: inline asm version
    result.b[3] = 0;
    for ( int i = 0; i < BASEPAIRS_PER_THREAD / 2; i++ )
    {
        result.b[i] = threadIdx.x * BASEPAIRS_PER_THREAD + EVEN_ODD * i + odd;
    }
    return result;
}

__device__ packed_bytes makePackedBytes( uint32_t word )
{
    packed_bytes result;
    result.w = word;
    return result;
}

__device__ packed_bytes makePackedMask( uint32_t word )
{
    packed_mask result;
    asm ( "prmt.b32 %0, %1, %1, 0xba98;" : "=r"( result.w ) : "r"( word ) ); // sign extend individual bytes
#ifdef ENABLE_SANITY_CHECK
    packed_bytes input = makePackedBytes( word );
    packed_mask check;
    for( int i = 0; i < 4; i++)
    {
        check.b[i] = ( input.b[i] & 0x80 ) ? UINT8_MAX : 0;
    }
    if ( result.w != check.w )
    {
        printf( "ERROR in makePackedMask( %08x ) == %08x != %08x\n",
                word, result.w, check.w );
    }
#endif
    return result;
}

__device__ packed_bytes operator+ ( const packed_bytes& a, const packed_bytes& b )
{
    packed_bytes result;
    result.w = a.w + b.w;
    return result;
}

__device__ packed_mask operator<= ( packed_bytes a, packed_bytes b )
{
    packed_mask result = { makePackedMask( 0x80808080 + b.w - a.w ) }; // note: only works if |a - b| < 128 !
#ifdef ENABLE_SANITY_CHECK
    packed_mask check;
    for( int i = 0; i < 4; i++)
    {
        check.b[i] = ( a.b[i] <= b.b[i] ) ? UINT8_MAX : 0;
    }
    if ( ( result.w & 0xffffff ) != ( check.w & 0xffffff ) )
    {
        printf( "ERROR in operator<=( %08x, %08x ) == %08x != %08x\n",
                a.w, b.w, result.w, check.w );
    }
#endif
    return result;
}

__device__ packed_bytes selectBytes( packed_bytes trueValue, packed_bytes falseValue, packed_mask mask )
{
    packed_bytes result;
//    result = { .w = trueValue.w & mask.w | falseValue.w & ~mask.w } ; // should compile to a single LOP3.LUT instruction in the compiler were clever enough
    asm ( "lop3.b32 %0, %1, %2, %3, 0xe4;"
         : "=r"( result.w ) : "r"( trueValue.w ), "r"( falseValue.w ), "r"( mask.w ) );
#ifdef ENABLE_SANITY_CHECK
    packed_bytes check;
    for( int i = 0; i < 4; i++)
    {
        check.b[i] = mask.b[i] ? trueValue.b[i] : falseValue.b[i];
    }
    if ( result.w != check.w )
    {
        printf( "ERROR in selectBytes( %08x, %08x, %08x ) == %08x != %08x\n",
                trueValue.w, falseValue.w, mask.w, result.w, check.w );
    }
#endif
#ifdef ENABLE_SANITY_CHECK
    bool fail = false;
    for( int i = 0; i < 4; i++)
    {
        if ( mask.b[i] != UINT8_MAX && mask.b[i] != 0 )
        {
            fail = true;
        }
    }
    if ( fail )
    {
        printf( "[%2d]FAIL %08x is not a valid packed_mask value\n", threadIdx.x, mask.w );
    }
#endif
    return result;
}

__device__ uint8_t min3( packed_bytes values)
{
    return min( min( values.b[0], values.b[1] ), values.b[2] );
}


} // namespace Matrix
} // namespace CudaTracePoints
