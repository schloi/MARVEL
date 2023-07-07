
#pragma once

#include "definitions.h"

namespace CudaTracePoints
{
namespace WarpHelper
{

#ifndef UNSAFE_SYNC
#define safe_syncthreads() CudaTracePoints::WarpHelper::safe_syncthreads_line( __LINE__ )
#else
#define safe_syncthreads() __syncthreads()
#endif

__device__ void safe_syncthreads_line( int line )
{
    __shared__ int linecheck, threadcount;
    if ( threadIdx.x == 0 )
    {
        linecheck   = line;
        threadcount = 0;
    }
    asm volatile( "barrier.sync 1;" );
    atomicAdd( &threadcount, 1 );
    asm volatile( "barrier.sync 2;" );
    if ( linecheck != line )
        printf( "thread %d linecheck failed. Got %d, expected %d\n", threadIdx.x, linecheck, line );
    if ( threadIdx.x == 0 && threadcount != blockDim.x )
        printf( "only %d threads participated in sync on line %d\n", threadcount, line );
    asm volatile( "barrier.sync 3;" );
}


} // namespace WarpHelper
} // namespace CudaTracePoints