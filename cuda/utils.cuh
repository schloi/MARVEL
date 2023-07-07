
#pragma once

#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <vector>

#include <cstdio>
#include <cstdlib>

/**
 * This enables the cuda profiler range instructions
 */
#define ENABLE_PROFILE_INSTRUCTIONS

/**
 * Macros for starting and closing ranges in the profiles.
 * THESE MACROS SHOULD BE USED INSTEAD OF THE FUNCTION CALLS
 */
#ifdef ENABLE_PROFILE_INSTRUCTIONS
/**
 * Starts a range is color and label. This macro should be used to start and end ranges
 * @param label The label attached to the range
 * @param color The range color
 * @return
 * nvtxRangeId_t
 */
#define rangeStartWithColor( label, color ) CudaUtils::nvtxRangeStartWithColor( label, color )
/**
 * Ends a range.
 * @param range nvtxRangeId_t
 */
#define rangeEnd( range ) nvtxRangeEnd( range )
#else
#define rangeStartWithColor( label, color ) 0
#define rangeEnd( range ) ( (void)range )
#endif

namespace CudaUtils
{

/**
 * Starts a range is color and label. DO NOT USE THIS FUNCTION. USE THE MACRO INSTEAD
 * @param label The label attached to the range
 * @param color The range color
 * @return
 * nvtxRangeId_t
 */
inline nvtxRangeId_t nvtxRangeStartWithColor( const char* label, uint32_t color )
{

    nvtxEventAttributes_t eventAttrib = { 0 };
    eventAttrib.version               = NVTX_VERSION;
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType             = NVTX_COLOR_ARGB;
    eventAttrib.color                 = color;
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii         = label;
    return nvtxRangeStartEx( &eventAttrib );
}
/**
 * Check and reports cuda errors. DO NOT USE THIS FUNCTION, USE THE MACRO INSTEAD
 * @param err
 * @param file
 * @param line
 * @return
 */
inline cudaError_t __cudaSafeCallWithCudaError( cudaError err, const char* file, const int line )
{
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        exit( 1 );
    }

    return err;
}

/**
 * Check and reports cuda errors. DO NOT USE THIS FUNCTION, USE THE MACRO INSTEAD
 * @param err
 * @param file
 * @param line
 * @return
 */
inline cudaError_t _cudaCheckError( const char* file, const int line )
{

    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        exit( 1 );
    }

#ifdef DEBUG
    err = cudaDeviceSynchronize();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString( err ) );
        exit( 1 );
    }
#endif
    return err;
}

} // namespace CudaUtils

/**
 * CUDA_SAFE_CALL should be used in all calls to the CUDA API, which returns error codes,
 * to check and report errors.
 */
#define CUDA_SAFE_CALL( err ) CudaUtils::__cudaSafeCallWithCudaError( err, __FILE__, __LINE__ )
/**
 * CUDA_CHECK_ERROR should be used after all calls to the CUDA API, which does not return error codes,
 * to check and report errors.
 */
#define CUDA_CHECK_ERROR() CudaUtils::_cudaCheckError( __FILE__, __LINE__ )

/**
 * Ceil of an integer division. Note that integer divisions are always truncated
 * and that is why ceil(a/b) is always equals ceil(a/b). this macro ensures
 * that the result INT_DIV_CEIL(a,b) is equal a/b is a is multiple of b otherwise
 * a/b + 1 if not.
 */
#define INT_DIV_CEIL( x, y ) ( 1 + ( ( x - 1 ) / y ) )

/**
 * likely and unlikely definitions. Those are used to tell the compiler
 * what to expect from a boolean condition and optimized the code
 * towards this expectation.
 *
 * - if (likely(cond)) { scope a } else { scope b}: makes the compiler favors the execution of scope a
 * - if (unlikely(cond)) { scope a } else { scope b}: makes the compiler favors the execution of scope b
 * - favors means generate a continuous assembly code, i.e., not jumps for the favored scope and a jump to the other
 */
#if 1
#define likely( cond ) __builtin_expect( ( cond ), true )
#define unlikely( cond ) __builtin_expect( ( cond ), false )
#else
#define likely( cond ) ( cond )
#define unlikely( cond ) ( cond )
#endif