
#pragma once

#include "cuda/utils.cuh"
#include <cuda_runtime.h>

#if defined( DEBUG ) || defined( _DEBUG )

template <typename T>
__device__ __host__ inline void printMatrix( T* matrix, int width, int height, bool inDeviceMemory )
{

    T* localMatrix;

    localMatrix = matrix;

    for ( int rowIndex = 0; rowIndex < height; rowIndex++ )
    {
        for ( int columnIndex = 0; columnIndex < width; columnIndex++ )
        {
            if ( columnIndex == 0 )
            {
                printf( "%d", readMatrixValue( localMatrix, width, height, rowIndex, columnIndex ) );
            }

            else
            {
                printf( ",%d", readMatrixValue( localMatrix, width, height, rowIndex, columnIndex ) );
            }
        }
        printf( "\n" );
    }
}

#endif

#ifdef LOCAL_ALIGNMENT_FORCE_SEQUENCE_CHECK

inline bool assertSequenceContent( char* sequence, size_t length, const char* sequenceId )
{
    bool ok = true;
    for ( size_t i = 0; i < length; i++ )
    {
        if ( sequence[ i ] < 0 || sequence[ i ] > 3 )
        {
            fprintf( stderr, "Sequence %s has an invalid value at '%lu'\n", sequenceId, i );
            ok = false;
        }
    }
    return ok;
}

#endif
