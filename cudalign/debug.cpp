
#include "debug.h"
#include "cuda/tracepoints/definitions.h"
#include <algorithm>
#include <cuda/utils.cuh>
#include <cuda_runtime.h>

#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
extern "C"
{
#include "db/DB.h"
}

matrix_int* levenshtein( char* aSequence, unsigned int aLength, unsigned int firstTracepoint, char* bSequence, uint32_t bLength, bool reverse )
{
#define MATRIX( i, j ) matrix[ ( i ) * ( bLength + 1 ) + ( j ) ]

    matrix_int* matrix;

    CUDA_SAFE_CALL( cudaHostAlloc( &matrix, sizeof( matrix_int ) * ( aLength + 1 ) * ( bLength + 1 ), cudaHostAllocMapped ) );

    printf( "Levenshtein %s matrix size  %d x %d = %lu Mb\n",
            reverse ? "reverse" : "forward",
            aLength + 1,
            ( bLength * 1 ),
            sizeof( matrix_int ) * ( aLength + 1 ) * ( bLength * 1 ) / 1024 / 1024 );

    printf( "A: " );
    if ( reverse )
    {
        for ( size_t i = aLength; i >= 0 && i >= aLength - 10; i-- )
        {
            printf( "%d ", aSequence[ i ] );
        }
    }
    else
    {
        for ( size_t i = 0; i < 10 && i < aLength; i++ )
        {
            printf( "%d ", aSequence[ i ] );
        }
    }

    printf( "\nB: " );
    if ( reverse )
    {
        for ( size_t i = bLength; i >= 0 && i >= bLength - 10; i-- )
        {
            printf( "%d ", bSequence[ i ] );
        }
    }
    else
    {
        for ( size_t i = 0; i < 10 && i < bLength; i++ )
        {
            printf( "%d ", bSequence[ i ] );
        }
    }
    printf( "\n" );

    for ( int i = 0; i < ( aLength + 1 ); i++ )
    {
        for ( int j = 0; j < ( bLength + 1 ); j++ )
        {
            if ( i == 0 )
            {
                MATRIX( i, j ) = j;
            }
            else if ( j == 0 )
            {
                MATRIX( i, j ) = i;
            }
            else
            {
                char a;
                char b;
                if ( reverse )
                {
                    a = aSequence[ aLength - i ];
                    b = bSequence[ bLength - j ];
                }
                else
                {
                    a = aSequence[ i - 1 ];
                    b = bSequence[ j - 1 ];
                }

                MATRIX( i, j ) = std::min<matrix_int>( ( ( a != b ) + MATRIX( i - 1, j - 1 ) ), 1 + std::min( MATRIX( i, j - 1 ), MATRIX( i - 1, j ) ) );
            }
        }
    }
    return matrix;
}

void computeLevenshtein( LocalAlignmentInput* inputs, size_t numberOfInputs, int firstA, int firstB )
{

    if ( numberOfInputs > 1 )
    {
        printf( "!WARNING! Computing CPU Levenshtein distance for more then one alignment. This could take forever!\n" );
    }

    for ( size_t i = 0; i < numberOfInputs; i++ )
    {

        printf( "Computing Levenshtein distance for %d x %d\n", inputs[ i ].pair.ReadIDs.aRead + firstA, inputs[ i ].pair.ReadIDs.bRead + firstB );

        auto aSequence = inputs[ i ].aSequence.hostSequence;
        auto bSequence = inputs[ i ].bSequence.hostSequence;

        uint32_t offset;
        uint32_t bFirstTracePoint;
        offset           = ( inputs[ i ].antiDiagonal - inputs[ i ].diagonal ) >> 1;
        bFirstTracePoint = inputs[ i ].complement
                               ? LOCAL_ALIGNMENT_TRACE_SPACE - ( inputs[ i ].bSequence.sequenceLength - offset ) % LOCAL_ALIGNMENT_TRACE_SPACE
                               : offset % LOCAL_ALIGNMENT_TRACE_SPACE;

        if ( bFirstTracePoint == LOCAL_ALIGNMENT_TRACE_SPACE )
        {
            bFirstTracePoint = 0;
        }

        inputs[ i ].aReverseMatrix = levenshtein(
            aSequence, offset + inputs[ i ].diagonal, ( ( inputs[ i ].diagonal + offset ) % LOCAL_ALIGNMENT_TRACE_SPACE ), bSequence, offset, true );
        inputs[ i ].aForwardMatrix = levenshtein( aSequence + ( offset + inputs[ i ].diagonal ),
                                                  inputs[ i ].aSequence.sequenceLength - offset - inputs[ i ].diagonal,
                                                  LOCAL_ALIGNMENT_TRACE_SPACE - ( ( inputs[ i ].diagonal + offset ) % LOCAL_ALIGNMENT_TRACE_SPACE ),
                                                  bSequence + offset,
                                                  inputs[ i ].bSequence.sequenceLength - offset,
                                                  false );

        inputs[ i ].bReverseMatrix = levenshtein( bSequence, offset, bFirstTracePoint, aSequence, offset + inputs[ i ].diagonal, true );

        inputs[ i ].bForwardMatrix = levenshtein( bSequence + offset,
                                                  inputs[ i ].bSequence.sequenceLength - offset,
                                                  LOCAL_ALIGNMENT_TRACE_SPACE - bFirstTracePoint,
                                                  aSequence + ( offset + inputs[ i ].diagonal ),
                                                  inputs[ i ].aSequence.sequenceLength - offset - inputs[ i ].diagonal,
                                                  false );
    }
}

#endif
