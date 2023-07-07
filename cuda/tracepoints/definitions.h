#pragma once

#include <cinttypes>
#include <cstddef>
#include <cstdint>

// Apologies for having a line of code in here - would like to find a better place
#ifndef __CUDACC__
# define __device__
#endif

#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define FULL_WARP_MASK 0xFFFFFFFF

#define BASEPAIRS_PER_THREAD 6
#define EVEN_ODD 2
#define BLOCK_SIZE WARP_SIZE // code assumes blocks fit into a single warp
#define BASEPAIRS_PER_WARP ( BASEPAIRS_PER_THREAD * BLOCK_SIZE )
#define LOCAL_ALIGNMENT_TRACE_SPACE 100
#define LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT 2

#ifndef BLOCKS_PER_SM
#define BLOCKS_PER_SM 16
#endif

#ifndef BLOCK_MULTIPLIER
#define BLOCK_MULTIPLIER 1
#endif

#ifndef B_INDEX_UPPER_BOUNDRY
#define B_INDEX_UPPER_BOUNDRY 150
#endif

#define MATRIX_INT_MAX INT32_MAX

#define MAX_EDIT_DISTANCE_OFFSET 250 // leave a little gap to INVALID_EDIT_DISTANCE in case of undetected overflow
#define INVALID_EDIT_DISTANCE 254    // leave a tiny gap to UINT8_MAX/2 in case of undetected overflow
#define INITIAL_INVALID_EDIT_DISTANCE 126    // leave a tiny gap to UINT8_MAX/2 in case of undetected overflow

typedef uint32_t matrix_int;
typedef uint8_t tracepoint_int;
typedef int16_t diagonal_int;

#define TRACE_STARTS_HERE 252 // first tracepoint
#define TRACE_ENDS_HERE   253 // trace has not passed the final tracepoint
#define INVALID_TRACE     254 // trace is outside range of antidigonal stored at previous tracepoint

#define FLOAT_SUB( x, y ) ( x - y )
#define FLOAT_TO_HALF( x ) ( x )

typedef float marvl_float_t;

typedef marvl_float_t marvl_error_t;

#if defined( ENABLE_DEBUG_PARAMETERS ) || defined( DEBUG ) || defined( ENABLE_SANITY_CHECK )
#define DEBUG_PARAMETERS , uint32_t inputIndex
#define DEBUG_PARAMETERS_VALUES , inputIndex
#else
#define DEBUG_PARAMETERS
#define DEBUG_PARAMETERS_VALUES
#endif

#define ERROR_RATE_WINDOW_SIZE 3
#if ERROR_RATE_WINDOW_SIZE > WARP_SIZE / 2
#error "ERROR_RATE_WINDOW_SIZE cannot be larger than WARP_SIZE/2, as the array elements are distributed amongst the threads of a half-warp"
#endif

#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )

#define LEVENSHTEIN_MATRIX_PARAMS , matrix_int* levenshteinMatrix
#define LEVENSHTEIN_MATRIX_PARAM_VALUES , levenshteinMatrix

#else
#define LEVENSHTEIN_MATRIX_PARAMS
#define LEVENSHTEIN_MATRIX_PARAM_VALUES

#endif

#define TRACE_POINT_VECTOR_SIZE( x ) ( INT_DIV_CEIL( ( x ), LOCAL_ALIGNMENT_TRACE_SPACE ) + 3 ) * 2

typedef union packed_bytes {
    uint8_t b[4]; // four bytes
    uint32_t w;   // one 32 bit word
} packed_bytes;

typedef packed_bytes packed_mask; // special use where each byte is either 0xff or zero
                                  // Other values are forbidden, but this is not enforced.

typedef struct EditDistanceTuple
{
    uint8_t editDistance;    // edit distance since start (seed)
    uint8_t prevIndex;       // index into prelim tracepoints at previous tracepoint
} EditDistanceTuple;

typedef struct PackedEditDistanceTuple
{
    // packed fields for up to for values:
    packed_bytes editDistance;    // edit distance since start (seed)
    packed_bytes prevIndex;       // index into prelim tracepoints at previous tracepoint
    __device__ inline EditDistanceTuple operator[] ( int i )
    {
        return { .editDistance = editDistance.b[i], .prevIndex = prevIndex.b[i] };
    }

} PackedEditDistanceTuple;


typedef struct preliminary_Tracepoint
{
    matrix_int bIndex;
#ifdef DEBUG
    matrix_int aIndex;
#endif
    matrix_int editDistanceBase;

    EditDistanceTuple editDistance[ BLOCK_SIZE * BASEPAIRS_PER_THREAD ]; // editDistance minus editDistanceBase, and index into preliminary_Tracepoint arrays at previous tracepoint
} preliminary_Tracepoint;

typedef struct Tracepoints
{
    // a pair is a tracepoint where:
    //    tracepoints[index % 2 == 0] is the edit distance
    //    tracepoints[index % 2 == 1] is the position
    tracepoint_int* tracepoints;
    // Number of elements in tracepoints
    uint32_t tracepointsLength;
    // The sum of all edit distances in tracepoints
    uint32_t differences;

    // Start and End indeces in both sequences where the tracepoints start and end
    uint32_t aStartIndex;
    uint32_t bStartIndex;
    uint32_t aEndIndex;
    uint32_t bEndIndex;
    bool skipped;
} Tracepoints;

/**
 * Sequence information with pointers to device and host
 */
typedef struct
{
    /**
     * Pointer to the read sequence on the device
     */
    char* deviceSequence;
    /**
     * Pointer to the read sequence on the host
     */
    char* hostSequence;
    /**
     * read length
     */
    uint32_t sequenceLength;
} SequenceInfo;

typedef union
{
    uint64_t unique;
    struct
    {
        int aRead;
        int bRead;
    } ReadIDs;
} ReadPairUnion;

typedef struct LocalAlignmentInput
{
    int32_t diagonal;
    int32_t antiDiagonal;
    uint32_t aStartIndex;
    marvl_float_t errorCorrelation;
    SequenceInfo aSequence;
    SequenceInfo bSequence;
    ReadPairUnion pair;
    int32_t diagonalBand;
    bool complement;

#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
    matrix_int* aReverseMatrix;
    matrix_int* bReverseMatrix;
    matrix_int* aForwardMatrix;
    matrix_int* bForwardMatrix;

#endif

} LocalAlignmentInput;

enum WaveDirection
{
    FORWARD = +1,
    REVERSE = -1
};

enum TracepointStopCondition
{
    CONTINUE,
    STOP,
    NONE
};
