
#pragma once

#ifdef __cplusplus
#include <cstdio>
#include <cstdlib>
#endif

#ifdef DEBUG
#define exit_or_debug( code ) _exit_or_debug( code, __FILE__, __LINE__ )
#else
#define exit_or_debug( code ) exit( code )
#endif

#include "cuda/tracepoints/definitions.h"

inline void _exit_or_debug( int exitCode, const char* fileName, const int lineNumber )
{

    fprintf( stderr, "Exiting with code %d in %s:%d", exitCode, fileName, lineNumber );
    exit( exitCode );
}
#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
matrix_int*  levenshtein( char* aSequence, unsigned int aLength, unsigned int firstTracepoint, char* bSequence, uint32_t bLength, bool reverse );

void computeLevenshtein( LocalAlignmentInput* inputs, size_t numberOfInputs, int firstA, int firstB );
#endif
