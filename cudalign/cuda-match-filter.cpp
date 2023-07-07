
#include "cuda-match-filter.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda/tracepoints/stream-interface.hpp>
#include <cuda/utils.cuh>
#include <future>
#include <iostream>
#include <lib/oflags.h>
#include <parallel/algorithm>
#include <pthread.h>

#define SIZE_TO_MiB( s ) ( s / 1024 / 1024 )

extern int Hitmin;

#include "debug.h"

inline void parseTracepoints( Path* pathBuffer, const Tracepoints* tracepoint, uint16* data )
{
    *pathBuffer = { data,
                    (int)tracepoint->tracepointsLength,
                    (int)tracepoint->differences,
                    (int)tracepoint->aStartIndex,
                    (int)tracepoint->bStartIndex,
                    (int)tracepoint->aEndIndex,
                    (int)tracepoint->bEndIndex };
    for ( size_t index = 0; index < tracepoint->tracepointsLength; index++ )
    {
        data[ index ] = tracepoint->tracepoints[ index ];
    }
    // memcpy( data, tracepoint->tracepoints, sizeof( tracepoint_int ) * tracepoint->tracepointsLength );
}

#if defined( DEBUG_OVERLAP ) || defined( ENABLE_SANITY_CHECK )
void printOverlap( const Alignment* overlap, const Trace_Buffer* buffer, bool comp, const char* message )
{

    printf( "Overlap: %d x %d%s -> %s\n", overlap->aread, overlap->bread, comp ? " (Complement)" : "", message );
    printf( "\tPath:\n" );
    printf( "\t\tA: %d .. %d [%d]\n", overlap->path->abpos, overlap->path->aepos, overlap->path->aepos - overlap->path->abpos );
    printf( "\t\tB: %d .. %d [%d]\n", overlap->path->bbpos, overlap->path->bepos, overlap->path->bepos - overlap->path->bbpos );
    printf( "\t\tLength: %d\n", overlap->path->tlen );
    printf( "\t\tDiffs: %d\n", overlap->path->diffs );
    printf( "\t\tFlags: %d\n", overlap->flags );

    auto* trace = buffer->trace + (size_t)overlap->path->trace;
    printf( "\t\tTracepoints (%p): ", trace );

    int size = 0;
    int diff = 0;
    for ( int32_t i = 0; i < overlap->path->tlen; i += 2 )
    {
        printf( "(% 3d, %3d) ", trace[ i ], trace[ i + 1 ] );
        size += trace[ i + 1 ];
        diff += trace[ i ];
    }
    printf( "[%d, %d]\n", diff, size );
}
#endif

#ifdef ENABLE_SANITY_CHECK
void sanityCheck( const Alignment* alignment, int64_t index, Trace_Buffer* buffer, const char* message )
{
    Path* path = alignment->path;
    bool ok    = true;
    if ( path->abpos > path->aepos )
    {
        printf( "%s: Path '% 4d x % 4d' [% 5ld] has A Start Index '%d' bigger than A End Index '%d'\n",
                message,
                alignment->aread,
                alignment->aread,
                index,
                path->abpos,
                path->aepos );
        ok = false;
    }

    if ( path->bbpos > path->bepos )
    {
        printf( "%s: Path '% 4d x % 4d' [% 5ld] has B Start Index '%d' bigger than B End Index '%d'\n",
                message,
                alignment->aread,
                alignment->aread,
                index,
                path->bbpos,
                path->bepos );
        ok = false;
    }

    auto aLength = alignment->alen;
    auto bLength = alignment->blen;

    if ( path->abpos > aLength )
    {
        printf( "%s: Path '% 4d x % 4d' [% 5ld] has A Start Index '%d' bigger than Sequence Length '%d'\n",
                message,
                alignment->aread,
                alignment->bread,
                index,
                path->abpos,
                aLength );
        ok = false;
    }

    if ( path->aepos > aLength )
    {
        printf( "%s: Path '% 4d x % 4d' [% 5ld] has A End Index '%d' bigger than Sequence Length '%d'\n",
                message,
                alignment->aread,
                alignment->bread,
                index,
                path->aepos,
                aLength );
        ok = false;
    }

    if ( path->bbpos > bLength )
    {
        printf( "%s: Path '% 4d x % 4d' [% 5ld] has B Start Index '%d' bigger than Sequence Length '%d'\n",
                message,
                alignment->aread,
                alignment->bread,
                index,
                path->bbpos,
                bLength );
        ok = false;
    }

    if ( path->bepos > bLength )
    {
        printf( "%s: Path '% 4d x % 4d' [% 5ld] has B End Index '%d' bigger than Sequence Length '%d'\n",
                message,
                alignment->aread,
                alignment->bread,
                index,
                path->bepos,
                bLength );
        ok = false;
    }

    int32_t alignmentLength = 0;
    uint16_t* trace         = buffer->trace + reinterpret_cast<size_t>( path->trace );
    for ( int32_t i = 1; i < path->tlen; i += 2 )
    {
        uint16_t dist = trace[ i ];
        alignmentLength += dist;

        if ( dist > 255 )
        {
            printf( "%s: Path '% 4d x % 4d' [% 5ld] has a bigger tracepoint then expected: ( % 3d, % 3d)\n",
                    message,
                    alignment->aread,
                    alignment->bread,
                    index,
                    trace[ i - 1 ],
                    trace[ i ] );
            ok = false;
        }
    }
    if ( path->bbpos + alignmentLength != path->bepos )
    {
        printf( "%s: Path '% 4d x % 4d' [% 5ld] has expected B End Index '%d' but tracepoints ending at '%d'\n",
                message,
                alignment->aread,
                alignment->bread,
                index,
                path->bepos,
                path->bbpos + alignmentLength );
        ok = false;
    }

    if ( !ok )
    {
        printOverlap( alignment, buffer, alignment->flags & COMP_FLAG, "" );
        exit( 1 );
    }
}
#endif

typedef struct
{
    uint64 max;
    uint64 top;
    Path* paths;
} PathBuffer;

typedef struct
{
    Trace_Buffer traceBuffer{};
    PathBuffer pathBuffer{};
    std::vector<Alignment> alignments;
    std::vector<std::pair<size_t, size_t>> postProcessingResults;
    size_t counter{};
} BufferGroup;

void complementPath( const Tracepoints* path, const LocalAlignmentInput* currentInput, Path* parsedPath )
{
    parsedPath->abpos = (int)( currentInput->bSequence.sequenceLength - path->aEndIndex );
    parsedPath->bbpos = (int)( currentInput->aSequence.sequenceLength - path->bEndIndex );
    parsedPath->aepos = (int)( currentInput->bSequence.sequenceLength - path->aStartIndex );
    parsedPath->bepos = (int)( currentInput->aSequence.sequenceLength - path->bStartIndex );

    auto trace = static_cast<uint16_t*>( parsedPath->trace );
    uint16_t p;
    int i, j;

    i = parsedPath->tlen - 2;
    j = 0;
    while ( j < i )
    {
        p              = trace[ i ];
        trace[ i ]     = trace[ j ];
        trace[ j ]     = p;
        p              = trace[ i + 1 ];
        trace[ i + 1 ] = trace[ j + 1 ];
        trace[ j + 1 ] = p;
        i -= 2;
        j += 2;
    }
}

void alignmentPostProcessingThreads( BufferGroup* bufferGroup, int threadNumber )
{
    Path tempPath{};
    Work_Data* workData = New_Work_Data();
    bufferGroup->postProcessingResults.clear();

    // TODO: check if the sorting is compatible with downstream bridging

    std::__parallel::sort( bufferGroup->alignments.begin(), bufferGroup->alignments.end(), [ & ]( const Alignment& a1, const Alignment& a2 ) {
        if ( a1.flags != a2.flags )
        {
            return a1.flags < a2.flags;
        }
        if ( a1.aread != a2.aread )
        {
            return a1.aread < a2.aread;
        }
        if ( a1.bread != a2.bread )
        {
            return a1.bread < a2.bread;
        }

        const Path& p1 = bufferGroup->pathBuffer.paths[ a1.pathBufferOffset ];
        const Path& p2 = bufferGroup->pathBuffer.paths[ a2.pathBufferOffset ];

        if ( p1.abpos != p2.abpos )
        {
            return p1.abpos < p2.abpos;
        }
        if ( p1.aepos != p2.aepos )
        {
            return p1.aepos < p2.aepos;
        }

        if ( p1.bbpos != p2.bbpos )
        {
            return p1.bbpos < p2.bbpos;
        }
        if ( p1.bepos != p2.bepos )
        {
            return p1.bepos < p2.bepos;
        }

        return p1.diffs < p2.diffs;
    } );

#ifdef ENABLE_SANITY_CHECK
    if ( bufferGroup->alignments.size() != bufferGroup->pathBuffer.top )
    {
        printf( "Number of Alignent and Paths does not match!!! ALignments: %zu, Paths: %llu", bufferGroup->alignments.size(), bufferGroup->pathBuffer.top );
        exit( 1 );
    }
#endif

    Path* sortedPath = static_cast<Path*>( malloc( sizeof( Path ) * bufferGroup->pathBuffer.top ) );
    for ( int i = 0; i < bufferGroup->alignments.size(); i++ )
    {
        sortedPath[ i ]                               = bufferGroup->pathBuffer.paths[ bufferGroup->alignments[ i ].pathBufferOffset ];
        bufferGroup->alignments[ i ].path             = bufferGroup->pathBuffer.paths + i;
        bufferGroup->alignments[ i ].pathBufferOffset = i;
    }

    memcpy( bufferGroup->pathBuffer.paths, sortedPath, bufferGroup->pathBuffer.top * sizeof( Path ) );

    free( sortedPath );

    auto process = [ & ]( int startIndex, int endIndex ) {
        int numberOfOverlaps = static_cast<int>( endIndex - startIndex );
#if defined( DEBUG_INPUT_A ) || defined( DEBUG_INPUT )
        if ( startIndex >= bufferGroup->alignments.size() )
        {
            return;
        }
#endif
        auto tempAlignment = bufferGroup->alignments[ startIndex ];
        tempAlignment.path = &tempPath;

#ifdef ENABLE_SANITY_CHECK
        for ( int j = startIndex; j < numberOfOverlaps + startIndex; j++ )
        {
            sanityCheck( bufferGroup->alignments.data() + j, j, &bufferGroup->traceBuffer, "Before handle redundancies:" );
        }
#endif
#ifndef DISABLE_ALIGNMENT_POST_PROCESSING
        if ( numberOfOverlaps > 1 )
        {
            numberOfOverlaps = Handle_Redundancies(
                bufferGroup->pathBuffer.paths + startIndex, numberOfOverlaps, nullptr, &tempAlignment, workData, &bufferGroup->traceBuffer );
        }
#endif
#ifdef ENABLE_SANITY_CHECK
        for ( int j = startIndex; j < numberOfOverlaps + startIndex; j++ )
        {
            sanityCheck( bufferGroup->alignments.data() + j, j, &bufferGroup->traceBuffer, "After handle redundancies:" );
        }
#endif
        int baseIndex = startIndex;
        for ( int j = startIndex + 1; j < numberOfOverlaps + startIndex; j++ )
        {
            auto p1 = bufferGroup->pathBuffer.paths + baseIndex;
            auto p2 = bufferGroup->pathBuffer.paths + j;

            if ( ( p1->abpos == p2->abpos || p1->aepos == p2->aepos ) && ( p1->bbpos == p2->bbpos || p1->bepos == p2->bepos ) )
            {
                if ( p1->aepos - p1->abpos > p2->aepos - p2->abpos )
                {
                    bufferGroup->alignments[ j ].flags = bufferGroup->alignments[ j ].flags | OVL_DISCARD;
                }
                else if ( p1->aepos - p1->abpos < p2->aepos - p2->abpos )
                {
                    bufferGroup->alignments[ baseIndex ].flags = bufferGroup->alignments[ j ].flags | OVL_DISCARD;
                    baseIndex                                  = j;
                }
                else
                {
                    if ( p1->bepos - p1->bbpos >= p2->bepos - p2->bbpos )
                    {
                        bufferGroup->alignments[ j ].flags = bufferGroup->alignments[ j ].flags | OVL_DISCARD;
                    }
                    else
                    {
                        bufferGroup->alignments[ baseIndex ].flags = bufferGroup->alignments[ j ].flags | OVL_DISCARD;
                        baseIndex                                  = j;
                    }
                }
            }
            else
            {
                baseIndex = j;
            }
        }

        bufferGroup->postProcessingResults.emplace_back( startIndex, startIndex + numberOfOverlaps );
    };

    bool initialized = false;
    Alignment previousAlignment;
    int startIndex = 0;
    int endIndex   = 0;

    for ( auto alignment : bufferGroup->alignments )
    {
        if ( initialized && ( previousAlignment.aread != alignment.aread || previousAlignment.bread != alignment.bread ||
                              COMP( previousAlignment.flags ) != COMP( alignment.flags ) ) )
        {
            process( startIndex, endIndex );
            startIndex = endIndex;
        }
        endIndex++;
        initialized       = true;
        previousAlignment = alignment;
    }
    process( startIndex, endIndex );

    Free_Work_Data( workData );
}

void alingnmentPostProcessing( BufferGroup* bufferGroup, size_t numberOfThreads )
{
    std::vector<std::pair<size_t, size_t>> results;
    std::vector<std::future<void>> futures;

    for ( size_t i = 0; i < numberOfThreads; i++ )
    {
        futures.push_back( std::async( alignmentPostProcessingThreads, bufferGroup + i, i ) );
    }

    std::for_each( futures.begin(), futures.end(), [ & ]( std::future<void>& future ) { future.get(); } );
}

void saveOverlapsToIOBuffer(
    Overlap_IO_Buffer* iobuf, int small, int tbytes, BufferGroup* bufferGroup, size_t numberOfThreads, size_t aReadOffset, size_t bReadOffset )
{
    for ( uint32_t j = 0; j < numberOfThreads; j++ )
    {

        for ( auto pair : bufferGroup[ j ].postProcessingResults )
        {
            for ( size_t i = pair.first; i < pair.second; i++ )
            {

                auto alignment = bufferGroup[ j ].alignments[ i ];
#ifdef ENABLE_SANITY_CHECK
                sanityCheck( &alignment, i, &bufferGroup[ j ].traceBuffer, "During Save" );
#endif

                if ( alignment.path->aepos - alignment.path->abpos + alignment.path->bepos - alignment.path->bbpos < MINOVER )
                {
                    continue;
                }

#ifndef WRITE_DISCARDED_ALIGNMENTS
                if ( alignment.flags & OVL_DISCARD )
                {
                    continue;
                }
#endif

                Overlap overlap = {
                    *alignment.path, alignment.flags, static_cast<int>( alignment.aread + aReadOffset ), static_cast<int>( alignment.bread + bReadOffset ) };
                overlap.path.trace = bufferGroup[ j ].traceBuffer.trace + reinterpret_cast<size_t>( alignment.path->trace );
                if ( small )
                {
                    Compress_TraceTo8( &overlap, 1 );
                }
                AddOverlapToBuffer( iobuf, &overlap, tbytes );
            }
        }
    }
}

void initializeBufferGroup( BufferGroup* bufferGroup, uint32_t numberOfThreads, size_t totalSequenceLength, size_t numberOfInputs )
{
    bufferGroup->traceBuffer.max   = 2 * ( totalSequenceLength / 100 ) / numberOfThreads / 8;
    bufferGroup->traceBuffer.top   = 0;
    bufferGroup->traceBuffer.trace = static_cast<uint16_t*>( malloc( sizeof( short ) * bufferGroup->traceBuffer.max ) );
    if ( !bufferGroup->traceBuffer.trace )
    {
        std::cerr << "A Trace Buffer buffer was not allocated. Requested size " << SIZE_TO_MiB( bufferGroup->traceBuffer.max * sizeof( uint16 ) ) << std::endl;
        perror( strerror( errno ) );
        exit( 1 );
    }

    bufferGroup->pathBuffer.max = INT_DIV_CEIL( numberOfInputs, numberOfThreads );
    bufferGroup->pathBuffer.max *= 1.1;
    bufferGroup->pathBuffer.top   = 0;
    bufferGroup->pathBuffer.paths = static_cast<Path*>( malloc( sizeof( Path ) * bufferGroup->pathBuffer.max ) );
    if ( !bufferGroup->pathBuffer.paths )
    {
        std::cerr << "A Path Buffer buffer was not allocated. Requested size " << SIZE_TO_MiB( bufferGroup->pathBuffer.max * sizeof( Path ) ) << std::endl;
        perror( strerror( errno ) );
        exit( 1 );
    }

    bufferGroup->counter = 0;
}

void freeBufferGroup( BufferGroup* bufferGroup )
{
    free( bufferGroup->traceBuffer.trace );
    free( bufferGroup->pathBuffer.paths );
}

int32_t readPairHash( const ReadPairUnion& readPair ) { return readPair.ReadIDs.aRead ^ readPair.ReadIDs.bRead; }

void gpu_report_thread( int cudaDeviceId,
                        CudaStreamInterface::CudaLocalAlignmentStreamManager* streamManager,
                        size_t totalSequenceLength,
                        Work_Data* workData,
                        Overlap_IO_Buffer* ioBuffer,
                        uint32_t numberOfThreads,
                        ResourceManager* resourceManager )
{
    CUDA_SAFE_CALL( cudaSetDevice( cudaDeviceId ) );

    INIT_TIMING

    streamManager->initialize();

    BufferGroup aBufferGroup[ numberOfThreads ];
    BufferGroup bBufferGroup[ numberOfThreads ];

    int small, tbytes;

    if ( MR_tspace <= TRACE_XOVR )
    {
        small  = 1;
        tbytes = sizeof( uint8 );
    }
    else
    {
        small  = 0;
        tbytes = sizeof( uint16 );
    }

    auto range = rangeStartWithColor( "queueJobsAndWait", 0xff2060ffu );

    for ( uint32_t i = 0; i < numberOfThreads; i++ )
    {
        initializeBufferGroup( aBufferGroup + i, numberOfThreads, totalSequenceLength, streamManager->getNumberOfInputs() );
        initializeBufferGroup( bBufferGroup + i, numberOfThreads, totalSequenceLength, streamManager->getNumberOfInputs() );
    }

    rangeEnd( range );

    printf( "\tNumber of blocks %zu\n", streamManager->getNumberOfBlocks() );

    printf( "Inputs:\n" );
    printf( "\tElement Count %lu\n", streamManager->getNumberOfInputs() );
    printf( "\tMinimum alignment lenght %d\n", MINOVER );

    range = rangeStartWithColor( "queueJobsAndWait", 0xff2060ffu );

    START_TIMING

    streamManager->queueJobsAndWait(
        [ & ]( const Tracepoints* path, const size_t& index, const bool isBPath, const CudaStreamInterface::CudaLocalAlignmentStreamManager* self ) {
            Trace_Buffer* buffer;
            PathBuffer* pathBuffer;
            std::vector<Alignment>* aligmmentBuffer;
            size_t* counter;

            auto currentInput = self->getHostInputs() + index;

            int32_t bufferIndex             = readPairHash( currentInput->pair ) % numberOfThreads;
            BufferGroup* currentBufferGroup = ( isBPath ? bBufferGroup : aBufferGroup ) + bufferIndex;

            buffer          = &currentBufferGroup->traceBuffer;
            pathBuffer      = &currentBufferGroup->pathBuffer;
            aligmmentBuffer = &currentBufferGroup->alignments;
            counter         = &currentBufferGroup->counter;

            if ( buffer->top + path->tracepointsLength >= buffer->max )
            {
                buffer->max   = ( buffer->top + path->tracepointsLength ) + 10 * TRACE_CHUNK;
                buffer->trace = static_cast<uint16*>( realloc( buffer->trace, sizeof( short ) * buffer->max ) );
                if ( buffer->trace == nullptr )
                {
                    printf( "Not able to reallocate %llu MB for the trace buffer: buffer->top %llu | length %u ",
                            SIZE_TO_MiB( buffer->max * sizeof( uint16 ) ),
                            buffer->top,
                            path->tracepointsLength );
                    exit( 1 );
                }
            }
            if ( pathBuffer->top + 1 >= pathBuffer->max )
            {
                pathBuffer->max   = static_cast<double>( pathBuffer->top + 1 ) * 1.1;
                pathBuffer->paths = static_cast<Path*>( realloc( pathBuffer->paths, sizeof( Path ) * pathBuffer->max ) );

                if ( pathBuffer->paths == nullptr )
                {
                    printf( "Not able to reallocate %llu MB for the path buffer: top %llu | length %zu ",
                            SIZE_TO_MiB( pathBuffer->max * sizeof( PathBuffer ) ),
                            pathBuffer->top,
                            static_cast<size_t>( static_cast<double>( pathBuffer->top + 1 ) * 1.1 ) );
                    exit( 1 );
                }
            }

            Path* parsedPath = pathBuffer->paths + pathBuffer->top;
            parseTracepoints( parsedPath, path, buffer->trace + buffer->top );

            if ( currentInput->complement && isBPath )
            {
                complementPath( path, currentInput, parsedPath );
            }

            // This is a patch to reuse the daligner code
            // Most parts of the code expect the trace of path to have buffer->top and not the actual pointer
            parsedPath->trace = reinterpret_cast<void*>( buffer->top );

            buffer->top += parsedPath->tlen;

            Alignment alignment;

            alignment.path             = nullptr;
            alignment.pathBufferOffset = pathBuffer->top;
            alignment.flags            = currentInput->complement ? COMP_FLAG : 0;
            alignment.alen             = isBPath ? currentInput->bSequence.sequenceLength : currentInput->aSequence.sequenceLength;
            alignment.blen             = isBPath ? currentInput->aSequence.sequenceLength : currentInput->bSequence.sequenceLength;
            alignment.aseq             = isBPath ? currentInput->bSequence.hostSequence : currentInput->aSequence.hostSequence;
            alignment.bseq             = isBPath ? currentInput->aSequence.hostSequence : currentInput->bSequence.hostSequence;
            alignment.aread            = isBPath ? currentInput->pair.ReadIDs.bRead : currentInput->pair.ReadIDs.aRead;
            alignment.bread            = isBPath ? currentInput->pair.ReadIDs.aRead : currentInput->pair.ReadIDs.bRead;

            aligmmentBuffer->push_back( alignment );

            pathBuffer->top++;

            ( *counter )++;
        } );

    printf( "Alignment Finished\n" );

    END_TIMING("gpu alignments")

    rangeEnd( range );

    resourceManager->lockCpuResources();

    START_TIMING

    range = rangeStartWithColor( "post-processing (A)", 0xff2060ffu );
    alingnmentPostProcessing( aBufferGroup, numberOfThreads );
    rangeEnd( range );

    END_TIMING("alignmentPostProcessing")

    START_TIMING

    range = rangeStartWithColor( "save to io buffer (A)", 0xff2060ffu );
    saveOverlapsToIOBuffer( ioBuffer, small, tbytes, aBufferGroup, numberOfThreads, streamManager->getAReadsOffset(), streamManager->getBReadsOffset() );
    rangeEnd( range );

    END_TIMING("saveOverlapsToIOBuffer")

    START_TIMING

    range = rangeStartWithColor( "post-processing (B)", 0xff2060ffu );
    alingnmentPostProcessing( bBufferGroup, numberOfThreads );
    rangeEnd( range );

    END_TIMING("alignmentPostProcessing")

    START_TIMING

    range = rangeStartWithColor( "save to io buffer (A)", 0xff2060ffu );
    saveOverlapsToIOBuffer( ioBuffer, small, tbytes, bBufferGroup, numberOfThreads, streamManager->getBReadsOffset(), streamManager->getAReadsOffset() );
    rangeEnd( range );

    END_TIMING("saveOverlapsToIOBuffer")

    resourceManager->unlockCpuResources();

    for ( uint32_t i = 0; i < numberOfThreads; i++ )
    {
        freeBufferGroup( aBufferGroup + i );
        freeBufferGroup( bBufferGroup + i );
    }
}

void pre_report_thread( ResourceManager* resourceManager,
                        const Report_Arg& data,
                        int cudaDeviceId,
                        int numberOfStreams,
                        const SequenceInfo* currentABlock,
                        const SequenceInfo* currentBBlock,
                        const SequenceInfo* currentBBlockComplement,
                        uint32_t numberOfThreads )
{

    resourceManager->lockCpuResources();

    int* currentBandScore  = data.score;
    int* nextBandScore     = data.score + 1;
    int* previousBandScore = data.score - 1;

    int* lastPairStartIndex = data.lastp;

    //  Work_Data* work = data.work;
    int64_t internalIndex;

    auto* localAligmentInputs    = static_cast<LocalAlignmentInput*>( malloc( sizeof( LocalAlignmentInput ) * ( data.end - data.beg ) ) );
    size_t currentPointer        = 0;
    uint64_t totalSequenceLength = 0;
    uint32_t maxSequenceLength   = 0;

    auto range = rangeStartWithColor( "sorting seeds", 0xff4040ffu );

    std::__parallel::sort( data.khit + data.beg, data.khit + data.end, []( const SeedPair& i1, const SeedPair& i2 ) {
        if ( i1.aread == i2.aread )
        {
            if ( i1.bread == i2.bread )
            {
                int diagBand1 = i1.diag >> Binshift;
                int diagBand2 = i2.diag >> Binshift;

                if ( diagBand1 == diagBand2 )
                {
                    return i1.apos < i2.apos;
                }
                else
                {
                    return diagBand1 < diagBand2;
                }
            }
            else
            {

                return i1.bread < i2.bread;
            }
        }
        else
        {
            return i1.aread < i2.aread;
        }
    } );
    rangeEnd( range );

    range          = rangeStartWithColor( "selection seeds", 0xff4040ffu );
    SeedPair* hits = data.khit;

    for ( int64_t index = data.beg; index < data.end; index = internalIndex )
    {
        for ( internalIndex = index;
              hits[ index ].aread == hits[ internalIndex ].aread && hits[ index ].bread == hits[ internalIndex ].bread && internalIndex < data.end;
              internalIndex++ )
        {

            if ( hits[ internalIndex ].aread == hits[ internalIndex ].bread )
            {
                continue;
            }

            int diagonalBand = hits[ internalIndex ].diag >> Binshift;

            currentBandScore[ diagonalBand ] += std::min( Kmer, hits[ internalIndex ].apos - lastPairStartIndex[ diagonalBand ] );
            lastPairStartIndex[ diagonalBand ] = hits[ internalIndex ].apos;

            if ( currentBandScore[ diagonalBand ] + previousBandScore[ diagonalBand ] >= Hitmin ||
                 currentBandScore[ diagonalBand ] + nextBandScore[ diagonalBand ] >= Hitmin )
            {
                // Diagonal and anti diagonal was refacored from filter.c 1862:1865 and 1894
                int diagonal          = hits[ internalIndex ].diag;
                int antiDiagonal      = hits[ internalIndex ].apos * 2 - hits[ internalIndex ].diag;
                auto errorCorrelation = static_cast<float>( data.aspec->ave_corr );

                int aRead = hits[ internalIndex ].aread;
                bool complement;
                if ( aRead >= data.ablock->nreads )
                {
                    complement = true;
                    aRead -= data.ablock->nreads;
                }
                else
                {
                    complement = false;
                }

                SequenceInfo aSequence = currentABlock[ aRead ];
                SequenceInfo bSequence;
                if ( complement )
                {
                    bSequence = currentBBlockComplement[ hits[ internalIndex ].bread ];
                }
                else
                {
                    bSequence = currentBBlock[ hits[ internalIndex ].bread ];
                }

                localAligmentInputs[ currentPointer++ ] = ( LocalAlignmentInput ){ diagonal,
                                                                                   antiDiagonal,
                                                                                   static_cast<uint32_t>( hits[ internalIndex ].apos ),
                                                                                   errorCorrelation,
                                                                                   aSequence,
                                                                                   bSequence,
                                                                                   { .ReadIDs = { .aRead = aRead, .bRead = hits[ internalIndex ].bread } },
                                                                                   diagonalBand,
                                                                                   complement };

                totalSequenceLength += std::max( aSequence.sequenceLength, bSequence.sequenceLength );
                maxSequenceLength = std::max( maxSequenceLength, std::max( aSequence.sequenceLength, bSequence.sequenceLength ) );

                currentBandScore[ diagonalBand ] -= Kmer;
            }
        }

        for ( internalIndex = index;
              hits[ index ].aread == hits[ internalIndex ].aread && hits[ index ].bread == hits[ internalIndex ].bread && internalIndex < data.end;
              internalIndex++ )
        {
            int diagonalBand                   = hits[ internalIndex ].diag >> Binshift;
            currentBandScore[ diagonalBand ]   = 0;
            lastPairStartIndex[ diagonalBand ] = 0;
        }
    }

    rangeEnd( range );

    struct cudaDeviceProp props
    {
    };
    CUDA_SAFE_CALL( cudaGetDeviceProperties( &props, cudaDeviceId ) );

#if defined( DEBUG_INPUT_A ) && defined( DEBUG_INPUT_B )
#define XSTR( x ) STR( x )
#define STR( x ) #x
#pragma message "Restricting inputs to input " XSTR( DEBUG_INPUT_A ) " x " XSTR( DEBUG_INPUT_B )

    std::vector<LocalAlignmentInput> inputVector;
    for ( uint64_t i = 0; i < currentPointer; i++ )
    {
        auto input = localAligmentInputs[ i ];
        if ( ( input.pair.ReadIDs.aRead == DEBUG_INPUT_A && input.pair.ReadIDs.bRead == DEBUG_INPUT_B ) ||
             ( input.pair.ReadIDs.aRead == DEBUG_INPUT_B && input.pair.ReadIDs.bRead == DEBUG_INPUT_A ) )
        {
            printf( "%d x %d in input index: %lu\n", DEBUG_INPUT_A, DEBUG_INPUT_B, i );
            inputVector.push_back( input );
        }
    }
    free( localAligmentInputs );
    localAligmentInputs = static_cast<LocalAlignmentInput*>( malloc( sizeof( LocalAlignmentInput ) * inputVector.size() ) );
    memcpy( localAligmentInputs, inputVector.data(), inputVector.size() * sizeof( LocalAlignmentInput ) );
    currentPointer = inputVector.size();
#endif

    resourceManager->unlockCpuResources();

    uint32_t numberOfBlocks = props.multiProcessorCount * BLOCKS_PER_SM * BLOCK_MULTIPLIER;

    CudaStreamInterface::CudaLocalAlignmentStreamManager manager(
        localAligmentInputs, currentPointer, numberOfBlocks, numberOfStreams, data.ablock->ufirst, data.bblock->ufirst, maxSequenceLength, resourceManager );

    gpu_report_thread( cudaDeviceId, &manager, totalSequenceLength, data.work, data.iobuf, numberOfThreads, resourceManager );

    free( localAligmentInputs );
}

void GPU_Match_Filter( char* aname,
                       HITS_DB* ablock,
                       char* bname,
                       HITS_DB* bblock,
                       KmerPos* atable,
                       int alen,
                       KmerPos** btable,
                       int blen,
                       Align_Spec* asettings,
                       uint32_t numberOfThreads,
                       int deviceId,
                       ResourceManager* resourceManager,
                       int numberOfStreams,
                       const std::string& sortPath,
                       const SequenceInfo* currentABlock,
                       const SequenceInfo* currentBBlock,
                       const SequenceInfo* currentBBlockComplement )
{

    resourceManager->lockCpuResources();

    pthread_t threads[ numberOfThreads ];
    Merge_Arg parmm[ numberOfThreads ];
    Report_Arg parmr{};

    SeedPair *khit, *hhit;
    SeedPair *work1, *work2;
    bool freeWork1 = false;
    int64 nhits;

    KmerPos *asort, *bsort;

    asort = atable;
    bsort = *btable;

#ifdef ENABLE_OVL_IO_BUFFER
    int i;
    Overlap_IO_Buffer* buffer = OVL_IO_Buffer( asettings );
    parmr.iobuf               = buffer;

#endif

    // MR_tspace = Trace_Spacing( aspec );

    if ( VERBOSE )
        printf( "\nComparing %s to %s\n", aname, bname );

    if ( alen == 0 || blen == 0 )
        return;

    {
        uint i;
        int j, p;
        uint64 c;
        int limit;
        for ( i = 0; i < numberOfThreads; i++ )
        {
            parmm[ i ].MG_alist  = asort;
            parmm[ i ].MG_blist  = bsort;
            parmm[ i ].MG_ablock = ablock;
            parmm[ i ].MG_bblock = bblock;
            parmm[ i ].MG_self   = ( aname == bname );
        }

        parmm[ 0 ].abeg = parmm[ 0 ].bbeg = 0;
        for ( i = 1; i < numberOfThreads; i++ )
        {
            p = (int)( ( ( (int64)alen ) * i ) / numberOfThreads );
            if ( p > 0 )
            {
                c = asort[ p - 1 ].code;
                while ( asort[ p ].code == c )
                    p += 1;
            }
            parmm[ i ].abeg = parmm[ i - 1 ].aend = p;
            parmm[ i ].bbeg = parmm[ i - 1 ].bend = find_tuple( asort[ p ].code, bsort, blen );
        }
        parmm[ numberOfThreads - 1 ].aend = alen;
        parmm[ numberOfThreads - 1 ].bend = blen;

        for ( i = 0; i < numberOfThreads; i++ )
            for ( j = 0; j < MAXGRAM; j++ )
                parmm[ i ].hitgram[ j ] = 0;

        INIT_TIMING
        START_TIMING

        for ( i = 0; i < numberOfThreads; i++ )
            pthread_create( threads + i, nullptr, count_thread, parmm + i );

        for ( i = 0; i < numberOfThreads; i++ )
            pthread_join( threads[ i ], nullptr );

        END_TIMING("count_thread")

        if ( VERBOSE )
            printf( "\n" );
        if ( MEM_LIMIT > 0 )
        {
            int64 histo[ MAXGRAM ];
            int64 tom, avail;

            for ( j = 0; j < MAXGRAM; j++ )
                histo[ j ] = parmm[ 0 ].hitgram[ j ];
            for ( i = 1; i < numberOfThreads; i++ )
                for ( j = 0; j < MAXGRAM; j++ )
                    histo[ j ] += parmm[ i ].hitgram[ j ];

            // avail = (int64) (MEM_LIMIT - (sizeof_DB(ablock) + sizeof_DB(bblock))) / sizeof(KmerPos);
            avail = ( int64 )( MEM_LIMIT ) / sizeof( KmerPos );
            if ( asort == bsort || avail > alen + 2 * blen )
                avail = ( avail - alen ) / 2;
            else
                avail = avail - ( alen + blen );
            avail *= ( .98 * sizeof( KmerPos ) ) / sizeof( SeedPair );

            tom = 0;
            for ( j = 0; j < MAXGRAM; j++ )
            {
                tom += j * histo[ j ];
                if ( tom > avail )
                    break;
            }
            limit = j;

            if ( limit <= 1 )
            {
                fprintf( stderr, "\nError: Insufficient " );
                if ( MEM_LIMIT == MEM_PHYSICAL )
                    fprintf( stderr, " physical memory (%.1fGb), reduce block size\n", ( 1. * MEM_LIMIT ) / 0x40000000ll );
                else
                {
                    fprintf( stderr, " memory allocation (%.1fGb),", ( 1. * MEM_LIMIT ) / 0x40000000ll );
                    fprintf( stderr, " reduce block size or increase allocation\n" );
                }
                fflush( stderr );
                exit( 1 );
            }
            if ( limit < 10 )
            {
                fprintf( stderr, "\nWarning: Sensitivity hampered by low " );
                if ( MEM_LIMIT == MEM_PHYSICAL )
                    fprintf( stderr, " physical memory (%.1fGb), reduce block size\n", ( 1. * MEM_LIMIT ) / 0x40000000ll );
                else
                {
                    fprintf( stderr, " memory allocation (%.1fGb),", ( 1. * MEM_LIMIT ) / 0x40000000ll );
                    fprintf( stderr, " reduce block size or increase allocation\n" );
                }
                fflush( stderr );
            }
            if ( VERBOSE )
            {
                printf( "   Capping mutual k-mer matches over %d (effectively -t%d)\n", limit, (int)sqrt( 1. * limit ) );
                fflush( stdout );
            }

            for ( i = 0; i < numberOfThreads; i++ )
            {
                parmm[ i ].nhits = 0;
                for ( j = 1; j < limit; j++ )
                    parmm[ i ].nhits += j * parmm[ i ].hitgram[ j ];
                parmm[ i ].limit = limit;
            }
        }
        else
            for ( i = 0; i < numberOfThreads; i++ )
                parmm[ i ].limit = INT32_MAX;

        nhits = parmm[ 0 ].nhits;
        for ( i = 1; i < numberOfThreads; i++ )
            parmm[ i ].nhits = nhits += parmm[ i ].nhits;

        if ( VERBOSE )
        {
            printf( "   Hit count = " );
            Print_Number( nhits, 0, stdout );
            if ( asort == bsort || nhits * sizeof( SeedPair ) >= blen * sizeof( KmerPos ) )
                printf( "\n   Highwater of %.2fGb space\n",
                        ( 1. * static_cast<double>( alen * sizeof( KmerPos ) + 2 * nhits * sizeof( SeedPair ) ) / 0x40000000ll ) );
            else
                printf( "\n   Highwater of %.2fGb space\n",
                        ( 1. * static_cast<double>( ( alen + blen ) * sizeof( KmerPos ) + nhits * sizeof( SeedPair ) ) / 0x40000000ll ) );
            fflush( stdout );
        }

        if ( nhits == 0 )
            return;

        if ( asort == bsort )
        {
            freeWork1 = true;
            hhit = work1 = (SeedPair*)Malloc( sizeof( SeedPair ) * ( nhits + 1 ), "Allocating daligner hit vectors" );
        }
        else
        {
            if ( nhits * sizeof( SeedPair ) >= blen * sizeof( KmerPos ) )
            {
                bsort   = (KmerPos*)Realloc( bsort, sizeof( SeedPair ) * ( nhits + 1 ), "Reallocating daligner sort vectors" );
                *btable = bsort;
            }

            hhit = work1 = (SeedPair*)bsort;
        }
        khit = work2 = (SeedPair*)Malloc( sizeof( SeedPair ) * ( nhits + 1 ), "Allocating daligner hit vectors" );
        if ( hhit == nullptr || khit == nullptr || bsort == nullptr )
            exit( 1 );

        for ( i = 0; i < numberOfThreads; i++ )
        {
            parmm[ i ].MG_blist = bsort;
            parmm[ i ].MG_hits  = khit;
        }

        for ( i = numberOfThreads - 1; i > 0; i-- )
            parmm[ i ].nhits = parmm[ i - 1 ].nhits;
        parmm[ 0 ].nhits = 0;

        START_TIMING

        for ( i = 0; i < numberOfThreads; i++ )
            pthread_create( threads + i, nullptr, merge_thread, parmm + i );

        for ( i = 0; i < numberOfThreads; i++ )
            pthread_join( threads[ i ], nullptr );

        END_TIMING("merge_thread")

#ifdef TEST_PAIRS
        printf( "\nSETUP SORT:\n" );
        for ( i = 0; i < HOW_MANY && i < nhits; i++ )
            printf( " %6d / %6d / %5d / %5d\n", khit[ i ].aread, khit[ i ].bread, khit[ i ].apos, khit[ i ].diag );
#endif
    }

    {
        int i, j;
        int pairsort[ 13 ];
        int areads = ablock->nreads - 1;
        int breads = bblock->nreads - 1;
        int maxlen = ablock->maxlen;
        int abits, bbits, pbits;

        abits = 1;
        while ( areads > 0 )
        {
            areads >>= 1;
            abits += 1;
        }

        bbits = 0;
        while ( breads > 0 )
        {
            breads >>= 1;
            bbits += 1;
        }

        pbits = 1;
        while ( maxlen > 0 )
        {
            maxlen >>= 1;
            pbits += 1;
        }

#if __ORDER_LITTLE_ENDIAN__ == __BYTE_ORDER__
        for ( i = 0; i <= ( pbits - 1 ) / 8; i++ )
            pairsort[ i ] = 8 + i;
        j = i;
        for ( i = 0; i <= ( bbits - 1 ) / 8; i++ )
            pairsort[ j + i ] = 4 + i;
        j += i;
        for ( i = 0; i <= ( abits - 1 ) / 8; i++ )
            pairsort[ j + i ] = i;
#else
        for ( i = 0; i <= ( pbits - 1 ) / 8; i++ )
            pairsort[ i ] = 11 + i;
        j = i;
        for ( i = 0; i <= ( bbits - 1 ) / 8; i++ )
            pairsort[ j + i ] = 7 - i;
        j += i;
        for ( i = 0; i <= ( abits - 1 ) / 8; i++ )
            pairsort[ j + i ] = 3 - i;
#endif
        pairsort[ j + i ] = -1;

        khit = (SeedPair*)Radix_Sort( nhits, khit, hhit, pairsort );

        khit[ nhits ].aread = 0x7fffffff;
        khit[ nhits ].bread = 0x7fffffff;
        khit[ nhits ].apos  = 0x7fffffff;
        khit[ nhits ].diag  = 0x7fffffff;
    }

#ifdef TEST_CSORT
    {
        int i;

        printf( "\nCROSS SORT %lld:\n", nhits );
        for ( i = 0; i < HOW_MANY && i <= nhits; i++ )
            printf( " %6d / %6d / %5d / %5d\n", khit[ i ].aread, khit[ i ].bread, khit[ i ].apos, khit[ i ].diag );
    }
#endif

    int max_diag = ( ( ablock->maxlen >> Binshift ) - ( ( -bblock->maxlen ) >> Binshift ) ) + 1;
    int* space;

    parmr.ablock = ablock;
    parmr.bblock = bblock;
    parmr.khit   = khit;
    parmr.aspec  = asettings;
    parmr.two    = aname != bname && SYMMETRIC;

    parmr.beg = 0;
    parmr.end = nhits;

    space = (int*)Malloc( 3 * max_diag * sizeof( int ), "Allocating space for report thread" );
    if ( space == nullptr )
        exit( 1 );

    for ( i = 0; i < 3 * max_diag; i++ )
        space[ i ] = 0;

    parmr.score = space - ( ( -bblock->maxlen ) >> Binshift );

    parmr.lastp = parmr.score + max_diag;
    parmr.lasta = parmr.lastp + max_diag;
    parmr.work  = New_Work_Data();

    parmr.ofile1 = nullptr;
    parmr.ofile2 = nullptr;

    resourceManager->unlockCpuResources();

    pre_report_thread( resourceManager, parmr, deviceId, numberOfStreams, currentABlock, currentBBlock, currentBBlockComplement, numberOfThreads );

    int lastRead;
    if ( parmr.bblock->part < parmr.ablock->part )
        lastRead = parmr.bblock->ufirst + parmr.bblock->nreads - 1;
    else
        lastRead = parmr.ablock->ufirst + parmr.ablock->nreads - 1;

    Write_Overlap_Buffer( parmr.aspec, sortPath.c_str(), aname, bname, lastRead );
    Reset_Overlap_Buffer( parmr.aspec );

    free( space );
    free( work2 );
    if ( freeWork1 )
    {
        free( work1 );
    }
    Free_Work_Data( parmr.work );
}