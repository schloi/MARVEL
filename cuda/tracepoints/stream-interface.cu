#include "cuda/tracepoints/tracepoints.cuh"
#include "cuda/utils.cuh"

#include "definitions.h"
#include "stream-interface.hpp"
#include <algorithm>
#include <unistd.h>

#ifdef DUMP_ERROR_FASTA
extern "C"
{
#include "db/DB.h"
};
#endif
#include "cudalign/debug.h"

namespace CudaStreamInterface
{

CudaLocalAlignmentStreamManager::CudaLocalAlignmentStreamManager( LocalAlignmentInput* inputs,
                                                                  size_t numberOfInputs,
                                                                  size_t numberOfBlocks,
                                                                  size_t numberOfStreams,
                                                                  size_t aReadsOffset,
                                                                  size_t bReadsOffset,
                                                                  size_t maxSequenceLength,
                                                                  ResourceManager* resourceManager ) :
    _deviceMemoryManager( resourceManager ),
    _hostInputs( inputs ), _numberOfInputs( numberOfInputs ), _aReadsOffset( aReadsOffset ), _bReadsOffset( bReadsOffset ),
    _tracePointVectorSize( ( INT_DIV_CEIL( maxSequenceLength, 100 ) + 3 ) * 2 ), _tracePointVectorTotalSize( this->_tracePointVectorSize * numberOfInputs ),
    _tracePointTotalSize( numberOfInputs ), _maxSequenceLength( maxSequenceLength )
{

#ifdef FORCE_SEQUENTIAL
    this->_numberOfBlocks  = 1;
    this->_numberOfStreams = 1;
#else
    this->_numberOfBlocks  = numberOfBlocks;
    this->_numberOfStreams = numberOfStreams;
#endif

#ifdef DEBUG
    CUDA_SAFE_CALL( cudaDeviceSetLimit( cudaLimitPrintfFifoSize, 64 * 1024 * 1024 ) );
#endif
}

void CudaLocalAlignmentStreamManager::initialize()
{
    auto range = rangeStartWithColor( "shuffling input", 0xff4040ffu );
    this->shuffleInputForProcessing();
    rangeEnd( range );

    this->initializeMemory();
    this->createStreams();
}

void CudaLocalAlignmentStreamManager::shuffleInputForProcessing()
{

    std::vector<uint64_t> sizeOfInputsForPairBand;

    this->_maxInputsPerBlock       = 0;
    uint64_t currentNumberOfInputs = 1;

#ifdef DEBUG_INPUT
#define XSTR( x ) STR( x )
#define STR( x ) #x
#pragma message "Restricting inputs to input " XSTR( DEBUG_INPUT )
    printf( "DEBUG_INPUT = %d -> Only this single input will be aligned\n", DEBUG_INPUT );
    if ( DEBUG_INPUT > this->_numberOfInputs )
    {
        this->_numberOfInputs = 0;
        return;
    }
    this->_hostInputs = this->_hostInputs + DEBUG_INPUT;

    this->_numberOfBlocks  = 1;
    this->_numberOfStreams = 1;
    this->_numberOfInputs  = 1;

#endif

#if defined( DEBUG ) && defined( COMPUTE_LEVENSHTEIN )
    computeLevenshtein( this->_hostInputs, this->_numberOfInputs, this->_aReadsOffset, this->_aReadsOffset );
#endif

    size_t freeMemory  = this->_deviceMemoryManager->getWorkMemorySize() - this->_numberOfInputs * ( sizeof( LocalAlignmentInput ) + sizeof( int64_t ) );
    uint64_t seedLimit = freeMemory / ( ( 2 * sizeof( Tracepoints ) + TRACE_POINT_VECTOR_SIZE( _maxSequenceLength ) * 2 * sizeof( tracepoint_int ) ) *
                                        this->_numberOfBlocks * this->_numberOfStreams );

    printf( "\tSeed limit: %lu\n", seedLimit );
    printf( "\tLongest sequence: %lu\n", _maxSequenceLength );

    for ( uint64_t i = 1; i < this->_numberOfInputs; i++ )
    {
        if ( ( this->_hostInputs[ i ].pair.unique == this->_hostInputs[ i - 1 ].pair.unique &&
               this->_hostInputs[ i ].diagonalBand == this->_hostInputs[ i - 1 ].diagonalBand ) &&
             currentNumberOfInputs < seedLimit )
        {
            currentNumberOfInputs++;
        }
        else
        {
            sizeOfInputsForPairBand.push_back( currentNumberOfInputs );
            this->_maxInputsPerBlock = std::max( this->_maxInputsPerBlock, currentNumberOfInputs );
            currentNumberOfInputs    = 1;
        }
    }
    this->_maxInputsPerBlock = std::max( this->_maxInputsPerBlock, std::max( currentNumberOfInputs, seedLimit ) );

    sizeOfInputsForPairBand.push_back( currentNumberOfInputs );

    currentNumberOfInputs = sizeOfInputsForPairBand[ 0 ];
    uint64_t currentIndex = sizeOfInputsForPairBand[ 0 ];

    this->_hostInputRanges.push_back( 0 );
    for ( size_t i = 1; i < sizeOfInputsForPairBand.size(); i++ )
    {

        uint64_t& size = sizeOfInputsForPairBand[ i ];
        if ( currentNumberOfInputs + size > this->_maxInputsPerBlock )
        {
            this->_hostInputRanges.push_back( currentIndex );
            currentNumberOfInputs = size;
        }
        else
        {
            currentNumberOfInputs += size;
        }
        currentIndex += size;
    }
    // Last start index
    this->_hostInputRanges.push_back( this->_numberOfInputs );

#ifdef ENABLE_SANITY_CHECK
    for ( unsigned long i : sizeOfInputsForPairBand )
    {
        if ( i > this->_maxInputsPerBlock )
        {
            printf( "One of the input ranges are bigger (%lu) than the allowed inputs per block (%zu) ", i, this->_maxInputsPerBlock );
        }
    }
    size_t total = 0;
    for ( size_t i = 1; i < _hostInputRanges.size(); i++ )
    {
        total += this->_hostInputRanges[ i ] - this->_hostInputRanges[ i - 1 ];
        if ( this->_hostInputRanges[ i ] - this->_hostInputRanges[ i - 1 ] > this->_maxInputsPerBlock )
        {
            printf( "One of the input ranges (%lu) are bigger (%lu) than the allowed inputs per block (%zu) ",
                    i - 1,
                    this->_hostInputRanges[ i ] - this->_hostInputRanges[ i - 1 ],
                    this->_maxInputsPerBlock );
        }
    }
    if ( total != this->_numberOfInputs )
    {
        printf( "The computed total (%zu) does not match the expected total (%zu)", total, this->_numberOfInputs );
    }

#endif
}

void CudaLocalAlignmentStreamManager::initializeMemory()
{
    // Inputs
    printf( "Memory allocation for inputs:\n" );
    printf( "\tLocal alignment inputs: %lu Mb on the device\n", this->_numberOfInputs * sizeof( LocalAlignmentInput ) / 1024 / 1024 );
    printf( "\tLocal alignment input indices: %lu Mb on the device\n", this->_hostInputRanges.size() * sizeof( uint64_t ) / 1024 / 1024 );

    // Ouputs

    if ( _maxSequenceLength > this->_deviceMemoryManager->getMaxReadLength() )
    {
        printf(
            "There is/are read(s) longer (%zu) then the max allowed read length (%zu).", _maxSequenceLength, this->_deviceMemoryManager->getMaxReadLength() );
    }

    // A sequece has "length"/"tracepoint space" tracepoints + 3 tracepoints for handling the edges of each wave.
    // 3 extras and not 4 because the forward wave reuses the last trance point of the reverse wave
    // Since a trancepoint has 2 values, then "tracepoint count * 2"
    this->_tracePointVectorSize = TRACE_POINT_VECTOR_SIZE( _maxSequenceLength );

    // Each chunk is the amount of memory needed per block, considering that one vector is needed for A and B (2 x a single vector)
    this->_tracePointVectorChunkSize = this->_tracePointVectorSize * this->_maxInputsPerBlock * 2;
    // For processing  one chunk is needed per block per stream
    this->_tracePointVectorBlockChunkSize = this->_numberOfBlocks * this->_tracePointVectorChunkSize;
    this->_tracePointVectorTotalSize      = this->_tracePointVectorBlockChunkSize * this->_numberOfStreams;

    // One tracepoint structure for A and one for B (2 x a single structure)
    this->_tracePointChunkSize = this->_maxInputsPerBlock * 2;
    // For processing  one chunk is needed per block per stream
    this->_tracePointBlockChunkSize = this->_tracePointChunkSize * this->_numberOfBlocks;
    this->_tracePointTotalSize      = this->_tracePointBlockChunkSize * this->_numberOfStreams;

    printf( "Memory allocation for outputs:\n" );
    printf( "\tTracepoint structures: %lu Mb on the device and host\n", this->_tracePointTotalSize * sizeof( Tracepoints ) / 1024 / 1024 );
    printf( "\tTracepoint vectors: %lu Mb host memory mapped to the device\n", this->_tracePointVectorTotalSize * sizeof( tracepoint_int ) / 1024 / 1024 );

    this->_workMemory = this->_deviceMemoryManager->getWorkMemory();

    this->_deviceInputs            = reinterpret_cast<LocalAlignmentInput*>( this->_workMemory->device );
    this->_deviceInputRanges       = reinterpret_cast<uint64_t*>( this->_deviceInputs + this->_numberOfInputs );
    this->_deviceTracePoints       = reinterpret_cast<Tracepoints*>( this->_deviceInputRanges + this->_hostInputRanges.size() );
    this->_deviceTracePointVectors = reinterpret_cast<tracepoint_int*>( this->_deviceTracePoints + this->_tracePointTotalSize );

    // throwAwayDI and throwAwayBlocks is just to align device and host memory and those are not used.
    auto throwAwayDI             = reinterpret_cast<LocalAlignmentInput*>( this->_workMemory->host );
    auto throwAwayBlocks         = reinterpret_cast<uint64_t*>( throwAwayDI + this->_numberOfInputs );
    this->_hostTracePoints       = reinterpret_cast<Tracepoints*>( throwAwayBlocks + this->_hostInputRanges.size() );
    this->_hostTracePointVectors = reinterpret_cast<tracepoint_int*>( this->_hostTracePoints + this->_tracePointTotalSize );

    /* Coping the inputs and input ranges. Note that even though the host memory is mapped onto the device,
     * these both arrays are created in another memory region
     */
    CUDA_SAFE_CALL( cudaMemcpy( this->_deviceInputs, this->_hostInputs, this->_numberOfInputs * sizeof( LocalAlignmentInput ), cudaMemcpyHostToDevice ) );
    CUDA_SAFE_CALL(
        cudaMemcpy( this->_deviceInputRanges, this->_hostInputRanges.data(), this->_hostInputRanges.size() * sizeof( uint64_t ), cudaMemcpyHostToDevice ) );

    // Initialization of all tracepoints
    for ( uint64_t streamId = 0; streamId < this->_numberOfStreams; streamId++ )
    {
        Tracepoints* streamTracePoints          = this->_hostTracePoints + ( streamId * this->_tracePointBlockChunkSize );
        tracepoint_int* streamTracePointVectors = this->_deviceTracePointVectors + ( streamId * this->_tracePointVectorBlockChunkSize );

        for ( uint64_t blockId = 0; blockId < this->_numberOfBlocks; blockId++ )
        {
            Tracepoints* blockTracePoints      = streamTracePoints + ( blockId * this->_tracePointChunkSize );
            tracepoint_int* aTracePointVectors = streamTracePointVectors + ( blockId * this->_tracePointVectorChunkSize );
            tracepoint_int* bTracePointVectors = aTracePointVectors + ( this->_tracePointVectorSize * this->_maxInputsPerBlock );

            for ( uint64_t inputId = 0; inputId < this->_maxInputsPerBlock; inputId++ )
            {
                // A and B
                blockTracePoints[ inputId * 2 ].tracepoints     = aTracePointVectors + inputId * this->_tracePointVectorSize;
                blockTracePoints[ inputId * 2 + 1 ].tracepoints = bTracePointVectors + inputId * this->_tracePointVectorSize;
            }
        }
    }

#if !defined( MAPPED_MEMORY ) && defined( DEVICE_HOST_MEMORY )
    CUDA_SAFE_CALL(
        cudaMemcpy( this->_deviceTracePoints, this->_hostTracePoints, this->_tracePointTotalSize * sizeof( Tracepoints ), cudaMemcpyHostToDevice ) );
#endif
    CUDA_SAFE_CALL( cudaStreamSynchronize( nullptr ) );
}

uint64_t CudaLocalAlignmentStreamManager::queueJob( uint8_t streamId, uint64_t firstInputRange, uint64_t workItemIndex )
{

    cudaStream_t stream       = this->_streams[ streamId ];
    uint32_t indicesToProcess = this->_numberOfBlocks;
    // Check if there is enough ranges to fill all blocks, otherwise it requests a smaller number of ranges to be processed
    if ( this->_hostInputRanges.size() - 1 - firstInputRange < indicesToProcess )
    {
        indicesToProcess = this->_hostInputRanges.size() - 1 - firstInputRange;
    }
    auto preliminaryTracepointsSMMemory = this->_deviceMemoryManager->getPreliminaryTracepointsSMMemory();
    this->_workItems[ workItemIndex ]   = { indicesToProcess, streamId, firstInputRange, this };

    // Fixing the memory region based on the current stream.
    auto streamDeviceMemory = this->_deviceTracePoints + streamId * this->_tracePointBlockChunkSize;
    CudaTracePoints::computeStreamedTracepointsBatch<<<this->_numberOfBlocks, BLOCK_SIZE, 0, stream>>>( this->_deviceInputs,
                                                                                                        this->_deviceInputRanges,
                                                                                                        firstInputRange,
                                                                                                        indicesToProcess,
                                                                                                        this->_tracePointChunkSize,

                                                                                                        streamDeviceMemory,
                                                                                                        preliminaryTracepointsSMMemory.preliminaryTracepoints,
                                                                                                        preliminaryTracepointsSMMemory.locks,
                                                                                                        preliminaryTracepointsSMMemory.perBlockSize );
    CUDA_CHECK_ERROR();

#if !defined( MAPPED_MEMORY ) && defined( DEVICE_HOST_MEMORY )
    CUDA_SAFE_CALL( cudaMemcpyAsync( this->_hostTracePoints + streamId * this->_tracePointBlockChunkSize,
                                     streamDeviceMemory,
                                     this->_tracePointBlockChunkSize * sizeof( Tracepoints ),
                                     cudaMemcpyDeviceToHost,
                                     stream ) );
    CUDA_SAFE_CALL( cudaMemcpyAsync( this->_hostTracePointVectors + streamId * this->_tracePointVectorBlockChunkSize,
                                     this->_deviceTracePointVectors + streamId * this->_tracePointVectorBlockChunkSize,
                                     this->_tracePointVectorBlockChunkSize * sizeof( tracepoint_int ),
                                     cudaMemcpyDeviceToHost,
                                     stream ) );
#endif
    // Using an inline lambda function since without capturing any context since it is not allowed as C function pointer parameter.
    CUDA_SAFE_CALL( cudaLaunchHostFunc(
        stream,
        []( void* data ) {
            auto item = static_cast<StreamWorkItem*>( data );
            item->manager->kernelFinished( item );
        },
        this->_workItems + workItemIndex ) );

    return indicesToProcess;
}

bool CudaLocalAlignmentStreamManager::filterTracepoints( Tracepoints* tracepoints, marvl_float_t errorCorrelation )
{
    if ( tracepoints->tracepointsLength <= 2 )
    {
        return false;
    }

    return (marvl_float_t)tracepoints->differences / ( marvl_float_t )( tracepoints->aEndIndex - tracepoints->aStartIndex ) <= 1. - errorCorrelation;
}

void CudaLocalAlignmentStreamManager::kernelFinished( StreamWorkItem* item )
{

    /* TODO:
     * investigate the need of this lock, It may be removed acording to:
     * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g05841eaa5f90f27124241baafb3e856f
     */
    std::unique_lock<std::mutex> lock( this->_mutex );
    _condition_variable.wait( lock, [] { return true; } );

    uint8_t streamId                     = item->streamId;
    uint64_t firstWorkItem               = item->firstWorkItem;
    Tracepoints* hostTracepoints         = this->_hostTracePoints + streamId * this->_tracePointBlockChunkSize;
    tracepoint_int* hostTracepointVector = this->_hostTracePointVectors + streamId * this->_tracePointVectorBlockChunkSize;

    for ( uint32_t blockId = 0; blockId < item->indicesToProcess; blockId++ )
    {

        Tracepoints* blockHostTracepoints = hostTracepoints + blockId * this->_tracePointChunkSize;

        tracepoint_int* aTracePointVectors = hostTracepointVector + ( blockId * this->_tracePointVectorChunkSize );
        tracepoint_int* bTracePointVectors = aTracePointVectors + ( this->_tracePointVectorSize * this->_maxInputsPerBlock );

        tracepoint_int* aTracePointVectorsMemoryTop = aTracePointVectors;
        tracepoint_int* bTracePointVectorsMemoryTop = bTracePointVectors;

        size_t baseIndex = this->_hostInputRanges[ blockId + firstWorkItem ];
        for ( uint32_t index = baseIndex; index < this->_hostInputRanges[ blockId + firstWorkItem + 1 ]; index++ )
        {
            Tracepoints* aPath;
            Tracepoints* bPath;
            uint32_t zeroBaseIndex = index - baseIndex;

            aPath = blockHostTracepoints + zeroBaseIndex * 2;
            bPath = blockHostTracepoints + zeroBaseIndex * 2 + 1;

            // tracepointsLength > 2 allows to discard simple tracepoint aligments, what is a aligment of a single bad region
            //  &&
            if ( !aPath->skipped )
            {
                aPath->tracepoints = aTracePointVectorsMemoryTop;
                aTracePointVectorsMemoryTop += aPath->tracepointsLength;
                if ( filterTracepoints( aPath, this->_hostInputs[ index ].errorCorrelation ) )
                {

#ifdef ENABLE_SANITY_CHECK
                    if ( this->sanityCheck( index, aPath, false, aTracePointVectors, 1.0 - this->_hostInputs[ index ].errorCorrelation ) )
                    {
#endif
                        if ( this->_callback )
                        {
                            this->_callback( aPath, index, false, this );
                        }
#ifdef ENABLE_SANITY_CHECK
                    }
#endif
                }
            }

            // tracepointsLength > 2 allows to discard simple tracepoint aligments, what is a aligment of a single bad region
            // &&            (  )
            if ( !bPath->skipped )
            {
                bPath->tracepoints = bTracePointVectorsMemoryTop;
                bTracePointVectorsMemoryTop += bPath->tracepointsLength;
                if ( filterTracepoints( bPath, this->_hostInputs[ index ].errorCorrelation ) )
                {
#ifdef ENABLE_SANITY_CHECK
                    if ( this->sanityCheck( index, bPath, true, bTracePointVectors, 1.0 - this->_hostInputs[ index ].errorCorrelation ) )
                    {
#endif
                        if ( this->_callback )
                        {
                            this->_callback( bPath, index, true, this );
                        }
#ifdef ENABLE_SANITY_CHECK
                    }
#endif
                }
            }
        }
    }

    this->_counter--;
    lock.unlock();
    _condition_variable.notify_all();
}

#ifdef ENABLE_SANITY_CHECK

void CudaLocalAlignmentStreamManager::printPath( Tracepoints* currentPath, uint32_t aRead, uint32_t bRead )
{
    uint32_t sum;
    printf( "-----------------------------\n" );
    printf( "%d x %d\n", aRead, bRead );
    printf( "Path A: %d -> %d [%d]\n", currentPath->aStartIndex, currentPath->aEndIndex, currentPath->aEndIndex - currentPath->aStartIndex );
    printf( "Path B: %d -> %d [%d]\n", currentPath->bStartIndex, currentPath->bEndIndex, currentPath->bEndIndex - currentPath->bStartIndex );
    printf( "Path Diffs: %d\n", currentPath->differences );
    printf( "Path Tracepoint Length: %d\n", currentPath->tracepointsLength );
    printf( "Path: " );
    sum = 0;
    for ( uint32_t i = 0; i < currentPath->tracepointsLength; i += 2 )
    {
        sum += currentPath[ 0 ].tracepoints[ i + 1 ];
        printf( "(% 3d, %3d) ", currentPath->tracepoints[ i ], currentPath->tracepoints[ i + 1 ] );
    }
    printf( " (%d) \n", sum );
    printf( "Flags: %d\n\n", currentPath->skipped );
}

bool CudaLocalAlignmentStreamManager::sanityCheck(
    uint32_t index, Tracepoints* path, bool isBPath, const tracepoint_int* tracepointBlockVector, marvl_float_t errorCorrelation )
{
    bool ok = true;

    if ( path->tracepointsLength + path->tracepoints >= this->_tracePointVectorBlockChunkSize / 2 + tracepointBlockVector )
    {
        printf( "Path %s at index %d overflows the tracepoint vector buffer\n", isBPath ? "B" : "A", index );
        ok = false;
    }

    if ( path->aStartIndex > path->aEndIndex )
    {
        printf( "Path %s at index %d has A Start Index '%d' bigger than A End Index '%d'\n", isBPath ? "B" : "A", index, path->aStartIndex, path->aEndIndex );
        ok = false;
    }

    if ( path->bStartIndex > path->bEndIndex )
    {
        printf( "Path %s at index %d has B Start Index '%d' bigger than B End Index '%d'\n", isBPath ? "B" : "A", index, path->bStartIndex, path->bEndIndex );
        ok = false;
    }

    SequenceInfo aSequence;
    SequenceInfo bSequence;
    LocalAlignmentInput* input = this->_hostInputs + index;
    if ( isBPath )
    {
        aSequence = input->bSequence;
        bSequence = input->aSequence;
    }
    else
    {
        aSequence = input->aSequence;
        bSequence = input->bSequence;
    }

    if ( path->aStartIndex > aSequence.sequenceLength )
    {
        printf( "Path %s at index %d has A Start Index '%d' bigger than Sequence Length '%d'\n",
                isBPath ? "B" : "A",
                index,
                path->aStartIndex,
                aSequence.sequenceLength );
        ok = false;
    }

    if ( path->aEndIndex > aSequence.sequenceLength )
    {
        printf( "Path %s at index %d has A End Index '%d' bigger than Sequence Length '%d'\n",
                isBPath ? "B" : "A",
                index,
                path->aEndIndex,
                aSequence.sequenceLength );
        ok = false;
    }

    if ( path->bStartIndex > bSequence.sequenceLength )
    {
        printf( "Path %s at index %d has B Start Index '%d' bigger than Sequence Length '%d'\n",
                isBPath ? "B" : "A",
                index,
                path->bStartIndex,
                bSequence.sequenceLength );
        ok = false;
    }

    if ( path->bEndIndex > bSequence.sequenceLength )
    {
        printf( "Path %s at index %d has B End Index '%d' bigger than Sequence Length '%d'\n",
                isBPath ? "B" : "A",
                index,
                path->bEndIndex,
                bSequence.sequenceLength );
        ok = false;
    }
    uint32_t alignmentLength = 0;
    if ( path->tracepointsLength == 0 )
    {
        printf( "Path %s at index %d has %d tracepoints.\n", isBPath ? "B" : "A", index, path->tracepointsLength );
        ok = false;
    }

    for ( uint32_t i = 0; i < path->tracepointsLength / LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT; i++ )
    {
        alignmentLength += path->tracepoints[ i * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT + 1 ];

        if ( i >= ERROR_RATE_WINDOW_SIZE - 1 )
        {
            marvl_float_t bSteps       = 0;
            marvl_float_t editDistance = 0;
            for ( int j = i; j > i - ERROR_RATE_WINDOW_SIZE; j-- )
            {

                bSteps += static_cast<marvl_float_t>( path->tracepoints[ j * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT + 1 ] );
                editDistance = static_cast<marvl_float_t>( path->tracepoints[ j * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT ] );
            }

            if ( editDistance / bSteps > ERROR_RATE_WINDOW_SIZE * errorCorrelation )
            {
                printf( "Path %s at index %d has tracepoint %d leads to a bigger window error rate then expected. EditDistance: %d, Steps in B: %d -> %d / %d "
                        "= %f > %f\n",
                        isBPath ? "B" : "A",
                        index,
                        i,
                        path->tracepoints[ i * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT ],
                        path->tracepoints[ i * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT + 1 ],
                        path->tracepoints[ i * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT ],
                        path->tracepoints[ i * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT + 1 ],
                        editDistance / bSteps,
                        ERROR_RATE_WINDOW_SIZE * errorCorrelation );
                ok = false;
            }
        }
#ifdef DEBUG_FUNNY
        if ( path->tracepoints[ i * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT + 1 ] > 180 ||
             path->tracepoints[ i * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT ] > 180 )
        {
            printf( "Path %s at index %d (%lu x %lu) has funny tracepoint at %d: ( %d , %d).\n",
                    isBPath ? "B" : "A",
                    index,
                    this->_hostInputs[ index ].pair.ReadIDs.aRead + this->_aReadsOffset,
                    this->_hostInputs[ index ].pair.ReadIDs.bRead + this->_bReadsOffset,
                    i,
                    path->tracepoints[ i * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT ],
                    path->tracepoints[ i * LOCAL_ALIGNMENT_TRACE_ELEMENTS_PER_TRACEPOINT + 1 ] );
            // ok = false;
        }
#endif
    }
    if ( path->bStartIndex + alignmentLength != path->bEndIndex )
    {
        printf( "Path %s at index %d () has expected B End Index '%d' but tracepoints ending at '%d'\n",
                isBPath ? "B" : "A",
                index,
                path->bEndIndex,
                path->bStartIndex + alignmentLength );
        ok = false;
    }
    if ( !ok )
    {
        printPath( path, input->pair.ReadIDs.aRead + this->_aReadsOffset, input->pair.ReadIDs.bRead + this->_bReadsOffset );
#ifdef DUMP_ERROR_FASTA

        char fileName[ 1024 ];
        sprintf( fileName,
                 "errors.%lux%lu-%s.fasta",
                 input->pair.ReadIDs.aRead + this->_aReadsOffset,
                 input->pair.ReadIDs.bRead + this->_bReadsOffset,
                 isBPath ? "B" : "A" );
        FILE* error = fopen( fileName, "w" );

        char* buffer = static_cast<char*>( malloc( sizeof( char ) * input->aSequence.sequenceLength + 1 ) );
        memcpy( buffer, input->aSequence.hostSequence, input->aSequence.sequenceLength );
        buffer[ input->aSequence.sequenceLength ] = 4;
        Lower_Read( buffer );

        fprintf( error, ">%d\n", input->pair.ReadIDs.aRead );
        fprintf( error, "%s\n", buffer );
        free( buffer );

        buffer = static_cast<char*>( malloc( sizeof( char ) * input->bSequence.sequenceLength + 1 ) );
        memcpy( buffer, input->bSequence.hostSequence, input->bSequence.sequenceLength );
        buffer[ input->bSequence.sequenceLength ] = 4;
        Lower_Read( buffer );

        fprintf( error, ">%d\n", input->pair.ReadIDs.bRead );
        fprintf( error, "%s\n", buffer );
        free( buffer );

        fclose( error );
#else
        exit( 1 );
#endif
    }

    return ok;
}
#endif
CudaLocalAlignmentStreamManager::~CudaLocalAlignmentStreamManager()
{

    for ( int i = 0; i < this->_numberOfStreams; i++ )
    {
        cudaStreamDestroy( this->_streams[ i ] );
    }

    free( this->_streams );

    if ( this->_workItems )
    {
        free( this->_workItems );
    }

    CUDA_SAFE_CALL( cudaStreamSynchronize( nullptr ) );
}

void CudaLocalAlignmentStreamManager::createStreams()
{
    this->_streams = (cudaStream_t*)malloc( sizeof( cudaStream_t ) * this->_numberOfStreams );
    for ( size_t i = 0; i < this->_numberOfStreams; i++ )
    {
        CUDA_SAFE_CALL( cudaStreamCreate( this->_streams + i ) );
    }
}

void CudaLocalAlignmentStreamManager::queueJobsAndWait( const LocalAligmentCallback& callback )
{
    if ( !this->_numberOfInputs )
    {
        return;
    }

    if ( callback )
    {
        this->_callback = callback;
    }

    if ( this->_workItems )
    {
        free( this->_workItems );
    }
    this->_workItems = (StreamWorkItem*)malloc( INT_DIV_CEIL( this->_hostInputRanges.size() - 1, this->_numberOfBlocks ) * sizeof( StreamWorkItem ) );

    this->_counter = INT_DIV_CEIL( this->_hostInputRanges.size() - 1, this->_numberOfBlocks );

    // this->_hostInputRanges.size() - 1: -1 is needed becouse this is an array of ranges
    // and last index is for closing the last range
    for ( uint64_t firstWorkItem = 0, streamId = 0, workItem = 0; firstWorkItem < this->_hostInputRanges.size() - 1; )
    {
        firstWorkItem += this->queueJob( streamId, firstWorkItem, workItem );
        streamId = ( streamId + 1 ) % this->_numberOfStreams;
        workItem++;
    }

    this->waitJobs();
    this->_deviceMemoryManager->freeWorkMemory( this->_workMemory );
}
void CudaLocalAlignmentStreamManager::waitJobs()
{
    for ( size_t i = 0; i < this->_numberOfStreams; i++ )
    {
        CUDA_SAFE_CALL( cudaStreamSynchronize( this->_streams[ i ] ) );
        CUDA_CHECK_ERROR();
    }
    /* TODO: Investigate the need of this lock.
     * It is intended to prevent the code to continue before the cudaLaunchHostFunc calls finishes.
     * It is unclear if cudaStreamSynchronize waits for those call since the stream becomes inactive
     * if there is no kernel calls after those.
     */
    std::unique_lock<std::mutex> lock( this->_mutex );
    _condition_variable.wait( lock, [ & ] { return _counter == 0; } );
}
size_t CudaLocalAlignmentStreamManager::getNumberOfBlocks() const { return _numberOfBlocks; }
LocalAlignmentInput* CudaLocalAlignmentStreamManager::getHostInputs() const { return _hostInputs; }
size_t CudaLocalAlignmentStreamManager::getNumberOfInputs() const { return _numberOfInputs; }
size_t CudaLocalAlignmentStreamManager::getAReadsOffset() const { return _aReadsOffset; }
size_t CudaLocalAlignmentStreamManager::getBReadsOffset() const { return _bReadsOffset; }

} // namespace CudaStreamInterface
