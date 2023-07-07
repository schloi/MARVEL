
#include "cuda/utils.cuh"
#include "resource-manager.h"
#include <algorithm>

SequenceInfo* ResourceManager::copyBlock2Device( HITS_DB* block )
{

    char* sequences = (char*)block->bases;

    char* deviceSequences;

    const size_t database_padding   = 4 - 1; // padding needed by database. Padding of 4 (DB.c:1397) and -1 because of the pointer adjustment (DB.c:1404)
    const size_t cuda_front_padding = 16;    // allow the CUDA reverse sequence reader to read up to 5 bytes before sequence start, rounded up to multiple of 16
                                             // for potentially better memcpy performance
    const size_t cuda_back_padding = 8;      // allow the CUDA reverse sequence reader to read up to 8 bytes after sequence end

    const size_t total_padding = database_padding + cuda_front_padding + cuda_back_padding;
    size_t size                = ( block->totlen + block->nreads + total_padding );
    CUDA_SAFE_CALL( deviceMalloc( &deviceSequences, size * sizeof( char ) ) );
    deviceSequences += cuda_front_padding; // Adjusting the pointer to consider the initial cuda padding
    CUDA_SAFE_CALL( cudaMemcpy( deviceSequences, sequences, ( size - cuda_front_padding - cuda_back_padding ) * sizeof( char ), cudaMemcpyHostToDevice ) );

    SequenceInfo* hostSequeneInfo;
    CUDA_SAFE_CALL( hostMalloc( &hostSequeneInfo, block->nreads * sizeof( SequenceInfo ) ) );

    for ( int i = 0; i < block->nreads; i++ )
    {

        hostSequeneInfo[ i ].sequenceLength = block->reads[ i ].rlen;
        hostSequeneInfo[ i ].deviceSequence = deviceSequences + block->reads[ i ].boff;
        hostSequeneInfo[ i ].hostSequence   = sequences + block->reads[ i ].boff;
#ifdef DUMP_ERROR_FASTA
        hostSequeneInfo[ i ].hostSequence = sequences + block->reads[ i ].boff;
#endif
    }

#ifdef DEBUG
    printf( "Device\n\tBases: %p -> %p [%lld]\n", deviceSequences, deviceSequences + block->totlen, block->totlen );
#endif

    return hostSequeneInfo;
}
/**
 * https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nsmid
 * @param numSMIDs memory pointer where to save the result values
 */
__global__ void getNumSMIDs( unsigned int* numSMIDs )
{
    unsigned int nsmid;
    asm( "mov.u32 %0, %%nsmid;" : "=r"( nsmid ) );
    *numSMIDs = nsmid;
}

PreliminaryTracepointSMMemory ResourceManager::getPreliminaryTracepointsSMMemory()
{

    if ( _preliminaryTracepointSMMemory.locks == nullptr )
    {
        uint32_t* result;
        CUDA_SAFE_CALL( cudaMalloc( &result, sizeof( *result ) ) );

        getNumSMIDs<<<1, 1>>>( result );
        CUDA_CHECK_ERROR();
        CUDA_SAFE_CALL( cudaDeviceSynchronize() );
        uint32_t numSMIDs = 0;
        CUDA_SAFE_CALL( cudaMemcpy( &numSMIDs, result, sizeof( *result ), cudaMemcpyDeviceToHost ) );

        _preliminaryTracepointSMMemory.maxSMid = numSMIDs;
#ifdef DEBUG
        printf( " maxSMid = %d\n", numSMIDs );
#endif
        size_t memoryPerBlock  = TRACE_POINT_VECTOR_SIZE( this->_maxReadLength );
        size_t totalMemorySize = memoryPerBlock * BLOCKS_PER_SM * numSMIDs;

        printf( "\nAllocating %lu MB in the device for traceback\n\n",
                totalMemorySize * sizeof( *( _preliminaryTracepointSMMemory.preliminaryTracepoints ) ) / 1024 / 1024 );

        CUDA_SAFE_CALL( deviceMalloc( &( _preliminaryTracepointSMMemory.locks ), sizeof( _preliminaryTracepointSMMemory.locks ) * numSMIDs ) );
        CUDA_SAFE_CALL( cudaMemset( _preliminaryTracepointSMMemory.locks, 0xff, sizeof( _preliminaryTracepointSMMemory.locks ) * numSMIDs ) );
        CUDA_SAFE_CALL( deviceMalloc( &( _preliminaryTracepointSMMemory.preliminaryTracepoints ),
                                      sizeof( *( _preliminaryTracepointSMMemory.preliminaryTracepoints ) ) * totalMemorySize ) );
        _preliminaryTracepointSMMemory.perBlockSize = memoryPerBlock;
        _preliminaryTracepointSMMemory.totalSize    = totalMemorySize;
        _preliminaryTracepointSMMemory.maxSMid      = numSMIDs;
    }
    return _preliminaryTracepointSMMemory;
}

ResourceManager::ResourceManager( size_t gpuResourceCounter, size_t cpuResourceCounter, size_t maxReadSize ) :
    _cpuConcurrencyCounter( cpuResourceCounter ), _workMemorySize( 0 ), _gpuConcurrencyCounter( gpuResourceCounter ), _hostWorkMemory( nullptr ),
    _deviceWorkMemory( nullptr ), _maxReadLength( maxReadSize )
{

    this->_workMemoryAreas = static_cast<WorkMemory*>( malloc( sizeof( WorkMemory ) * this->_gpuConcurrencyCounter ) );
    this->_workMemoryFree  = static_cast<bool*>( malloc( sizeof( bool ) * this->_gpuConcurrencyCounter ) );
}

ResourceManager::~ResourceManager() { cleanUp(); }

void ResourceManager::cleanUp()
{
    std::for_each( deviceMemoryPointers.begin(), deviceMemoryPointers.end(), []( void* pointer ) { cudaFree( pointer ); } );
    deviceMemoryPointers.clear();

    std::for_each( hostMemoryPointers.begin(), hostMemoryPointers.end(), []( void* pointer ) { cudaFreeHost( pointer ); } );
    hostMemoryPointers.clear();

    CUDA_SAFE_CALL( cudaFreeHost( _hostWorkMemory ) );
    //    CUDA_SAFE_CALL( cudaFree( _deviceWorkMemory ) );
    free( this->_workMemoryAreas );
    free( this->_workMemoryFree );

    CUDA_SAFE_CALL( cudaFree( _preliminaryTracepointSMMemory.locks ) );
    CUDA_SAFE_CALL( cudaFree( _preliminaryTracepointSMMemory.preliminaryTracepoints ) );
    _preliminaryTracepointSMMemory = PreliminaryTracepointSMMemory{};

    _workMemorySize = 0;
}

template <typename T> cudaError_t ResourceManager::deviceMalloc( T** pointer, size_t size )
{
    cudaError_t error = cudaMalloc( pointer, size );
    deviceMemoryPointers.push_back( *pointer );
    return error;
}

template <typename T> cudaError_t ResourceManager::hostMalloc( T** pointer, size_t size )
{
    cudaError_t error = cudaMallocHost( pointer, size );
    hostMemoryPointers.push_back( *pointer );
    return error;
}
void ResourceManager::allocWorkMemory( size_t totalMemory )
{
    _workMemorySize = totalMemory;
    CUDA_SAFE_CALL( cudaHostAlloc( &_hostWorkMemory, _workMemorySize, cudaHostAllocMapped ) );
    CUDA_SAFE_CALL( cudaHostGetDevicePointer( &_deviceWorkMemory, _hostWorkMemory, 0 ) );

    for ( int i = 0; i < this->_gpuConcurrencyCounter; i++ )
    {
        _workMemoryAreas[ i ] = { .index  = 1,
                                  .size   = getWorkMemorySize(),
                                  .host   = static_cast<char*>( _hostWorkMemory ) + i * getWorkMemorySize(),
                                  .device = static_cast<char*>( _deviceWorkMemory ) + i * getWorkMemorySize() };
        _workMemoryFree[ i ]  = true;
    }

    this->getPreliminaryTracepointsSMMemory();
}

size_t ResourceManager::getWorkMemorySize() const { return _workMemorySize / _gpuConcurrencyCounter; }

WorkMemory* ResourceManager::getWorkMemory()
{
    std::unique_lock<std::mutex> look( _gpuMutex );
    _gpuConditionalVariable.wait( look, [ & ] {
        return std::any_of( this->_workMemoryFree, this->_workMemoryFree + this->_gpuConcurrencyCounter, []( bool value ) { return value; } );
    } );
    WorkMemory* memory = nullptr;
    for ( int i = 0; i < this->_gpuConcurrencyCounter; i++ )
    {
        if ( _workMemoryFree[ i ] )
        {
            _workMemoryFree[ i ] = false;
            memory               = _workMemoryAreas + i;
            break;
        }
    }
    look.unlock();
    return memory;
}
void ResourceManager::freeWorkMemory( const WorkMemory* workMemory )
{
    size_t index             = this->_workMemoryAreas - workMemory;
    _workMemoryFree[ index ] = true;
    _gpuConditionalVariable.notify_one();
}
void ResourceManager::lockCpuResources()
{
    std::unique_lock<std::mutex> look( _cpuMutex );
    _cpuConditionalVariable.wait( look, [ & ] { return _cpuConcurrencyCounter > 0; } );
    _cpuConcurrencyCounter--;
    look.unlock();
}
void ResourceManager::unlockCpuResources()
{
    _cpuConcurrencyCounter++;
    _cpuConditionalVariable.notify_one();
}
size_t ResourceManager::getMaxReadLength() const { return _maxReadLength; }
