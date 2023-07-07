
#pragma once

#include "tracepoints/definitions.h"
#include <condition_variable>
#include <cstdint>
#include <cuda_runtime.h>
#include <mutex>
#include <vector>

#ifndef TOTAL_MEMORY_IN_GB
#define TOTAL_MEMORY_IN_GB 4
#endif

/**
 * This define tells the code that this resource manager is using mapped memory
 * \note
 *  Not using mapped memory for this implementation needs careful rethinking and refactor.
 *  The two issues about using not mapped device memory are:
 *  1. Not enough available memory for holding the inputs and outputs and do the traceback
 *  2. Performance
 */
#define MAPPED_MEMORY 1

extern "C"
{
#include "db/DB.h"
}

/**
 * This struct holds the information about the allocated memory region on the Device and Host to be used
 */
typedef struct WorkMemory
{
    /**
     * Index of the struct. Used to distinguish multiple memory regions
     */
    int index;
    /**
     * The size of the allocated memory region
     */
    size_t size;
    /**
     * Index of the struct. points to the host memory region. This memory region should be used a 'mirror'
     * of the device memory region on WorkMemory::device
     */
    void* host;
    /**
     * points to the device memory region. This memory region should be used a 'mirror'
     * of the host memory region on WorkMemory::host
     */
    void* device;

} WorkMemory;

/**
 * This struct holds a information of the per SM allocated memory region. The memory region is used by the kernel blocks and
 * each block selects a memory segment based on the SM it is running in. Having multiple blocks running in a single SM is handled
 * by locks to allow each block to pick a distinct memory region.
 */
typedef struct PreliminaryTracepointSMMemory
{
    /** Hold the locks for each memory segment assign to each SM. This is a bitwise when a free segment has a bit set to 1.
     * \note
     * might want to use 128 byte type to avoid cacheline ping-pong between SMs
     */
    uint32_t* locks = nullptr;
    /**
     * Pointer to the whole allocated memory region on the device
     */
    preliminary_Tracepoint* preliminaryTracepoints = nullptr;
    /**
     * The amount of memory needed per block
     */
    size_t perBlockSize = 0;
    /**
     * The total size of the allocated memory region
     */
    size_t totalSize = 0;
    /**
     * The max SM id given by the GPU.
     * /note A GPU could have disabled SM, which means, e.g., if a GPU has 80 active
     * it could have some inactive ones being maxSMid > 80
     *
     */
    uint32_t maxSMid = 0;
} PreliminaryTracepointSMMemory;

/**
 * This class handles resource allocation. Resources being computational and memory resources
 * in the CPU and GPU.
 *
 * All memory allocation done in the GPU is done thru this class and reused/recycled
 * during the application.
 *
 * For the computational resources, this class implements locks to protect some high demanding
 * code region to prevent more threads than allowed to enter the protected code region
 *
 * /note
 *  Since this class controls only memory that is used by the GPU, being
 *  it device memory and host memory, the locks for the GPU must guarantee that
 *  there is enough memory available to the GPU to process its alignments
 */
class ResourceManager
{
  private:
    /**
     * Mutex used to protect GPU resource usage
     */
    std::mutex _gpuMutex;
    /**
     * Conditional variable used to lock and unlock based on the GPU resource counter
     */
    std::condition_variable _gpuConditionalVariable;

    /**
     * Mutex used to protect CPU resource usage
     */
    std::mutex _cpuMutex;
    /**
     * Conditional variable used to lock and unlock based on the CPU resource counter
     */
    std::condition_variable _cpuConditionalVariable;

    /**
     * All allocated device memory pointers
     */
    std::vector<void*> deviceMemoryPointers;
    /**
     * All allocated host memory pointers
     */
    std::vector<void*> hostMemoryPointers;

    /**
     * Wrapper function to ::cudaMallocHost that stores the pointer to be freed in ResourceManager::~ResourceManager()
     * @tparam T The type of the pointer
     * @param pointer Pointer to allocated host memory
     * @param size Requested allocation size in bytes
     * @return
     * ::cudaSuccess,
     * ::cudaErrorInvalidValue,
     * ::cudaErrorMemoryAllocation
     *
     */
    template <typename T> cudaError_t hostMalloc( T** pointer, size_t size );
    /**
     * Wrapper function to ::cudaMalloc that stores the pointer to be freed in ResourceManager::~ResourceManager()
     * @tparam T The type of the pointer
     * @param pointer Pointer to allocated device memory
     * @param size Requested allocation size in bytes
     * @return
     * ::cudaSuccess,
     * ::cudaErrorInvalidValue,
     * ::cudaErrorMemoryAllocation
     *
     */
    template <typename T> cudaError_t deviceMalloc( T**, size_t size );

    /**
     * CPU Resource counter. This value is used by ResourceManager::_cpuConditionalVariable
     */
    size_t _cpuConcurrencyCounter;

    /**
     * Total size of memory allocated on the device, and also on the host for moving data
     */
    size_t _workMemorySize;
    /**
     * Total number of concurrency allowed on the GPU
     */
    size_t _gpuConcurrencyCounter;
    /**
     * Max read length that is supported by the memory.
     */
    size_t _maxReadLength;

    /**
     * Array of WorkMemory. Size is given by  ResourceManager::_gpuConcurrencyCounter
     */
    WorkMemory* _workMemoryAreas;
    /**
     * Locks for each WorkMemory in ResourceManager::_workMemoryAreas. Size is given by  ResourceManager::_gpuConcurrencyCounter
     */
    bool* _workMemoryFree;

    /**
     * Host memory allocated to 'mirror' the device memory in ResourceManager::_hostWorkMemory
     */
    void* _hostWorkMemory;
    /**
     * Device memory allocated to 'mirror' the host memory in ResourceManager::_deviceWorkMemory
     */
    void* _deviceWorkMemory;

    /**
     * Per SM memory information
     */
    PreliminaryTracepointSMMemory _preliminaryTracepointSMMemory{};

  public:
    /**
     * Constructor
     *
     * @param gpuResourceCounter amount of concurrency allowed to the GPU
     * @param cpuResourceCounter amount of concurrency allowed to the CPU
     * @param maxReadSize max read length. This is an estimated value to allow memory allocation. If the actual read length are in the data set is bigger
     *                    than this values, most likely the this is give a out of bounds memory access. This values is constraint by the device available memory
     */
    explicit ResourceManager( size_t gpuResourceCounter = 1, size_t cpuResourceCounter = 1, size_t maxReadSize = 1e6 );
    ~ResourceManager();

    /**
     * Free all memory allocated by this class
     */
    void cleanUp();

    /**
     * Returns ResourceManager::_workMemorySize
     * @return
     *      ResourceManager::_workMemorySize
     */
    size_t getWorkMemorySize() const;

    /**
     * Allocates memory on the device and host to allow the GPU to compute its alignments and output results
     * and also allocates the memory 'attached' to each SM
     * @param totalMemory The amount of memory dedicated to computing alignments.
     *
     * \note
     * Current implemnetation uses mapped memory, which means that this allcation size is only limited by
     * host memory
     */
    void allocWorkMemory( size_t totalMemory = TOTAL_MEMORY_IN_GB * 1024UL * 1024UL * 1024UL );

    /**
     * Copies a block into the device to be used by the alignment code
     * @param block The block to be copied to the device
     * @return
     *      The block information with device pointer
     */
    SequenceInfo * copyBlock2Device( HITS_DB* block );

    /**
     * Returns a memory region for doing the alignment. If there is no memory region available, it locks until one is free.
     * @return
     *      The information about the memory region
     */
    WorkMemory* getWorkMemory();

    /**
     * It releases a work memory regition to be picked up by the next thread. This does not frees memory but just flags
     * it as free to be used by the next thread.
     * @param workMemory
     */
    void freeWorkMemory( const WorkMemory* workMemory );

    /**
     * Locks CPU resources. This is used to prevent to threads to enter a high demanding code region
     */
    void lockCpuResources();
    /**
     * Unlock CPU resources
     */
    void unlockCpuResources();

    /**
     * Returns ResourceManager::_maxReadLength
     * @return
     *      ResourceManager::_maxReadLength
     */
    size_t getMaxReadLength() const;

    /**
     * Return the per SM memory information
     * @return
     *      ResourceManager::_preliminaryTracepointSMMemory
     */
    PreliminaryTracepointSMMemory getPreliminaryTracepointsSMMemory();
};