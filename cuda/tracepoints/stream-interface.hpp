#pragma once

#include "definitions.h"
#include <condition_variable>
#include <cstdint>
#include <cuda/resource-manager.h>
#include <cuda_runtime.h>
#include <functional>
#include <mutex>
#include <vector>

namespace CudaStreamInterface
{
class CudaLocalAlignmentStreamManager;
/**
 * Call back defition for the aligment
 * @param path the tracepoint output from the alignment
 * @param index the input index which generated this result
 * @param isBPath this is set to true is the result refers to BxA and false for AxB
 * @param self A pointer to the class calling the call back
 *
 * \note
 *
 * Any implementation of a call back function using this definition should follow the restriction of
 * Nvidia regarding CUDA API calls inside a ::cudaLaunchHostFunc. This also applies to the callback
 */
typedef std::function<void( const Tracepoints* path, const size_t& index, const bool isBPath, const CudaLocalAlignmentStreamManager* self )>
    LocalAligmentCallback;

/**
 * This class manages the GPU work load and returns its results
 * /note
 * No memory allocation is done in this class, and it uses only memory allocated by the resource manager
 */
class CudaLocalAlignmentStreamManager
{
  private:
    /**
     * This struct is used to pass information to the cuda host function calls. This is colled as defined by
     *  a set of inputs that are given to a kernel to process.
     */
    typedef struct
    {
        /**
         * How many indices in the input list should be processed
         */
        uint32_t indicesToProcess;
        /**
         * Stream id which processed the kernel call
         */
        uint8_t streamId;
        /**
         * First item in the input list that was processed by the kernel call
         */
        uint64_t firstWorkItem;
        /**
         * A self reference to the this class
         */
        CudaLocalAlignmentStreamManager* manager;
    } StreamWorkItem;

    /**
     * The amount of blocks that will be triggered in the kernel call
     */
    size_t _numberOfBlocks;
    /**
     * The A read number offset given by the input A block
     */
    size_t _aReadsOffset;
    /**
     * The B read number offset given by the input B block
     */
    size_t _bReadsOffset;

    /**
     * Array of streams. Size is set in CudaLocalAlignmentStreamManager::_numberOfStreams
     */
    cudaStream_t* _streams{};
    /**
     * The amount of stream handling the kernel calls
     */
    size_t _numberOfStreams;

    /**
     * The needed size to hold the tracepoint output array of a single alignment. This size is computed using #TRACE_POINT_VECTOR_SIZE
     * which translates to: ceil(<the max sequence length> / <trace space:100>) + 3 * 2
     *  - ceil(<the max sequence length> / <trace space:100>) gives the amount of tracepoints an alignment would have
     *  - +3: Since the sequence length usually is not a multiple of the trace space, we need 1 extra tracepoint in the beginning and 1 at the end
     *      considering that we have reverse and forward wave, we need these 2 extra for each wave. But one can from the reverse wave can be reused
     *      by the forward wave, which gives +3
     *  - * 2: Each tracepoint is a pair of values.
     */
    size_t _tracePointVectorSize;
    /**
     * The needed size to hold the tracepoint output array of all alignments computed by a single block.
     */
    size_t _tracePointVectorChunkSize{};
    /**
     * The needed size to hold the tracepoint output array of all alignments computed by a all block.
     */
    size_t _tracePointVectorBlockChunkSize{};
    /**
     * The needed size to hold the tracepoint output array of all alignments computed by a all block of all streams.
     */
    size_t _tracePointVectorTotalSize;

    /**
     * Pointer to the memory region allocated for the tracepoint output array in the host
     */
    tracepoint_int* _hostTracePointVectors{};
    /**
     * Pointer to the memory region allocated for the tracepoint output array in the device
     */
    tracepoint_int* _deviceTracePointVectors{};

    /**
     * The needed size to hold the tracepoint output structure of all alignments computed by a single block.
     */
    size_t _tracePointChunkSize{};
    /**
     * The needed size to hold the tracepoint output structure of all alignments computed by a all block.
     */
    size_t _tracePointBlockChunkSize{};
    /**
     * The needed size to hold the tracepoint output structure of all alignments computed by a all block of all streams.
     */
    size_t _tracePointTotalSize;
    /**
     * Pointer to the memory region allocated for the tracepoint output structure in the host
     */
    Tracepoints* _hostTracePoints{};
    /**
     * Pointer to the memory region allocated for the tracepoint output structure in the device
     */
    Tracepoints* _deviceTracePoints{};

    /**
     * The length of the longest sequence in the block
     */
    size_t _maxSequenceLength;

    /**
     * Array of inputs in the device memory where the input information should be read from. Size is given by CudaLocalAlignmentStreamManager::_numberOfInputs
     */
    LocalAlignmentInput* _deviceInputs{};
    /**
     * Array of inputs in  the host memory where the input information should be read from. Size is given by CudaLocalAlignmentStreamManager::_numberOfInputs
     */
    LocalAlignmentInput* _hostInputs;
    /**
     * The size of the input arrays.
     */
    size_t _numberOfInputs;
    /**
     * The vector holds the input ranges that will be distributed to each block. All ranges are given as:
     * _hostInputRanges[i] <= range < _hostInputRanges[i + 1]
     */
    std::vector<uint64_t> _hostInputRanges;
    /**
     * A device copy of  CudaLocalAlignmentStreamManager::_hostInputRanges
     *
     * \see CudaLocalAlignmentStreamManager::_hostInputRanges
     */
    uint64_t* _deviceInputRanges{};
    /**
     * How many inputs each block can compute
     */
    size_t _maxInputsPerBlock{};

    /**
     * The memory used to manage input information from host to the device and output from the device to the host
     */
    WorkMemory* _workMemory = nullptr;

    /**
     * Reference to the callback set in CudaLocalAlignmentStreamManager:queueJobsAndWait.
     * This callback will be called after each computed alignment after the kernel finishes
     */
    LocalAligmentCallback _callback;

    /**
     * Array of work items. An work item is defined by a set of inputs that are given to a kernel to process
     */
    StreamWorkItem* _workItems{};

    /**
     * Mutex lock around the kernelFinish function.
     * It may be removed see:
     * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g05841eaa5f90f27124241baafb3e856f
     */
    std::mutex _mutex;
    /**
     * Conditional variable around the kernelFinish and waitJobs functions
     * It prevents exiting waitJobs before all kernelFinish calls are done
     */
    std::condition_variable _condition_variable;
    /**
     * The counter of expected kernelFinish calls to be done before returning from waitJobs
     */
    uint32_t _counter{};

    /**
     * Queue an work item in a stream to be processed. It queues either
     * CudaLocalAlignmentStreamManager::_numberOfBlocks input ranges, it enough ranges are available,
     * or it queues only the remaining.
     *
     * @param streamId Which stream will receive the work item
     * @param firstInputRange The first input range which should be queued.
     * @param workItemIndex The work item index of this job. References CudaLocalAlignmentStreamManager::_workItems
     *
     * @returns
     *      The amount of input ranges queued in the call
     */
    uint64_t queueJob( uint8_t streamId, uint64_t firstInputRange, uint64_t workItemIndex );

    /**
     * Wait for all queued kernels
     */
    void waitJobs();
    /**
     * Callback for a kernel call. It is called using ::cudaLaunchHostFunc. This function read all outputs from a kernel call
     * and call CudaLocalAlignmentStreamManager::_callback to report the results to the caller of CudaLocalAlignmentStreamManager::queueJobsAndWait
     *
     * \note
     * This function should follow the restriction of Nvidia regarding CUDA API calls inside a ::cudaLaunchHostFunc. This also applies to the callback
     *
     * \see
     * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g05841eaa5f90f27124241baafb3e856f
     * @param item
     */
    void kernelFinished( StreamWorkItem* item );

    /**
     * Create necessary streams for processing the inputs
     */
    void createStreams();
    /**
     * It slices and distribute memory regions for all blocks and stream and copy any necessary value from host to device.
     * The memory is sliced like described in https://confluence.bix-digital.com/display/MARVL/Work+distribution+of+Alignments+to+Cuda+Blocks
     *
     * \note
     * There is not any memory allocation happening inside this function.
     */
    void initializeMemory();
    /**
     * This rearranges all inputs in order to create the input ranges in CudaLocalAlignmentStreamManager::_hostInputRanges
     */
    void shuffleInputForProcessing();
    /**
     * this function is used to filter out some really bad tracepoints using the following rules:
     * - Overall error rate >= 1.0 - errorCorrelation
     * - A single tracepoint. (this is most likely not triggered anymore, needs double checking)
     *
     * this is/should be used to compensate the 'relaxed' stopping condition on the GPU
     *
     * @param tracepoints
     * @param errorCorrelation
     * @return
     */
    static bool filterTracepoints( Tracepoints* tracepoints, marvl_float_t errorCorrelation );
    /**
     * The resource manager attached to the instance which is responsible of providing allocated memory
     */
    ResourceManager* _deviceMemoryManager;

#ifdef ENABLE_SANITY_CHECK
    /**
     * It check all the tracepoint for their consistency. This code is used to check if the algorithm
     * is behaving correctly and it should not be used in production
     *
     * @param index Input Index of the tracepoint
     * @param path the Tracepoint to be checked
     * @param isBPath true if the result is coming from BxA and false if AxB
     * @param tracepointBlockVector The pointer to the memory region where the tracepoint array is stored. Used to check memory overflow
     * @param errorCorrelation error correlation
     * @return
     * true if the tracepoints are ok, false otherwise
     */
    bool sanityCheck( uint32_t index, Tracepoints* path, bool isBPath, const tracepoint_int* tracepointBlockVector, marvl_float_t errorCorrelation );
    /**
     * This functions prints Tracepoints
     * @param currentPath The tracepoints to be printed
     * @param aRead the A read that produced the tracepoints
     * @param bRead the B read that produced the tracepoints
     */
    static void printPath( Tracepoints* currentPath, uint32_t aRead, uint32_t bRead );
#endif

  public:
    /**
     * Constructor
     *
     * @param inputs Arrays of inputs to be processed
     * @param numberOfInputs number of inputs to be processed
     * @param numberOfBlocks number of GPU blocks that should be used to process all inputs
     * @param numberOfStreams number of GPU streams that should be used to process all inputs
     * @param aReadsOffset the A read number offset
     * @param bReadsOffset the B read number offset
     * @param maxSequenceLength the exact length of the longest read
     * @param resourceManager the resource manager for the instance to get allocated memory from
     *
     */
    CudaLocalAlignmentStreamManager( LocalAlignmentInput* inputs,
                                     size_t numberOfInputs,
                                     size_t numberOfBlocks,
                                     size_t numberOfStreams,
                                     size_t aReadsOffset,
                                     size_t bReadsOffset,
                                     size_t maxSequenceLength,
                                     ResourceManager* resourceManager );

    /**
     * Destructor. It calls CudaLocalAlignmentStreamManager::cleanUp()
     */
    ~CudaLocalAlignmentStreamManager();
    /**
     * Initialized all necessary fields before it can process any input
     * \note
     * It must be called before CudaLocalAlignmentStreamManager::queueJobsAndWait
     */
    void initialize();
    /**
     * Queues all inputs into the GPU streams and waits all inputs to be finished. It does not
     * queue all inputs in a single kernel but it queues subsets of the inputs in separate kernel
     * call rotating the available streams.
     *
     * /note
     * This calls uses CudaLocalAlignmentStreamManager::_resourceManager to lock the GPU resources
     *
     * @param callback the call which will be called after each kernel run
     */
    void queueJobsAndWait( const LocalAligmentCallback& callback );

    /**
     * Returns CudaLocalAlignmentStreamManager::_numberOfBlocks
     * @return
     *      CudaLocalAlignmentStreamManager::_numberOfBlocks
     */
    size_t getNumberOfBlocks() const;

    /**
     * Returns CudaLocalAlignmentStreamManager::_hostInputs
     * @return
     *      CudaLocalAlignmentStreamManager::_hostInputs
     */
    LocalAlignmentInput* getHostInputs() const;
    /**
     * Returns CudaLocalAlignmentStreamManager::_numberOfInputs
     * @return
     *      CudaLocalAlignmentStreamManager::_numberOfInputs
     */
    size_t getNumberOfInputs() const;
    /**
     * Returns CudaLocalAlignmentStreamManager::_aReadsOffset
     * @return
     *      CudaLocalAlignmentStreamManager::_aReadsOffset
     */
    size_t getAReadsOffset() const;
    /**
     * Returns CudaLocalAlignmentStreamManager::_bReadsOffset
     * @return
     *      CudaLocalAlignmentStreamManager::_bReadsOffset
     */
    size_t getBReadsOffset() const;

};

} // namespace CudaStreamInterface
