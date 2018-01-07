/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Paddle/paddle/pserver/ParameterServer2.h
// Paddle/paddle/pserver/ParameterServer2.cpp

#pragma once

#include <algorithm>
#include <fstream>
#include <atomic>
#include <limits>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <stddef.h>
#include <stdlib.h>

#include "platform/paddle_locks.h"
#include "platform/paddle_threadlocal.h"
#include "utils/paddle_matrix.h"
#include "utils/paddle_vector.h"
#include "utils/paddle_parameter.h"
#include "utils/paddle_proto.h"
#include "utils/paddle_pserver_proto_server.h"

DECLARE_int32(port);

DEFINE_int32(pserver_num_threads, 1, "number of threads for sync op exec");
DEFINE_double(async_lagged_ratio_min,
              1.0,
              "control config_.async_lagged_grad_discard_ratio() min value");
DEFINE_double(
    async_lagged_ratio_default,
    1.5,
    "if async_lagged_grad_discard_ratio is not set in trainer_config.conf"
    "use it as defalut value");

namespace bubblefs {
namespace mypaddle {

// @TODO(yanfei):
// if armed with high density computation resource per node, pserver could also
// utilize GPU to reduce overhead. if this mechanism is used, it could pipeline
// network receiving and GPU computation to reduce the network overhead even
// further. the pipeline could help to accelerate BIG model training.
// @TODO:(yanfei)
// for cpu and less/low gpu machine, the time exhausted by forward and backward
// could be larger than optimization at pserver. However, if armed with lots of
// gpus per node and if the model size is so large enough that limited cpu
// computation causes big optmization latency, the GPU may be required by
// pserver.

/**
 * Client interface for the parameter server
 *
 * it implements several rpc API for remote parameter client usage.
 * for sync-sgd, client needs one controller thread to build connections
 * to all pservers, these controller connections do barriers
 * synchronization with these connections used for transfering data.
 * each data connection uses block based fine grained synchronization
 * to gain better scalability. Merging gradients from different trainers
 * are concurrently executed with block units, so that some network
 * overhead will be hidden in merging gradient.
 * for async-sgd, the difference is that pserver will do optimization
 * immediately if the gradients are ready, so that pserver needs to
 * prepare separate buffer to store value for sending back to trainer
 * to prevent from being polluted.
 */
class ParameterServer2 : public ProtoServer {
protected:
  /// parameter_ mutex.
  RWLock parameterMutex_;

  typedef std::pair<size_t, int64_t> BlockKey;
  struct BlockKeyHash {
    size_t operator()(const BlockKey& key) const {
      return std::hash<size_t>()(key.first) + key.second;
    }
  };

  // TODO(yanfei):
  // if index data structure is based on parameters instead of blocks, the
  // lookup performance could be better. In addition, the block memory
  // access almost exhibits good locality, so index data structure and
  // block data structure can be refined further, especially if gpu is used
  // for pserver.
  /**
   * all parameters are stored in CpuVector with a blockMap_ data structure
   * to index block data required by requests.
   */
  typedef std::unordered_map<BlockKey, int64_t, BlockKeyHash> BlockMap;
  /// <(para, block), global offset(byte) in all parameters>
  BlockMap blockOffsetMap_;
  /// <(para, block), global idx [0, nBlocksInAllParameters]>
  BlockMap blockIdMap_;

  std::vector<CpuVectorPtr> vectors_;
  std::vector<CpuMatrixPtr> matrices_;
  std::vector<CpuMemHandlePtr> dataMems_;

  // TODO(yanfei):
  // if storing sparse_remote_update() flag in request instead of
  // reading configMap_, and storing config within new block wise
  // overview data structure, the config mapping, block mapping
  // can be unified in single clean data structure. Use para_id
  // to index parameters, use offset to index block within parameter
  // and keep two index into single one.
  /**
   * mapping between parameter and config
   * different parameter allows different config, such as decay_rate.
   * for each request, it need to read config for adding gradient
   * and optmization.
   */
  std::unordered_map<size_t, ParameterConfig> configMap_;

  /**
   * to parallelize the multi-thread and multi-connnection
   * computation at pserver, it use block unit to reduce
   * the contention for computation, even further use block
   * level optimizater control for each block for some special
   * reason annotated below.
   */
  struct BlockInfo {
    const ParameterConfig* config;
    std::unique_ptr<std::mutex> lock;
    /// global offset for all parameters
    uint64_t offset;
    /**
     *
     * Async sgd in pserver is very different from sync sgd.
     * Each trainer follows startBatch, update*, finishBatch as in
     * sync sgd, but all these actions are almost executed by
     * multi-core and multi-thread simutaneously, so that async
     * sgd optimization is based on block level in reality, then
     * per block optimization is necessary indeed. In addition,
     * per block optimization is also perfered for performance
     * with multithreads.
     */
    std::unique_ptr<ParameterOptimizer> optimizer;
  };
  std::vector<BlockInfo> blockInfos_;

  typedef std::vector<std::pair<int64_t, int64_t>> BlockSegments;
  /// Because some blocks might not be fully used. We keep a
  /// record of which segments are used.
  BlockSegments usedSegments_;

  /// record pserver status, all status defined in ParameterService.pb
  PServerStatus status_;
  /// record all samples processed which could be used by optimizater
  std::atomic<int64_t> numSamplesProcessed_;
  double cost_;
  int mpiSize_;
  int dataSize_;
  /// configuration for current parameter optimizer
  OptimizationConfig config_;

  /**
   * The ReadWriteBuffer is based on std::vector, but aligned for avx/sse
   * compute. And add some helper method to allocate memory aligned blocks.
   *
   * @param T          type of element.
   * @param AlignBytes the memory aligned bytes for allocated blocks.
   */
  template <typename T, size_t AlignBytes>
  class ReadWriteBuffer
      : public std::vector<T, AlignedAllocator<T, AlignBytes>> {
  public:
    static_assert(sizeof(T) % AlignBytes == 0 || AlignBytes % sizeof(T) == 0,
                  "Type T must be able to aligned.");

    /**
     * @brief IsTLargerThanAlign compiled time calculated constant for is type
     * T larger than alignments.
     */
    constexpr static bool IsTLargerThanAlign = sizeof(T) >= AlignBytes;

    static_assert(std::is_pod<T>::value, "T must be POD type.");

    /**
     * @brief if AlignBytes > sizeof(T), then will calcuate how many elements
     * can be stored in AlignBytes.
     */
    constexpr static size_t AlignElementCount = AlignBytes / sizeof(T);

    static_assert(AlignElementCount ==
                          (AlignElementCount & -AlignElementCount) ||
                      AlignBytes > sizeof(T),
                  "AlignElementCount should be exp of 2");

    /**
     * @brief Resize Buffer, with block count that will be allocated. Each block
     * will be memory aligned in AlignBytes.
     * @param size The element count in all blocks.
     * @param alignBlockCount The block count that will be allocated.
     */
    void resizeWithAlignHints(size_t size, size_t alignBlockCount = 1) {
      if (IsTLargerThanAlign) {  //! So, each elements is memory aligned.
        this->resize(size);
      } else {
        //! at most, we need such elements in buffer to make sure each block is
        //! aligned.
        this->resize(size + alignBlockCount * (AlignElementCount - 1));
      }
    }

    /**
     * @brief reset aligned allocate blocks.
     */
    void resetAlignAlloc() { this->curOffset_ = 0; }

    /**
     * @brief get next aligned block address.
     * @param blockSize is the element count in each block.
     * @return Aligned block address.
     */
    T* nextBlock(size_t blockSize) {
      T* r = &this->operator[](curOffset_);
      curOffset_ += blockSize;

      if (!IsTLargerThanAlign) {
        curOffset_ =
            (curOffset_ + AlignElementCount - 1) & ~(AlignElementCount - 1);
      }
      return r;
    }

  private:
    size_t curOffset_;
  };

  /// to buffer the data from network for further processing to
  /// reduce redundant memory allocation.
  ThreadLocal<ReadWriteBuffer<real, ALIGN_HINT>> readWriteBuffer_;

  /// size of the parameter
  int64_t size_;

  /// for synchronized training, check details in addGradient()
  /// and doOperation()
  ThreadBarrier gradientReadyBarrier_;
  ThreadBarrier parameterReadyBarrier_;
  ThreadBarrier passBarrier_;
  ThreadLocal<std::vector<SendParameterRequest>> requestVec_;
  ThreadLocal<std::vector<ProtoResponseCallbackEx>> callbackVec_;

  std::atomic<int> numPassFinishClients_;
  bool allClientPassFinish_;

  std::vector<std::unique_ptr<ThreadBarrier>> synchronizeBarriers_;
  std::atomic<int> serverId_;

  /**
   *
   * for lagged async gradient gradient commit control in Async Sgd.
   * discard lagged gradients from too slow nodes, whose gradients
   * exhibits bad quality.
   * Algorithm:
   * pserver:
   * 1. initial asyncUpdaterSteps = 0, asyncTrainerSteps_[N] = 0.
   * syncUpdaterSteps means
   *    the version of parameter value.
   * 2. when pull arrives, record asyncUpdateSteps_ into
   * syncTrainerSteps_[trainer_id]
   * 3. when push arrives, compare asyncUpdateSteps_ with
   * syncTrainerSteps_[trainer_id]
   *    if delta > threshold, discard current gradient, else commit
   *    gradient.
   * 4. reset asyncUpdaterSteps_ and asyncTrainerSteps_[N] when pass
   * finished
   * Note:
   * it can not discard all lag-gradient strictly in some special
   * condition. part of gradients could be discarded if
   * ConcurrentRemoteParameterUpdater is sed.
   * this algorithm is implemented in asynSGD()
   */
  int64_t asyncLaggedThreshold_;
  std::atomic<int64_t> asyncUpdateSteps_;
  std::vector<int64_t> asyncTrainerSteps_;
  size_t asyncLaggedGradientsNum_;
  /// stat all async update
  std::vector<size_t> asyncUpdateStat_;
  /// stat per trainer_id
  std::vector<size_t> asyncTrainerDiscardStat_;
  /// stat per trainer_id
  std::vector<size_t> asyncTrainerCommitStat_;

  /// only used by controller and other control cmd from trainer number 0
  std::unique_ptr<SyncThreadPool> syncThreadPool_;

  /// pserver for sparse remote update parameters
  bool isSparseServer_;

  /// barrier performance tuning sync-sgd required
  std::atomic<int64_t> batchId_;

public:
  struct Buffer {
    real* base;
    size_t size;
  };

protected:
  /// async gradient commit control
  bool asyncGrdientCommitCheckAndStat(const SendParameterRequest& request);

public:
  /// disable default parameter for overloading
  /// @rdmaCpu:the id of cpu core hosting RDMA server(0-N)
  /// -1 means using TCP transport instead of RDMA
  ParameterServer2(const std::string& addr, int port, int rdmaCpu = -1);

  ~ParameterServer2() {}

  static const std::string kRetMsgInvalidMatrixHandle;
  static const std::string kRetMsgInvalidVectorHandle;
  static const std::string kRetMsgUnknownOperation;

  /// service functions
  template <typename Dtype>
  void reduceAndSendData(const SendDataRequest& request,
                         std::unique_ptr<MsgReader>& msgReader,
                         ProtoResponseCallbackEx& callback);

  void templateReduceSum(const SendDataRequest& request,
                         std::unique_ptr<MsgReader>& msgReader,
                         ProtoResponseCallbackEx& callback);

  /**
   * @brief framework for sending parameters
   *
   * @note  different parameter data type can be sent to pserver.
   *        in most case, the api is used to send gradients from
   *        trainer to pserver.
   *        it also can be used to retrieve parameters from pserver
   */
  void sendParameter(const SendParameterRequest& request,
                     std::unique_ptr<MsgReader> msgReader,
                     ProtoResponseCallbackEx callback);

  void sendData(const SendDataRequest& request,
                std::unique_ptr<MsgReader> msgReader,
                ProtoResponseCallbackEx callback);

  /**
   * @brief send config to pserver
   *
   * @note  it can help pserver to understand the configuration for
   * optimization,
   *        logging control, duplicated initialization, etc.
   */
  void setConfig(const SetConfigRequest& request,
                 ProtoResponseCallback callback);

  /**
   * @brief get status for pserver
   *
   * @note  used to check if parameters are ready at pserver
   */
  void getStatus(const GetStatusRequest& request,
                 ProtoResponseCallback callback);

  /**
   * @brief set status for pserver
   *
   * @note  used to check if parameters are ready at pserver, since parameters
   *        at pserver are initialized by trainer
   */
  void setStatus(const SetStatusRequest& request,
                 ProtoResponseCallback callback);

  /**
   * @brief framework for doing some operation at pserver end
   *
   * @note  if sync-sgd is used, controller will calling op_SGD action
   *        for gradient optimization.
   *        check avaiable operations in opFuncs[]
   */
  void doOperation(const DoOperationRequest& request,
                   ProtoResponseCallback callback);

  /// Create a column vector. The size is the dimension of parameter
  void createVector(const CreateVectorRequest& request,
                    ProtoResponseCallback callback);

  void releaseVector(const ReleaseVectorRequest& request,
                     ProtoResponseCallback callback);

  /// Create a column major matrix. The number of rows is the dimension of
  /// parameter. The number of columns is specifed by num_cols.
  void createMatrix(const CreateMatrixRequest& request,
                    ProtoResponseCallback callback);

  void releaseMatrix(const ReleaseMatrixRequest& request,
                     ProtoResponseCallback callback);
  /**
   * @brief stateful control for indicationg sync pass start
   *
   * @note  it is valuable for logging and state control,
   *        especially for sync-sgd control
   */
  void waitPassStart(const WaitPassStartRequest& request,
                     ProtoResponseCallback callback);

  /**
   * @brief stateful control for indicationg sync pass end
   *
   * @note  it is valuable for logging and state control,
   *        especially for sync-sgd control
   */
  void waitPassFinish(const WaitPassFinishRequest& request,
                      ProtoResponseCallback callback);

  /**
   * @brief synchronize all distributed trainers
   *
   * @note  it's general api for synchronizing trainer and pserver
   */
  void synchronize(const SynchronizeRequest& request,
                   ProtoResponseCallback callback);

  /**
   * @brief stateful control for indicating async pass is finished
   *
   * @note  it is valuable for logging control, state reset, etc.
   */
  void asyncFinishPass(const SynchronizeRequest& request,
                       ProtoResponseCallback callback);

  void loadValueVector(const LoadValueRequest& request,
                       ProtoResponseCallback callback);

  void saveValueVector(const SaveValueRequest& request,
                       ProtoResponseCallback callback);

public:
  /**
   * @brief initialize parameter server
   */
  bool init();

  /**
   * @brief set parameters at pserver
   *
   * @note  do parameter initialization if neccessy.
   */
  void setParameter(const SendParameterRequest& request,
                    std::vector<Buffer>& inputBuffers,
                    SendParameterResponse* response,
                    std::vector<Buffer>* outputBuffers);

  /**
   * @brief receive gradients and do optimization for async-sgd
   *
   * @note  this api asynchronizately receives all data from all
   *        trainers, and immediately do optimization and return
   *        optimizated value for trainer.
   *        this above routine are block based atomic updating,
   *        which means different block could based different stale
   *        gradient.
   *        it will discard some lagged gradients by default for
   *        better convergence.
   */
  void asyncSGD(const SendParameterRequest& request,
                std::vector<Buffer>& inputBuffers,
                SendParameterResponse* response,
                std::vector<Buffer>* outputBuffers);

  /**
   * @brief merge gradients from all trainer
   *
   * @note  this api use block based parallelization as fine grained
   *        parallelization which benifits lock contention and latency
   *        hidden for communication, also can harness multi-core
   *        efficiently.
   *        it also implements the synchronization for sync-sgd
   */
  void addGradient(const SendParameterRequest& request,
                   std::vector<Buffer>& inputBuffers,
                   SendParameterResponse* response,
                   std::vector<Buffer>* outputBuffers);

  /**
   * @brief get dense parameters from pserver
   *
   * @note  for some specified condition, trainer will get parameters from
   *        pservers.
   *        e.g.
   *        if all parameters are stored at perver end for big model training
   *        trainer can use it to retrieve all parameters if necessary.
   */
  void getParameter(const SendParameterRequest& request,
                    std::vector<Buffer>& inputBuffers,
                    SendParameterResponse* response,
                    std::vector<Buffer>* outputBuffers);

  /**
   * @brief get sparse value from parameter server
   *
   * @note  with sparse enabled, pservers own all latest value
   *        while trainer only retrieve value that only are needed.
   *        e.g.
   *        trainer will do prefetch action to retrieve necessary latest
   *        value from pserver for sparse calculation.
   */
  void getParameterSparse(const SendParameterRequest& request,
                          std::vector<Buffer>& inputBuffers,
                          SendParameterResponse* response,
                          std::vector<Buffer>* outputBuffers);

protected:
  void mergeSegments(BlockSegments* segments);

  /// set the unused segments to zero
  void clearUnusedSegments(CpuVector* vec);

  // TODO(yanfei):
  // if read data and do optimization interleavely block by block,
  // the performance could be better for gaining less network congestion.
  /// read all data from connection and store it in static pre-allocated buffer
  void readAllBlocks(MsgReader* msgReader,
                     std::vector<ParameterServer2::Buffer>* buffers);

  const ParameterConfig& getParameterConfig(const ParameterBlock& block) {
    CHECK_LT(block.para_id(), -1UL) << "invalid parameter id:"
                                    << block.para_id();
    const auto it = configMap_.find(block.para_id());
    CHECK(it != configMap_.end()) << "can not find parameter id: "
                                  << block.para_id();
    return it->second;
  }

  /// it implictly check blockOffsetMap_ while retrieving blockId
  const ParameterConfig& getParameterConfig(int64_t blockId) const {
    CHECK(blockId >= 0 && blockId < (int64_t)blockInfos_.size())
        << "block idx out of range, id: " << blockId
        << " info size: " << blockInfos_.size();
    return *(blockInfos_[blockId].config);
  }

  template <class Response>
  bool isValidVectorHandle(int64_t handle, Response* response) {
    if (handle < 0 || (size_t)handle >= vectors_.size()) {
      LOG(ERROR) << "Invalid vector handle " << handle;
      response->set_return_message(kRetMsgInvalidVectorHandle);
      return false;
    }
    return true;
  }

  template <class Response>
  bool isValidMatrixHandle(int64_t handle, Response* response) {
    if (handle < 0 || (size_t)handle >= matrices_.size()) {
      LOG(ERROR) << "Invalid matrix handle " << handle;
      response->set_return_message(kRetMsgInvalidMatrixHandle);
      return false;
    }
    return true;
  }

  /**
   * @brief get block offset
   *
   * @note  block.begin_dim is added to the block offset.
   *        return -1 if block cannot be found
   */
  int64_t getBlockOffset(const ParameterBlock& block) const {
    BlockKey key(block.para_id(), block.block_id());
    auto it = blockOffsetMap_.find(key);
    if (it == blockOffsetMap_.end()) {
      return -1;
    }
    return it->second;
  }

  /// return -1 if block cannot be found
  int64_t getBlockId(const ParameterBlock& block) const {
    BlockKey key(block.para_id(), block.block_id());
    auto it = blockIdMap_.find(key);
    if (it == blockIdMap_.end()) {
      return -1;
    }
    return it->second;
  }

  /**
   * @brief prepare data for sending back
   *
   * @note  modify reponse and outputBuffers for sending parameter
   *        back to client. The buffer for socket sending uses
   *        vectors_[parameterType] directly
   *        for dense with sync-sgd
   */
  void sendBackParameter(const ParameterBlock& block,
                         int parameterType,
                         SendParameterResponse* response,
                         std::vector<Buffer>* outputBuffers);

  /**
   * @brief prepare data for sending back
   *
   * @note  modify response and outputBuffers for sending parameter
   *        back to client. The buffer for socket sending uses buffer->base
   *        The parameter values are copied from vectors_[parameterType]
   *        to buffer->base.
   *        for dense with async-sgd
   */
  void sendBackParameter(const ParameterBlock& block,
                         int parameterType,
                         SendParameterResponse* response,
                         Buffer* buffer,
                         std::vector<Buffer>* outputBuffers);
  /**
   * @brief prepare data for sending back
   *
   * @note  specified for sparse
   */
  void sendBackParameterSparse(const ParameterBlock& block,
                               int parameterType,
                               SendParameterResponse* response,
                               Buffer* buffer,
                               size_t width,
                               std::vector<Buffer>* outputBuffers);

  /**
   * framework routine for block parallelization
   * e.g.
   * for optimization on all blocks at pserver end, this routine can facilitize
   * the parallelize of do optimization on all blocks with multithreads.
   */
  typedef std::function<void(int64_t blockId, const VectorPtr vecs[])> ExecFunc;
  void parallelExecForEachBlock(ExecFunc func);
  void blockTraverse(BlockInfo& info,
                     const ParameterConfig& config,
                     int64_t offset,
                     size_t size,
                     const VectorPtr vecs[],
                     const ParameterOptimizer::TraverseCallback& callback);

public:
  typedef void (ParameterServer2::*OperatorFunction)(const Operation& operation,
                                                     OperationResult* result);

  /**
   * doOperation will call following operations indirectly
   * e.g.
   * for sync-sgd control, the controller in remote updater will send op_SGD
   * command to pserver, then send sendParameter request to pserver immediately.
   * the two function at pserver end will do cooperation to achieve the sync-sgd
   * gradient merge and optimization.
   * the most following operations are specified for owlqn, all operations are
   * under the context of doOperation function
   */
  static OperatorFunction opFuncs[];

  void op_SGD(const Operation& operation, OperationResult* result);

  void op_RESET(const Operation& operation, OperationResult* result);

  void op_utv(const Operation& operation, OperationResult* result);

  void op_au_bv(const Operation& operation, OperationResult* result);

  void op_COPY(const Operation& operation, OperationResult* result);

  void op_au(const Operation& operation, OperationResult* result);

  void op_au_bv_cw(const Operation& operation, OperationResult* result);

  void op_make_steepest_desc_dir(const Operation& operation,
                                 OperationResult* result);

  void op_fix_dir_signs(const Operation& operation, OperationResult* result);

  void op_dir_deriv(const Operation& operation, OperationResult* result);

  void op_fix_omega_signs(const Operation& operation, OperationResult* result);

  void op_cost(const Operation& operation, OperationResult* result);

  void op_start_pass(const Operation& operation, OperationResult* result);
  void op_finish_pass(const Operation& operation, OperationResult* result);

  void op_apply(const Operation& operation, OperationResult* result);

  void op_randomize(const Operation& operation, OperationResult* result);

  void op_load(const Operation& operation, OperationResult* result);
  void op_save(const Operation& operation, OperationResult* result);
};

const std::string ParameterServer2::kRetMsgInvalidMatrixHandle =
    "Invalid matrix handle";
const std::string ParameterServer2::kRetMsgInvalidVectorHandle =
    "Invalid vector handle";
const std::string ParameterServer2::kRetMsgUnknownOperation =
    "Unknown operation";

ParameterServer2::ParameterServer2(const std::string& addr,
                                   int port,
                                   int rdmaCpu)
    : ProtoServer(addr, port, rdmaCpu),
      dataSize_(0),
      size_(0),
      gradientReadyBarrier_(FLAGS_num_gradient_servers + 1),
      parameterReadyBarrier_(FLAGS_num_gradient_servers + 1),
      passBarrier_(FLAGS_num_gradient_servers + 1),
      numPassFinishClients_(0),
      allClientPassFinish_(false),
      serverId_(-1),
      batchId_(-1) {
  /**
   * register function for remote client calling, these functions
   * will be mapped to a data structure for quick looking up. each
   * request from trainer can contains one function name to indicate
   * remote action. this architecture looks like rpc style for pserver.
   */
  REGISTER_SERVICE_FUNCTION_EX(ParameterServer2, sendParameter);
  REGISTER_SERVICE_FUNCTION_EX(ParameterServer2, sendData);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, setConfig);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, setStatus);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, getStatus);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, doOperation);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, createVector);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, releaseVector);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, createMatrix);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, releaseMatrix);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, waitPassStart);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, waitPassFinish);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, synchronize);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, asyncFinishPass);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, loadValueVector);
  REGISTER_SERVICE_FUNCTION(ParameterServer2, saveValueVector);

  /// thread pool for parallelizing some computations
  if (FLAGS_pserver_num_threads > 1) {
    syncThreadPool_.reset(new SyncThreadPool(FLAGS_pserver_num_threads, false));
  }
}

bool ParameterServer2::init() {
  vectors_.resize(NUM_PARAMETER_TYPES);
  configMap_.clear();

  numSamplesProcessed_ = 0;
  cost_ = 0;
  char* mpienv = getenv("OMPI_COMM_WORLD_SIZE");
  if (mpienv != NULL) {
    mpiSize_ = atoi(mpienv);
  } else {
    mpiSize_ = 1;
  }
  status_ = PSERVER_STATUS_NOT_SET;
  dataMems_.resize(FLAGS_num_gradient_servers);
  synchronizeBarriers_.resize(SyncObject_ARRAYSIZE);
  for (auto& barrier : synchronizeBarriers_) {
    barrier.reset(new ThreadBarrier(FLAGS_num_gradient_servers));
  }

  // initialization for dicarding lagging gradient
  asyncUpdateSteps_ = 0;
  asyncTrainerSteps_.resize(FLAGS_num_gradient_servers);
  asyncTrainerSteps_.assign(asyncTrainerSteps_.size(), 0);
  asyncLaggedGradientsNum_ = 0;
  asyncUpdateStat_.resize(static_cast<int>(FLAGS_num_gradient_servers *
                                           FLAGS_async_lagged_ratio_default));
  asyncUpdateStat_.assign(asyncUpdateStat_.size(), 0);
  asyncTrainerDiscardStat_.resize(FLAGS_num_gradient_servers);
  asyncTrainerDiscardStat_.assign(asyncTrainerDiscardStat_.size(), 0);
  asyncTrainerCommitStat_.resize(FLAGS_num_gradient_servers);
  asyncTrainerCommitStat_.assign(asyncTrainerCommitStat_.size(), 0);

  return true;
}

void ParameterServer2::getStatus(const GetStatusRequest& request,
                                 ProtoResponseCallback callback) {
  (void)request;
  GetStatusResponse response;
  response.set_status(status_);
  callback(response);
}

void ParameterServer2::setStatus(const SetStatusRequest& request,
                                 ProtoResponseCallback callback) {
  status_ = request.status();
  SetStatusResponse response;
  callback(response);
}

void ParameterServer2::setConfig(const SetConfigRequest& request,
                                 ProtoResponseCallback callback) {
  {
    std::lock_guard<RWLock> guard(parameterMutex_);

    serverId_ = request.server_id();
    isSparseServer_ = request.is_sparse_server();

    if (!request.save_dir().empty()) {
      mkDir(request.save_dir().c_str());
    }

    for (const auto& config : request.param_configs()) {
      CHECK(!configMap_.count(config.para_id()))
          << "Duplicated parameter name: " << config.name();
      configMap_[config.para_id()] = config;
      CHECK_EQ(config.sparse_remote_update(), isSparseServer_);
    }

    config_ = request.opt_config();
    if (config_.algorithm() == TrainAlgorithm::AsyncSGD) {
      auto asyncLaggedRatio = config_.async_lagged_grad_discard_ratio();
      if (asyncLaggedRatio <= FLAGS_async_lagged_ratio_min) {
        LOG(INFO) << "WARNING: async_lagged_grad_discard_ratio is too small"
                  << "reset to default, async_lagged_grad_discard_ratio = "
                  << FLAGS_async_lagged_ratio_default;
        asyncLaggedRatio = FLAGS_async_lagged_ratio_default;
      }
      asyncLaggedThreshold_ =
          static_cast<int64_t>(FLAGS_num_gradient_servers * asyncLaggedRatio);
      LOG(INFO) << "discard lagged async gradient ratio: " << asyncLaggedRatio
                << " asyncLaggedhreshold: " << asyncLaggedThreshold_;
    }
    if (isSparseServer_ && config_.num_batches_per_send_parameter() > 1) {
      /// sparse server must NOT use local update mode
      config_.set_num_batches_per_send_parameter(1);
    }

    if (config_.num_batches_per_send_parameter() > 1 &&
        config_.center_parameter_update_method() == "average") {
      /// scaling L1/L2 decay rate as large as L1/L2 apply in trainer
      /// if parameter regularization in pserver
      for (auto& pair : configMap_) {
        ParameterConfig& config = pair.second;
        if (config_.num_batches_per_send_parameter() ==
            config.num_batches_regularization()) {
          real scale =
              config_.delta_add_rate() * config.num_batches_regularization();
          if (config_.algorithm() == "sgd") {
            scale *= FLAGS_num_gradient_servers;
          }
          config.set_decay_rate(config.decay_rate() * scale);
          if (config.decay_rate() > 0.1f) {
            LOG(FATAL) << "L2 decay=" << config.decay_rate()
                       << " for parameter:" << config.name()
                       << " is too large after scale in pserver!";
          }
          config.set_decay_rate_l1(config.decay_rate_l1() * scale);
          if (config.decay_rate_l1() > 0.1f) {
            LOG(FATAL) << "L1 decay=" << config.decay_rate_l1()
                       << " for parameter:" << config.name()
                       << " is too large after scale in pserver!";
          }

          LOG(INFO) << "parameter:" << config.name()
                    << " decay apply in pserver,"
                    << " L1 decay=" << config.decay_rate_l1()
                    << " L2 decay=" << config.decay_rate();
        }
      }
    }
  }

  SetConfigResponse response;
  callback(response);
}

real bufferSum(const std::vector<ParameterServer2::Buffer>& buffers) {
  real sum = 0;
  for (const auto buffer : buffers) {
    for (size_t i = 0; i < buffer.size; ++i) {
      sum += buffer.base[i];
    }
  }
  return sum;
}

void ParameterServer2::mergeSegments(BlockSegments* segments) {
  if (segments->empty()) {
    return;
  }
  std::sort(segments->begin(), segments->end());
  auto curr = segments->begin();
  for (auto it = segments->begin(); it != segments->end(); ++it) {
    if (it->first <= curr->second) {
      curr->second = std::max(curr->second, it->second);
    } else {
      ++curr;
      *curr = *it;
    }
  }
  ++curr;
  segments->erase(curr, segments->end());
}

void ParameterServer2::setParameter(const SendParameterRequest& request,
                                    std::vector<Buffer>& inputBuffers,
                                    SendParameterResponse* response,
                                    std::vector<Buffer>* outputBuffers) {
  (void)response;
  (void)outputBuffers;
  LOG(INFO) << "pserver: setParameter";
  std::lock_guard<RWLock> guard(parameterMutex_);

  int64_t numBlocks = blockIdMap_.size();
  CHECK_EQ(blockIdMap_.size(), blockOffsetMap_.size());
  /// total bytes for all the added blocks
  int64_t totalSize = size_;
  std::vector<int64_t> offsets;
  offsets.reserve(request.blocks_size());
  std::vector<int64_t> blockIds;
  blockIds.reserve(request.blocks_size());
  int bufferIndex = 0;

  if (!request.blocks().size()) {
    LOG(WARNING)
        << "--ports_num or --ports_num_for_sparse might be too large, "
        << "or total dense parameter size or sparse parameters size "
        << "might be too small, this psever doesn't store any parameter.";
    return;
  }

  for (const auto& block : request.blocks()) {
    /// block size for parameter(e.g. 128 for sparse row, 1K for dense)
    uint64_t blockSize = getParameterConfig(block).parameter_block_size();
    BlockKey key(block.para_id(), block.block_id());
    if (inputBuffers.size()) {  // if !=PSERVER_UPDATE_MODE_SET_PARAM_ZERO
      Buffer buffer = inputBuffers[bufferIndex];
      ++bufferIndex;
      CHECK_EQ(buffer.size, block.block_size())
          << "data size is too big:"
          << " block_size=" << block.block_size()
          << " data_size=" << buffer.size;
    }

    /// add a new block
    if (blockIdMap_.count(key) == 0) {
      blockOffsetMap_[key] = totalSize;
      blockIdMap_[key] = numBlocks;
      ++numBlocks;
      totalSize += blockSize;
    }
    offsets.push_back(blockOffsetMap_[key]);
    blockIds.push_back(blockIdMap_[key]);
  }

  size_ = totalSize;
  LOG(INFO) << "pserver: new cpuvector: size=" << size_;
  if (!vectors_[PARAMETER_VALUE]) {
    /// vectors_
    const auto types = sgdOptimizerGetTypes(config_, true /*inPserver*/);
    for (const auto type : types) {
      vectors_[type].reset(new CpuVector(size_));
      vectors_[type]->zeroMem();
    }

    blockInfos_.resize(numBlocks);
    for (auto& info : blockInfos_) {
      info.lock.reset(new std::mutex());
    }
  } else {
    CHECK_EQ((size_t)size_, vectors_[PARAMETER_VALUE]->getSize())
        << "Currently adding new blocks is not supported. "
        << "All blocks must be added in one setParameter call";
  }

  VectorPtr buf = vectors_[PARAMETER_VALUE];
  usedSegments_.reserve(offsets.size());
  /// if offsets is empty, means parameter_block_size is too big or too many
  /// nodes.
  if (offsets.empty()) {
    LOG(WARNING) << "in setParameter: offsets is empty";
  }
  for (size_t i = 0; i < offsets.size(); ++i) {
    size_t blockId = blockIds[i];
    BlockInfo& info = blockInfos_[blockId];
    const ParameterConfig& config = getParameterConfig(request.blocks(i));
    info.config = &config;
    info.offset = offsets[i];
    info.optimizer.reset(sgdOptimizerCreate(
        config_, config, config.sparse_remote_update(), true /*inPserver*/));
    if (config.sparse_remote_update()) {
      size_t width = config.dims(1);
      CHECK_EQ(config.parameter_block_size(), width)
          << "block size: " << config.parameter_block_size()
          << "width : " << width;
    }
    info.optimizer->init(1, info.config);
    usedSegments_.push_back(std::make_pair(
        offsets[i], offsets[i] + request.blocks(i).block_size()));
  }
  mergeSegments(&usedSegments_);

  if (request.update_mode() == PSERVER_UPDATE_MODE_SET_PARAM) {
    /// copy param from trainer
    for (size_t i = 0; i < offsets.size(); ++i) {
      Buffer buffer = inputBuffers[i];
      real* start = buf->getPoint(offsets[i]);
      CHECK_LE(offsets[i] + buffer.size, buf->getSize());
      memcpy(start, buffer.base, sizeof(real) * buffer.size);
    }
  } else {
    CHECK(request.update_mode() == PSERVER_UPDATE_MODE_SET_PARAM_ZERO);
    /// nothing to do, value vector zero mem already
  }
}

void ParameterServer2::addGradient(const SendParameterRequest& request,
                                   std::vector<Buffer>& inputBuffers,
                                   SendParameterResponse* response,
                                   std::vector<Buffer>* outputBuffers) {
  VLOG(1) << "pserver: addGradient";

  {
    ReadLockGuard guard(parameterMutex_);
    int bufferIndex = 0;
    for (const auto& block : request.blocks()) {
      int64_t offset = getBlockOffset(block);
      CHECK_GE(offset, 0) << "Only existing parameter block is allowed: "
                          << " id=" << block.para_id()
                          << " block id=" << block.block_id();

      int64_t blockId = getBlockId(block);
      CHECK_GE(blockId, 0) << "Only existing parameter block is allowed: "
                           << " id=" << block.para_id()
                           << " block id=" << block.block_id();

      Buffer buffer = inputBuffers[bufferIndex];
      ++bufferIndex;

      const real* gradientBuffer = buffer.base;
      real* gradientSumBuffer = vectors_[PARAMETER_GRADIENT]->getPoint(offset);

      size_t size = buffer.size;

      BlockInfo& info = blockInfos_[blockId];
      const ParameterConfig& config = getParameterConfig(blockId);
      if (config.sparse_remote_update()) {
        CHECK_EQ(size, config.parameter_block_size());
      } else {  // dense
        CHECK_LE(size, config.parameter_block_size());
      }
      std::lock_guard<std::mutex> guard(*info.lock);
      simd::addTo(gradientSumBuffer, gradientBuffer, size);
    }
  }
  if (request.batch_status() == BATCH_FINISH ||
      request.batch_status() == BATCH_START_AND_FINISH) {
    numSamplesProcessed_ += request.num_samples();
    cost_ += request.cost();
    VLOG(1) << "num samples: " << numSamplesProcessed_
            << ", new cost:" << cost_;

    /// notify doOperation gradient ready
    gradientReadyBarrier_.wait();

    /// wait doOperation finish
    parameterReadyBarrier_.wait();
    VLOG(1) << "start send back";
  }
}

bool ParameterServer2::asyncGrdientCommitCheckAndStat(
    const SendParameterRequest& request) {
  const auto trainerId = request.trainer_id();
  int64_t trainerSteps = asyncTrainerSteps_[trainerId];
  CHECK_GE(asyncUpdateSteps_, trainerSteps)
      << " async update steps overflows "
      << " trainer id: " << trainerId
      << " async update steps in pserver: " << asyncUpdateSteps_
      << " async update steps in request: " << trainerSteps;

  asyncUpdateSteps_++;
  bool commitGradient = true;

  int64_t delta = asyncUpdateSteps_ - trainerSteps;
  if (delta >= asyncLaggedThreshold_) {
    VLOG(1) << "discard Async Update: "
            << " trainer id: " << trainerId
            << " pserver steps: " << asyncUpdateSteps_
            << " request steps: " << trainerSteps;
    asyncLaggedGradientsNum_++;
    commitGradient = false;
  }
  /// stat on lagged steps, to get total discard distribution
  if (static_cast<size_t>(delta) < asyncUpdateStat_.size()) {
    asyncUpdateStat_[delta]++;
  } else {
    asyncUpdateStat_[asyncUpdateStat_.size() - 1]++;
  }
  /// stat on trainerId and discard, to get trainer condition
  if (commitGradient) {
    asyncTrainerCommitStat_[trainerId]++;
  } else {
    asyncTrainerDiscardStat_[trainerId]++;
  }

  return commitGradient;
}

static ThreadLocal<std::vector<bool>> localBlockBitset_;

void ParameterServer2::asyncSGD(const SendParameterRequest& request,
                                std::vector<Buffer>& inputBuffers,
                                SendParameterResponse* response,
                                std::vector<Buffer>* outputBuffers) {
  int64_t numBlocks = blockIdMap_.size();
  auto& localBlockBitset = *localBlockBitset_;

  if (isSparseServer_) {
    if (localBlockBitset.empty()) {
      localBlockBitset.resize(numBlocks);
    }
    localBlockBitset.assign(numBlocks, false);
  }

  ReadLockGuard guard(parameterMutex_);

  if (request.send_back_parameter()) {
    outputBuffers->reserve(request.blocks_size());
  }

  bool commitGradient = asyncGrdientCommitCheckAndStat(request);

  VectorPtr* vecs = parameter::getThreadLocalBuffer();
  size_t bufferIndex = 0;
  for (const auto& block : request.blocks()) {
    int64_t offset = getBlockOffset(block);
    CHECK_GE(offset, 0) << "Only existing parameter block is allowed: "
                        << " id=" << block.para_id()
                        << " block id=" << block.block_id();
    int64_t blockId = getBlockId(block);
    CHECK_GE(blockId, 0) << "Only existing parameter block is allowed: "
                         << " id=" << block.para_id()
                         << " block id=" << block.block_id();
    Buffer buffer = inputBuffers[bufferIndex];
    ++bufferIndex;

    size_t size = buffer.size;

    BlockInfo& info = blockInfos_[blockId];
    const ParameterConfig& config = getParameterConfig(blockId);

    std::lock_guard<std::mutex> guard(*info.lock);
    /// gradients are too obsolete, will be discarded
    if (commitGradient) {
      info.optimizer->startBatch(numSamplesProcessed_);

      for (const auto type : info.optimizer->getParameterTypes()) {
        vecs[type]->subVecFrom(*vectors_[type], offset, size);
      }
      vecs[PARAMETER_GRADIENT]->subVecFrom(buffer.base, 0, size);
      info.optimizer->update(vecs, config, isSparseServer_ ? 0 : -1);

      if (auto callback = info.optimizer->needSpecialTraversal(config)) {
        blockTraverse(info, config, offset, size, vecs, callback);
      }
      info.optimizer->finishBatch();
    }

    if (commitGradient && isSparseServer_) {
      localBlockBitset[blockId] = true;
    }

    if (!isSparseServer_ && request.send_back_parameter()) {  // dense
      int type = request.send_back_parameter_type();
      sendBackParameter(block, type, response, &buffer, outputBuffers);
    }
  }  /// foreach block

  asyncTrainerSteps_[request.trainer_id()] = asyncUpdateSteps_;

  if (commitGradient && isSparseServer_) {
    /// find blocks that trainer do not request update
    for (int64_t blockId = 0; blockId < numBlocks; ++blockId) {
      if (localBlockBitset[blockId]) {
        continue;
      }

      BlockInfo& info = blockInfos_[blockId];
      const ParameterConfig& config = *info.config;
      size_t size = config.parameter_block_size();

      std::lock_guard<std::mutex> guard(*info.lock);
      info.optimizer->startBatch(numSamplesProcessed_);
      if (auto callback = info.optimizer->needSpecialTraversal(config)) {
        blockTraverse(info, config, info.offset, size, vecs, callback);
      }
      info.optimizer->finishBatch();
    }
  }

  if (commitGradient && (request.batch_status() == BATCH_FINISH ||
                         request.batch_status() == BATCH_START_AND_FINISH)) {
    numSamplesProcessed_ += request.num_samples();
  }

  /// show some performance log if needed
  if (request.trainer_id() == 0) {
    /// batchId_ is approximately equal to "real batchId_"
    batchId_++;
  }
}

void ParameterServer2::getParameter(const SendParameterRequest& request,
                                    std::vector<Buffer>& inputBuffers,
                                    SendParameterResponse* response,
                                    std::vector<Buffer>* outputBuffers) {
  (void)inputBuffers;
  LOG(INFO) << "pserver: getParameter";
  ReadLockGuard guard(parameterMutex_);
  for (const auto& block : request.blocks()) {
    int type = request.send_back_parameter_type();
    sendBackParameter(block, type, response, outputBuffers);
  }
}

void ParameterServer2::getParameterSparse(const SendParameterRequest& request,
                                          std::vector<Buffer>& inputBuffers,
                                          SendParameterResponse* response,
                                          std::vector<Buffer>* outputBuffers) {
  (void)inputBuffers;
  auto& buffer = *readWriteBuffer_;
  size_t numReals = 0;
  for (const auto& block : request.blocks()) {
    numReals += getParameterConfig(block).dims(1);
  }
  buffer.resize(numReals);

  VLOG(3) << "pserver: getParameterSparse, numReals=" << numReals;

  ReadLockGuard guard(parameterMutex_);
  size_t offset = 0;
  for (const auto& block : request.blocks()) {
    size_t width = getParameterConfig(block).dims(1);
    Buffer buf = {buffer.data() + offset, width};
    int type = request.send_back_parameter_type();
    sendBackParameterSparse(block, type, response, &buf, width, outputBuffers);
    offset += width;
  }
}

void ParameterServer2::sendBackParameter(const ParameterBlock& block,
                                         int parameterType,
                                         SendParameterResponse* response,
                                         std::vector<Buffer>* outputBuffers) {
  ParameterBlock* returnBlock = response->add_blocks();
  returnBlock->set_para_id(block.para_id());
  returnBlock->set_block_id(block.block_id());
  returnBlock->set_begin_pos(block.begin_pos());
  returnBlock->set_block_size(block.block_size());

  int64_t offset = getBlockOffset(block);
  CHECK_GE(offset, 0) << "Only existing parameter block is allowed: "
                      << " id=" << block.para_id()
                      << " block id=" << block.block_id();

  real* valueBuffer = vectors_[parameterType]->getPoint(offset);
  outputBuffers->push_back({valueBuffer, (size_t)block.block_size()});
}

void ParameterServer2::sendBackParameter(const ParameterBlock& block,
                                         int parameterType,
                                         SendParameterResponse* response,
                                         Buffer* buffer,
                                         std::vector<Buffer>* outputBuffers) {
  ParameterBlock* returnBlock = response->add_blocks();
  returnBlock->set_para_id(block.para_id());
  returnBlock->set_block_id(block.block_id());
  returnBlock->set_begin_pos(block.begin_pos());
  returnBlock->set_block_size(block.block_size());

  int64_t offset = getBlockOffset(block);
  CHECK_GE(offset, 0) << "Only existing parameter block is allowed: "
                      << " id=" << block.para_id()
                      << " block id=" << block.block_id();

  size_t size = buffer->size;
  real* valueBuffer = vectors_[parameterType]->getPoint(offset);
  /// copy to second buffer to avoid to be polluted by other request
  memcpy(buffer->base, valueBuffer, sizeof(real) * size);
  outputBuffers->push_back({buffer->base, size});
}

void ParameterServer2::sendBackParameterSparse(
    const ParameterBlock& block,
    int parameterType,
    SendParameterResponse* response,
    Buffer* buffer,
    size_t width,
    std::vector<Buffer>* outputBuffers) {
  ParameterBlock* returnBlock = response->add_blocks();
  returnBlock->set_para_id(block.para_id());
  returnBlock->set_block_id(block.block_id());
  returnBlock->set_begin_pos(block.begin_pos());
  returnBlock->set_block_size(block.block_size());
  int64_t offset = getBlockOffset(block);
  CHECK_GE(offset, 0) << "Only existing parameter block is allowed: "
                      << " id=" << block.para_id()
                      << " block id=" << block.block_id();

  real* valueBuffer = vectors_[parameterType]->getPoint(offset);
  CHECK_EQ(buffer->size, width);
  memcpy(buffer->base, valueBuffer, width * sizeof(real));
  outputBuffers->push_back(*buffer);
}

void ParameterServer2::readAllBlocks(
    MsgReader* msgReader, std::vector<ParameterServer2::Buffer>* buffers) {
  auto& buffer = *readWriteBuffer_;
  size_t numBlocks = msgReader->getNumBlocks();
  buffer.resizeWithAlignHints(msgReader->getTotalLength() / sizeof(real),
                              numBlocks);
  std::vector<void*> bufs(numBlocks);
  buffers->clear();
  buffers->reserve(numBlocks);
  buffer.resetAlignAlloc();
  for (size_t i = 0; i < numBlocks; ++i) {
    size_t len = msgReader->getBlockLength(i);
    CHECK_EQ(len % sizeof(real), (size_t)0);
    size_t size = len / sizeof(real);
    bufs[i] = buffer.nextBlock(size);
    buffers->push_back({(real*)bufs[i], size});
  }
  msgReader->readBlocks(bufs);
}

void ParameterServer2::sendParameter(const SendParameterRequest& request,
                                     std::unique_ptr<MsgReader> msgReader,
                                     ProtoResponseCallbackEx callback) {
  SendParameterResponse response;
  std::vector<Buffer> inputBuffers;
  std::vector<Buffer> outputBuffers;
  readAllBlocks(msgReader.get(), &inputBuffers);
  msgReader.reset();

  switch (request.update_mode()) {
    case PSERVER_UPDATE_MODE_SET_PARAM:
    case PSERVER_UPDATE_MODE_SET_PARAM_ZERO:
      setParameter(request, inputBuffers, &response, &outputBuffers);
      break;
    case PSERVER_UPDATE_MODE_GET_PARAM:
      getParameter(request, inputBuffers, &response, &outputBuffers);
      break;
    case PSERVER_UPDATE_MODE_GET_PARAM_SPARSE:
      getParameterSparse(request, inputBuffers, &response, &outputBuffers);
      break;
    case PSERVER_UPDATE_MODE_ASYNC_SGD:
      asyncSGD(request, inputBuffers, &response, &outputBuffers);
      break;
    case PSERVER_UPDATE_MODE_ADD_GRADIENT:
      addGradient(request, inputBuffers, &response, &outputBuffers);
      break;
    case PSERVER_UPDATE_MODE_AVERAGE_PARAMETER:
      break;
  }
  switch (request.update_mode()) {
    case PSERVER_UPDATE_MODE_ADD_GRADIENT:
      (*requestVec_).push_back(request);
      (*callbackVec_).push_back(callback);
      if (request.batch_status() == BATCH_FINISH ||
          request.batch_status() == BATCH_START_AND_FINISH) {
        for (size_t i = 0; i < (*requestVec_).size(); i++) {
          ReadLockGuard guard(parameterMutex_);
          SendParameterRequest& request = (*requestVec_)[i];
          SendParameterResponse responseTemp;

          std::vector<iovec> outputIovs;
          if (request.send_back_parameter()) {
            CHECK(!isSparseServer_);
            std::vector<Buffer> outputBuffersTemp;
            for (const auto& block : request.blocks()) {
              int type = request.send_back_parameter_type();
              sendBackParameter(block, type, &responseTemp, &outputBuffersTemp);
            }
            outputIovs.reserve(outputBuffersTemp.size());
            for (auto buffer : outputBuffersTemp) {
              outputIovs.push_back({buffer.base, buffer.size * sizeof(real)});
            }
          }

          ProtoResponseCallbackEx& callbackTemp = (*callbackVec_)[i];
          callbackTemp(responseTemp, outputIovs);
        }
        (*requestVec_).clear();
        (*callbackVec_).clear();
      }
      break;
    case PSERVER_UPDATE_MODE_SET_PARAM:
    case PSERVER_UPDATE_MODE_SET_PARAM_ZERO:
    case PSERVER_UPDATE_MODE_GET_PARAM:
    case PSERVER_UPDATE_MODE_GET_PARAM_SPARSE:
    case PSERVER_UPDATE_MODE_ASYNC_SGD:
    case PSERVER_UPDATE_MODE_AVERAGE_PARAMETER:
      std::vector<iovec> outputIovs;
      outputIovs.reserve(outputBuffers.size());
      for (auto buffer : outputBuffers) {
        outputIovs.push_back({buffer.base, buffer.size * sizeof(real)});
      }
      callback(response, outputIovs);
      break;
  }
}

template <typename Dtype>
void ParameterServer2::reduceAndSendData(const SendDataRequest& request,
                                         std::unique_ptr<MsgReader>& msgReader,
                                         ProtoResponseCallbackEx& callback) {
  SendDataResponse response;
  response.set_type(request.type());
  response.set_server_id(serverId_);

  auto sendData = reinterpret_cast<Dtype*>(dataMems_[0].get()->getBuf());
  size_t rawMemSize = dataMems_[0].get()->getSize();
  CHECK_EQ(rawMemSize % sizeof(Dtype), 0U);
  size_t dataMemSize = rawMemSize / sizeof(Dtype);
  for (size_t i = 1; i < dataMems_.size(); ++i) {
    CHECK_EQ(dataMems_[i].get()->getSize(), rawMemSize);
    auto data = reinterpret_cast<Dtype*>(dataMems_[i].get()->getBuf());
    for (size_t j = 0; j < dataMemSize; ++j) {
      sendData[j] += data[j];
    }
  }
  std::vector<iovec> outputIovs;
  auto block = response.add_blocks();
  outputIovs.push_back({sendData, rawMemSize});
  block->set_total_size(rawMemSize);
  block->set_data_size(sizeof(Dtype));
  callback(response, outputIovs);
}

void ParameterServer2::templateReduceSum(const SendDataRequest& request,
                                         std::unique_ptr<MsgReader>& msgReader,
                                         ProtoResponseCallbackEx& callback) {
  const auto& block = request.blocks(0);
  switch (block.data_type()) {
    case TRANS_FLOAT:
      reduceAndSendData<float>(request, msgReader, callback);
      break;
    case TRANS_DOUBLE:
      reduceAndSendData<double>(request, msgReader, callback);
      break;
    case TRANS_INT32:
      reduceAndSendData<int>(request, msgReader, callback);
      break;
    case TRANS_UINT32_T:
      reduceAndSendData<uint32_t>(request, msgReader, callback);
      break;
    case TRANS_INT64_T:
      reduceAndSendData<int64_t>(request, msgReader, callback);
      break;
    case TRANS_UINT64_T:
      reduceAndSendData<uint64_t>(request, msgReader, callback);
      break;
    default:
      LOG(FATAL) << "not supported";
      break;
  }
}

void ParameterServer2::sendData(const SendDataRequest& request,
                                std::unique_ptr<MsgReader> msgReader,
                                ProtoResponseCallbackEx callback) {
  SendDataResponse response;
  response.set_type(request.type());
  response.set_server_id(serverId_);

  switch (request.update_mode()) {
    case DATA_UPDATE_MODE_SET_OWN: {
      CHECK_EQ(msgReader->getNumBlocks(), (size_t)(request.blocks_size()));
      size_t totalLen = msgReader->getTotalLength();
      if (totalLen > 0) {
        CHECK_EQ(msgReader->getNumBlocks(), 1U)
            << "Only one block currently support now!";
        const auto& block = request.blocks(0);
        if (0 == dataSize_) {
          dataSize_ = block.data_size();
        } else {
          CHECK_EQ(dataSize_, block.data_size());
        }
        int64_t serverId = request.server_id();
        if (serverId_ < 0) {
          serverId_ = serverId;
        } else {
          CHECK_EQ(serverId_, serverId);
        }
        int64_t clientId = request.client_id();
        dataMems_[clientId] = std::make_shared<CpuMemoryHandle>(totalLen);
        CHECK_EQ(totalLen % sizeof(block.data_size()), 0U);
        msgReader->readNextBlock(dataMems_[clientId].get()->getBuf());
      }
      msgReader.reset();
      std::vector<iovec> outputIovs;
      callback(response, outputIovs);
      break;
    }
    case DATA_UPDATE_MODE_GET_ALL: {
      /// Currently only support DATA_REDUCE_SUM
      /// And their Operations are just add
      CHECK(DATA_REDUCE_SUM == request.type());
      templateReduceSum(request, msgReader, callback);
      break;
    }
    default: { LOG(FATAL) << "not supported"; }
  }
}

void ParameterServer2::clearUnusedSegments(CpuVector* vec) {
  real* data = vec->getData();
  if (usedSegments_.empty()) {
    return;
  }
  memset(data, 0, sizeof(real) * usedSegments_[0].first);
  memset(data + usedSegments_.back().second,
         0,
         sizeof(real) * (size_ - usedSegments_.back().second));
  size_t n = size_ - usedSegments_.back().second;

  for (size_t i = 1; i < usedSegments_.size(); ++i) {
    memset(
        data + usedSegments_[i - 1].second,
        0,
        sizeof(real) * (usedSegments_[i].first - usedSegments_[i - 1].second));
    n += usedSegments_[i].first - usedSegments_[i - 1].second;
  }
}

void ParameterServer2::parallelExecForEachBlock(ExecFunc func) {
  SyncThreadPool::execHelper(
      syncThreadPool_.get(), [&](int tid, size_t numThreads) {
        int64_t numBlocks = blockIdMap_.size();
        VectorPtr* vecs = parameter::getThreadLocalBuffer();
        for (int64_t blockId = tid; blockId < numBlocks;
             blockId += numThreads) {
          func(blockId, vecs);
        }
      });
}

void ParameterServer2::blockTraverse(
    BlockInfo& info,
    const ParameterConfig& config,
    int64_t offset,
    size_t size,
    const VectorPtr vecs[],
    const ParameterOptimizer::TraverseCallback& callback) {
  /// setup sub bufs
  for (const auto type : info.optimizer->getParameterTypes()) {
    vecs[type]->subVecFrom(*vectors_[type], offset, size);
  }
  callback(vecs, config, config.sparse_remote_update() ? 0 : -1LU);
}

void ParameterServer2::op_SGD(const Operation& operation,
                              OperationResult* result) {
  (void)operation;
  (void)result;

  if (allClientPassFinish_) {
    /// when all clients signal pass finished, the update
    /// is empty.
    return;
  }

  {
    parallelExecForEachBlock([&](int64_t blockId, const VectorPtr vecs[]) {
      BlockInfo& info = blockInfos_[blockId];
      const ParameterConfig& config = getParameterConfig(blockId);
      int64_t offset = info.offset;
      size_t size = config.parameter_block_size();

      info.optimizer->startBatch(numSamplesProcessed_);

      for (const auto type : info.optimizer->getParameterTypes()) {
        vecs[type]->subVecFrom(*vectors_[type], offset, size);
      }
      info.optimizer->update(
          vecs, config, config.sparse_remote_update() ? 0 : -1LU);
      vecs[PARAMETER_GRADIENT]->zeroMem();

      if (auto callback = info.optimizer->needSpecialTraversal(config)) {
        blockTraverse(info, config, offset, size, vecs, callback);
      }
      info.optimizer->finishBatch();
    });
  }

  batchId_++;
}

void ParameterServer2::op_start_pass(const Operation& operation,
                                     OperationResult* result) {
  (void)operation;
  (void)result;

  parallelExecForEachBlock([&](int64_t blockId, const VectorPtr vecs[]) {
    BlockInfo& info = blockInfos_[blockId];
    info.optimizer->startPass();
  });
}

void ParameterServer2::op_finish_pass(const Operation& operation,
                                      OperationResult* result) {
  (void)operation;
  (void)result;

  parallelExecForEachBlock([&](int64_t blockId, const VectorPtr vecs[]) {
    BlockInfo& info = blockInfos_[blockId];
    const ParameterConfig& config = getParameterConfig(blockId);
    size_t size = config.parameter_block_size();

    /// catch up with
    if (auto callback = info.optimizer->startCatchUpWith()) {
      blockTraverse(info, config, info.offset, size, vecs, callback);
      info.optimizer->finishCatchUpWith();
    }

    /// finish pass
    info.optimizer->finishPass();
  });
  batchId_ = 0;
}

void ParameterServer2::op_apply(const Operation& operation,
                                OperationResult* result) {
  (void)operation;
  (void)result;

  parallelExecForEachBlock([&](int64_t blockId, const VectorPtr vecs[]) {
    BlockInfo& info = blockInfos_[blockId];
    const ParameterConfig& config = getParameterConfig(blockId);
    int64_t offset = info.offset;
    size_t size = config.parameter_block_size();

    // catch up with
    if (auto callback = info.optimizer->startCatchUpWith()) {
      blockTraverse(info, config, offset, size, vecs, callback);
      info.optimizer->finishCatchUpWith();
    }

    // apply to PARAMETER_APPLY
    if (auto callback = info.optimizer->apply()) {
      blockTraverse(info, config, offset, size, vecs, callback);
    }
  });
}

void ParameterServer2::op_randomize(const Operation& operation,
                                    OperationResult* result) {
  LOG(INFO) << "ParameterServer2::op_randomize: serverId=" << serverId_;

  CpuVector& valueVec = *vectors_[PARAMETER_VALUE];

  parallelExecForEachBlock([&](int64_t blockId, const VectorPtr vecs[]) {
    BlockInfo& info = blockInfos_[blockId];
    const ParameterConfig& config = getParameterConfig(blockId);
    size_t size = config.parameter_block_size();

    vecs[PARAMETER_VALUE]->subVecFrom(valueVec, info.offset, size);
    Parameter::randomize(vecs[PARAMETER_VALUE], config);
  });
}

void ParameterServer2::loadValueVector(const LoadValueRequest& request,
                                       ProtoResponseCallback callback) {
  LoadValueResponse response;
  LOG(INFO) << "ParameterServer2::loadValueVector: serverId=" << serverId_;

  constexpr int kBufLen = 100;
  char buf[kBufLen];
  snprintf(buf, kBufLen, "/pserver.%04d", static_cast<int>(serverId_));
  std::string filename = request.dir_name() + buf;

  std::ifstream fs(filename, std::ios_base::binary);
  CHECK(fs) << "Fail to open " << filename;

  CpuVector& vec = *vectors_[PARAMETER_VALUE];
  Parameter::Header header;
  CHECK(fs.read(reinterpret_cast<char*>(&header), sizeof(header)))
      << "Fail to read parameters in pserver";
  CHECK(Parameter::isHeaderFormatSupported(header.format))
      << "Incorrect format version: " << header.format;
  CHECK_EQ(header.size, (size_t)size_)
      << "The size (" << header.size << ") in the file does not match the size "
      << "(" << size_ << ") of the pserver: " << serverId_;
  CHECK_EQ(header.valueSize, sizeof(real)) << "Unsupported valueSize "
                                           << header.valueSize;
  CHECK(fs.read(reinterpret_cast<char*>(vec.getData()),
                header.size * sizeof(real)));

  callback(response);
}

void ParameterServer2::saveValueVector(const SaveValueRequest& request,
                                       ProtoResponseCallback callback) {
  SaveValueResponse response;
  LOG(INFO) << "ParameterServer2::SaveValueVector: serverId=" << serverId_;

  mkDir(request.dir_name().c_str());

  constexpr int kBufLen = 100;
  char buf[kBufLen];
  snprintf(buf, kBufLen, "/pserver.%04d", static_cast<int>(serverId_));
  std::string filename = request.dir_name() + buf;

  std::ofstream fs(filename, std::ios_base::binary);
  CHECK(fs) << "Fail to open " << filename;

  CpuVector& vec = vectors_[PARAMETER_APPLY] ? *vectors_[PARAMETER_APPLY]
                                             : *vectors_[PARAMETER_VALUE];
  Parameter::Header header;
  // TODO(TJ): save param headerFormat_
  header.format = PARAM_FORMAT_ORIGINAL;
  header.valueSize = sizeof(real);
  header.size = size_;

  CHECK_EQ(header.size, vec.getSize());

  CHECK(fs.write(reinterpret_cast<char*>(&header), sizeof(header)))
      << "Fail to write parameter in pserver: " << serverId_;

  CHECK(fs.write(reinterpret_cast<char*>(vec.getData()),
                 header.size * sizeof(real)))
      << "Fail to write parameter in pserver: " << serverId_;

  callback(response);
}

void ParameterServer2::op_RESET(const Operation& operation,
                                OperationResult* result) {
  (void)result;
  CpuVector* u = vectors_[operation.pvectors(0)].get();
  u->reset(operation.scalars(0));
  clearUnusedSegments(u);
}

void ParameterServer2::op_utv(const Operation& operation,
                              OperationResult* result) {
  real* u = vectors_[operation.pvectors(0)]->getData();
  real* v = vectors_[operation.pvectors(1)]->getData();
  int64_t size = size_;
  double sum = 0;
  for (int64_t i = 0; i < size; ++i) {
    sum += (double)u[i] * (double)v[i];
  }
  result->add_scalars(sum);
}

void ParameterServer2::op_au_bv(const Operation& operation,
                                OperationResult* result) {
  (void)result;
  real* u = vectors_[operation.pvectors(0)]->getData();
  real* v = vectors_[operation.pvectors(1)]->getData();
  int64_t size = size_;
  real a = operation.scalars(0);
  real b = operation.scalars(1);
  for (int64_t i = 0; i < size; ++i) {
    v[i] = a * u[i] + b * v[i];
  }
}

void ParameterServer2::op_COPY(const Operation& operation,
                               OperationResult* result) {
  (void)result;
  real* u = vectors_[operation.pvectors(0)]->getData();
  real* v = vectors_[operation.pvectors(1)]->getData();
  int64_t size = size_;
  for (int64_t i = 0; i < size; ++i) {
    v[i] = u[i];
  }
}

void ParameterServer2::op_au(const Operation& operation,
                             OperationResult* result) {
  (void)result;
  real* u = vectors_[operation.pvectors(0)]->getData();
  int64_t size = size_;
  real a = operation.scalars(0);
  for (int64_t i = 0; i < size; ++i) {
    u[i] *= a;
  }
}

void ParameterServer2::op_au_bv_cw(const Operation& operation,
                                   OperationResult* result) {
  (void)result;
  real* u = vectors_[operation.pvectors(0)]->getData();
  real* v = vectors_[operation.pvectors(1)]->getData();
  real* w = vectors_[operation.pvectors(2)]->getData();
  int64_t size = size_;
  real a = operation.scalars(0);
  real b = operation.scalars(1);
  real c = operation.scalars(2);
  for (int64_t i = 0; i < size; ++i) {
    w[i] = a * u[i] + b * v[i] + c * w[i];
  }
}

void ParameterServer2::op_make_steepest_desc_dir(const Operation& operation,
                                                 OperationResult* result) {
  (void)result;
  real* dir = vectors_[operation.pvectors(0)]->getData();
  real* grad = vectors_[operation.pvectors(1)]->getData();
  real* x = vectors_[operation.pvectors(2)]->getData();
  int64_t size = size_;
  real l1weight = operation.scalars(0);
  for (int64_t i = 0; i < size; ++i) {
    if (x[i] < 0) {
      dir[i] = -grad[i] + l1weight;
    } else if (x[i] > 0) {
      dir[i] = -grad[i] - l1weight;
    } else {
      if (grad[i] < -l1weight) {
        dir[i] = -grad[i] - l1weight;
      } else if (grad[i] > l1weight) {
        dir[i] = -grad[i] + l1weight;
      } else {
        dir[i] = 0;
      }
    }
  }
}

void ParameterServer2::op_fix_dir_signs(const Operation& operation,
                                        OperationResult* result) {
  (void)result;
  real* dir = vectors_[operation.pvectors(0)]->getData();
  real* steepestDescDir = vectors_[operation.pvectors(1)]->getData();
  int64_t size = size_;
  for (int64_t i = 0; i < size; ++i) {
    if (dir[i] * steepestDescDir[i] <= 0) {
      dir[i] = 0;
    }
  }
}

void ParameterServer2::op_fix_omega_signs(const Operation& operation,
                                          OperationResult* result) {
  (void)result;
  real* x = vectors_[operation.pvectors(0)]->getData();
  real* newx = vectors_[operation.pvectors(1)]->getData();
  int64_t size = size_;
  for (int64_t i = 0; i < size; ++i) {
    if (x[i] * newx[i] < 0) {
      newx[i] = 0;
    }
  }
}

void ParameterServer2::op_dir_deriv(const Operation& operation,
                                    OperationResult* result) {
  real* dir = vectors_[operation.pvectors(0)]->getData();
  real* grad = vectors_[operation.pvectors(1)]->getData();
  real* x = vectors_[operation.pvectors(2)]->getData();
  int64_t size = size_;
  real l1weight = operation.scalars(0);
  double sum = 0;
  for (int64_t i = 0; i < size; ++i) {
    if (dir[i] != 0) {
      if (x[i] < 0) {
        sum += dir[i] * (grad[i] - l1weight);
      } else if (x[i] > 0) {
        sum += dir[i] * (grad[i] + l1weight);
      } else if (dir[i] < 0) {
        sum += dir[i] * (grad[i] - l1weight);
      } else if (dir[i] > 0) {
        sum += dir[i] * (grad[i] + l1weight);
      }
    }
  }
  result->add_scalars(sum);
}

void ParameterServer2::op_cost(const Operation& operation,
                               OperationResult* result) {
  real* x = vectors_[operation.pvectors(0)]->getData();
  real* newgrad = vectors_[operation.pvectors(1)]->getData();
  int64_t size = size_;
  real l1weight = operation.scalars(0);
  real l2weight = operation.scalars(1);
  double cost_real = cost_ / mpiSize_;
  double sum_weight_l1 = 0;
  double sum_weight_l2 = 0;
  for (int64_t i = 0; i < size; ++i) {
    sum_weight_l1 += std::abs(x[i]);
    sum_weight_l2 += x[i] * x[i];
    newgrad[i] += 2.0 * l2weight * x[i];
  }
  cost_real += l1weight * sum_weight_l1 + l2weight * sum_weight_l2;
  result->add_scalars(cost_real);
}

ParameterServer2::OperatorFunction ParameterServer2::opFuncs[] = {
    nullptr,                         // PSERVER_OP_utu = 0;
    &ParameterServer2::op_utv,       // PSERVER_OP_utv = 1;
    &ParameterServer2::op_au,        // PSERVER_OP_au = 2;
    &ParameterServer2::op_au_bv,     // PSERVER_OP_au_bv = 3;
    nullptr,                         // PSERVER_OP_aAx_bu = 4;
    &ParameterServer2::op_SGD,       // PSERVER_OP_SGD = 5;
    &ParameterServer2::op_RESET,     // PSERVER_OP_RESET = 6;
    &ParameterServer2::op_COPY,      // PSERVER_OP_COPY = 7;
    &ParameterServer2::op_au_bv_cw,  // PSERVER_OP_au_bv_cw = 8;
    &ParameterServer2::op_make_steepest_desc_dir,
    /// PSERVER_OP_MAKE_STEEPEST_DESC_DIR = 9;
    &ParameterServer2::op_fix_dir_signs,    // PSERVER_OP_FIX_SIGNS = 10;
    &ParameterServer2::op_dir_deriv,        // PSERVER_OP_DIR_DERIV = 11;
    &ParameterServer2::op_fix_omega_signs,  // PSERVER_OP_FIX_OMEGA_SIGNS = 12;
    &ParameterServer2::op_cost,             // PSERVER_OP_COST = 13
    &ParameterServer2::op_start_pass,       // PSERVER_OP_START_PASS = 14
    &ParameterServer2::op_finish_pass,      // PSERVER_OP_FINISH_PASS = 15
    &ParameterServer2::op_randomize,        // PSERVER_OP_RANDOMIZE = 16
    &ParameterServer2::op_apply,            // PSERVER_OP_APPLY = 17
};

void ParameterServer2::doOperation(const DoOperationRequest& request,
                                   ProtoResponseCallback callback) {
  if (request.wait_for_gradient()) {
    /// wait gradient update
    gradientReadyBarrier_.wait();
    allClientPassFinish_ = numPassFinishClients_ == FLAGS_num_gradient_servers;
  }

  DoOperationResponse response;
  response.set_pass_finish(allClientPassFinish_);

  for (const auto& op : request.operations()) {
    OperationResult* opResult = response.add_results();
    if (op.operation() >= ARRAYSIZE(opFuncs)) {
      LOG(ERROR) << "Unknown operation " << op.operation();
      response.set_return_message(kRetMsgUnknownOperation);
    }
    OperatorFunction opFunc = opFuncs[op.operation()];
    if (!opFunc) {
      LOG(ERROR) << "Operation not implemented: " << op.operation();
      response.set_return_message(kRetMsgUnknownOperation);
    }
    (this->*opFunc)(op, opResult);
  }

  if (request.send_back_parameter()) {
    /// clean current cost
    cost_ = 0;

    if (allClientPassFinish_ && request.release_pass()) {
      /// This signals that all clients finish one pass, so waitPassFinish()
      /// will stop waiting.
      numPassFinishClients_ = 0;
    }

    /// notify addGradient() to send back parameter
    parameterReadyBarrier_.wait();
  }
  callback(response);
}

void ParameterServer2::waitPassStart(const WaitPassStartRequest& request,
                                     ProtoResponseCallback callback) {
  passBarrier_.wait();
  callback(WaitPassStartResponse());
}

void ParameterServer2::waitPassFinish(const WaitPassFinishRequest& request,
                                      ProtoResponseCallback callback) {
  numPassFinishClients_ += 1;

  while (numPassFinishClients_ != 0) {
    /// notify doOperation gradient ready
    gradientReadyBarrier_.wait();
    /// wait doOperation finish
    parameterReadyBarrier_.wait();
  }

  callback(WaitPassFinishResponse());
}

void ParameterServer2::synchronize(const SynchronizeRequest& request,
                                   ProtoResponseCallback callback) {
  synchronizeBarriers_[request.sync_object_id()]->wait();
  dataSize_ = 0;
  callback(SynchronizeResponse());
}

void ParameterServer2::asyncFinishPass(const SynchronizeRequest& request,
                                       ProtoResponseCallback callback) {
  synchronizeBarriers_[request.sync_object_id()]->wait();
  callback(SynchronizeResponse());

  if (request.trainer_id() == 0) {
    batchId_ = 0;
  }
}

void ParameterServer2::createVector(const CreateVectorRequest& request,
                                    ProtoResponseCallback callback) {
  (void)request;
  CreateVectorResponse response;
  LOG(INFO) << "ParameterServer2::createVector: size=" << size_;
  CpuVectorPtr vec = std::make_shared<CpuVector>(size_);
  int64_t handle = -1;
  {
    std::lock_guard<RWLock> guard(parameterMutex_);
    handle = vectors_.size();
    vectors_.push_back(vec);
  }
  response.set_handle(handle);
  callback(response);
}

void ParameterServer2::releaseVector(const ReleaseVectorRequest& request,
                                     ProtoResponseCallback callback) {
  ReleaseVectorResponse response;
  CpuVectorPtr vec;
  {
    std::lock_guard<RWLock> guard(parameterMutex_);
    vec.swap(vectors_[request.handle()]);
  }
  callback(response);
}

void ParameterServer2::createMatrix(const CreateMatrixRequest& request,
                                    ProtoResponseCallback callback) {
  CreateMatrixResponse response;
  /// We need to create column major matrix of size_ * num_cols
  /// Matrix is row majoar. Need to tranpose when use it.
  CpuMatrixPtr mat = std::make_shared<CpuMatrix>(request.num_cols(), size_);
  int64_t handle = -1;
  {
    std::lock_guard<RWLock> guard(parameterMutex_);
    handle = matrices_.size();
    matrices_.push_back(mat);
  }
  response.set_handle(handle);
  callback(response);
}

void ParameterServer2::releaseMatrix(const ReleaseMatrixRequest& request,
                                     ProtoResponseCallback callback) {
  ReleaseMatrixResponse response;
  CpuMatrixPtr mat;
  {
    std::lock_guard<RWLock> guard(parameterMutex_);
    mat.swap(matrices_[request.handle()]);
  }
  callback(response);
}

}  // namespace mypaddle
}  // namespace bubblefs