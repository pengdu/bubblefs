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

// Paddle/paddle/parameter/LearningRateScheduler.h
// Paddle/paddle/parameter/Argument.h
// Paddle/paddle/parameter/Parameter.h
// Paddle/paddle/parameter/ParameterOptimizer.h
// Paddle/paddle/parameter/Weight.h
// Paddle/paddle/parameter/Regularizer.h

#pragma once

#include <stdint.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "platform/paddle_locks.h"
#include "platform/paddle_threadlocal.h"
#include "utils/paddle_matrix.h"
#include "utils/paddle_vector.h"

namespace bubblefs {
namespace mypaddle {

class LearningRateScheduler {
public:
  static LearningRateScheduler* create(const OptimizationConfig& config);
  virtual ~LearningRateScheduler() {}
  virtual real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) = 0;

  static ClassRegistrar<LearningRateScheduler, OptimizationConfig> registrar_;
};  


typedef std::shared_ptr<std::vector<std::string>> SVectorPtr;

struct Argument {
  Argument()
      : in(nullptr),
        value(nullptr),
        ids(nullptr),
        grad(nullptr),
        strs(nullptr),
        frameHeight(0),
        frameWidth(0),
        frameDepth(0),
        sequenceStartPositions(nullptr),
        subSequenceStartPositions(nullptr),
        cpuSequenceDims(nullptr),
        deviceId(-1),
        allCount(0),
        valueCount(0),
        gradCount(0),
        dataId(0) {}
  Argument(const Argument& argument) {
    *this = argument;
    valueCount = 0;
    gradCount = 0;
    dataId = argument.dataId;
  }
  ~Argument() {}

  void operator=(const Argument& argument) {
    in = argument.in;
    value = argument.value;
    ids = argument.ids;
    grad = argument.grad;
    strs = argument.strs;
    sequenceStartPositions = argument.sequenceStartPositions;
    subSequenceStartPositions = argument.subSequenceStartPositions;
    cpuSequenceDims = argument.cpuSequenceDims;
    deviceId = argument.deviceId;
    allCount = argument.allCount;
    frameHeight = argument.frameHeight;
    frameWidth = argument.frameWidth;
    frameDepth = argument.frameDepth;
    dataId = argument.dataId;
  }

  MatrixPtr in;  // used if needed
  MatrixPtr value;
  IVectorPtr ids;  // a sequence of ids. Can be use for class id for costLayer
  MatrixPtr grad;  // If empty, gradient is not needed.
  SVectorPtr strs;

  // A dataBatch includes batchSize frames, one frame maybe not only vector
  size_t frameHeight;
  size_t frameWidth;
  size_t frameDepth;

  // If NULL, each position is treated independently.
  // Otherwise, its size should be #NumberOfSequences + 1.
  // The first position is always 0 and
  // the last position should be equal to batchSize.
  ICpuGpuVectorPtr sequenceStartPositions;

  // If NULL, each sequence has no subsequence.
  // Otherwise, its size should be #NumberOfSubSequences + 1.
  // The first position is always 0 and
  // the last position should be equal to batchSize.
  ICpuGpuVectorPtr subSequenceStartPositions;

  // dimension of sequence, stored only in CPU
  IVectorPtr cpuSequenceDims;

  int deviceId;            // the GPU device id which the argument in
  int allCount;            // the number of output layers using this argument
  mutable int valueCount;  // waiting this member when layer do forward
  mutable int gradCount;   // waiting this member when layer do backward
  mutable LockedCondition valueReadyCond;
  mutable LockedCondition gradReadyCond;

  int dataId;  // dataProvider id

  /* Increase the reference count of the argument. */
  void countIncrement() { allCount++; }

  int getAllCount() const { return allCount; }

  void waitValueReady() const {
    valueReadyCond.wait([this] { return (valueCount != 0); });

    std::lock_guard<std::mutex> guard(*valueReadyCond.mutex());
    valueCount--;
  }

  void notifyValueReady() const {
    valueReadyCond.notify_all([this] { valueCount = allCount; });
  }

  void waitGradReady() const {
    gradReadyCond.wait([this] { return (gradCount == allCount); });
    gradCount = 0;
  }

  void notifyGradReady() const {
    gradReadyCond.notify_all([this] { gradCount++; });
  }

  int64_t getBatchSize() const {
    if (value) return value->getHeight();
    if (ids) return ids->getSize();
    if (grad) return grad->getHeight();
    if (in) return in->getHeight();
    if (strs) return strs->size();
    return 0;
  }
  size_t getFrameHeight() const { return frameHeight; }
  size_t getFrameWidth() const { return frameWidth; }
  size_t getFrameDepth() const { return frameDepth; }
  void setFrameHeight(size_t h) { frameHeight = h; }
  void setFrameWidth(size_t w) { frameWidth = w; }
  void setFrameDepth(size_t d) { frameDepth = d; }

  int64_t getNumSequences() const {
    return sequenceStartPositions ? sequenceStartPositions->getSize() - 1
                                  : getBatchSize();
  }

  int64_t getNumSubSequences() const {
    return subSequenceStartPositions ? subSequenceStartPositions->getSize() - 1
                                     : getBatchSize();
  }

  bool hasSeq() const { return sequenceStartPositions != nullptr; }
  bool hasSubseq() const { return subSequenceStartPositions != nullptr; }

  const int* getCpuStartPositions() const {
    return hasSubseq() ? subSequenceStartPositions->getData(false)
                       : sequenceStartPositions->getData(false);
  }

  static inline real sum(const std::vector<Argument>& arguments) {
    real cost = 0;
    for (auto& arg : arguments) {
      if (arg.value) {
        SetDevice device(arg.deviceId);
        cost += arg.value->getSum();
      }
    }
    return cost;
  }

  /**
   * @brief (value, ids, grad, sequenceStartPositions) of output are subset of
   *        input. Note that, output share the same memory of input.
   *
   * @param input[in]       input
   * @param offset[in]      offset in terms of rows
   * @param height[in]      height of output.value
   * @param width[in]       width of output.value
   * @param useGpu[in]
   * @param trans[in]       whether input.value is transform
   * @param seqFlag[in]     whether input has sequenceStartPositions
   * @param seqStart[in]    offset of input.sequenceStartPositions
   * @param seqSize[in]     lenght of output.sequenceStartPositions
   */
  void subArgFrom(const Argument& input,
                  size_t offset,
                  size_t height,
                  size_t width,
                  bool useGpu,
                  bool trans = false,
                  bool seqFlag = false,
                  size_t seqStart = 0,
                  size_t seqSize = 0);
  /*
   * for sequence input:
   *   startSeq: the sequence id of start
   *   copySize: how many sequences need to copy
   *   return value: how many samples are copied
   * for non-sequence input:
   *   startSeq: the sample id of start
   *   copySize: how many samples need to copy
   *   return value: how many samples are copied
   * Note that when specifying the stream explicitly in this case,
   * synchronize should also be called somewhere after this function
   */
  int32_t resizeAndCopyFrom(const Argument& src,
                            int32_t startSeq,
                            int32_t copySize,
                            bool useGpu,
                            hl_stream_t stream);

  /*
   * same with the above function, except that the stream is
   * HPPL_STREAM_DEFAULT and synchronize is automatically called
   * inside it
   */
  int32_t resizeAndCopyFrom(const Argument& src,
                            int32_t startSeq,
                            int32_t copySize,
                            bool useGpu = FLAGS_use_gpu);

  void resizeAndCopyFrom(const Argument& src, bool useGpu, hl_stream_t stream);

  /*
   * same with the above function, except that the stream is
   * HPPL_STREAM_DEFAULT and synchronize is automatically called
   * inside it
   */
  void resizeAndCopyFrom(const Argument& src, bool useGpu = FLAGS_use_gpu);

  /*
    @brief Concatenate several arguments into one and put the result into it.
    @param args : a vector of argument, each element of which is a frame in a
    batch of sequences.
    @param selectRows : select several row of args to concatenate
    @param seqStartPos : sequence start positions in the final Argument
    @param hl_stream_t : cuda stream
    @param passTyoe : type of task, training or testing
   */
  void concat(const std::vector<Argument>& args,
              const std::vector<int>& selectRows,
              const std::vector<int>& seqStartPos,
              const std::vector<int>& copySize,
              bool useGpu,
              hl_stream_t stream,
              PassType passType);

  /*
    Concatenate several args into one and put the result into this.
   */
  void concat(const std::vector<Argument>& src,
              bool useGpu = FLAGS_use_gpu,
              hl_stream_t stream = HPPL_STREAM_DEFAULT,
              PassType passType = PASS_TEST);

  /*
   * split vector<Argument> to several vectors according to dataId
   */
  static void splitByDataId(const std::vector<Argument>& argus,
                            std::vector<std::vector<Argument>>* arguGroups);

  struct SeqInfo {
    // Equal to sequence length for sequence data
    // Equal to number of subsequences for subsequence data
    int topLevelLength;

    int seqStart;
    int seqId;

    // Equal to topLevelLength for sequence data
    // Equal to sum of the length of subsequences for subsequence data
    int subLevelLength;

    // Only used for subsequence data, start position of this sequence
    // is subSequenceStartPositions, i.e.
    // subSequenceStartPositions[subSeqStart] == seqStart
    int subSeqStart;
  };
  /*
    Get SeqInfo for each sequence of this argument
    Elements in *seqInfo are sorted by topLevelLength in descending order
  */
  void getSeqInfo(std::vector<SeqInfo>* segInfo) const;

  /*
   Check Whether sequenceStartPositions is subset of
   subSequenceStartPositions.
   */
  void checkSubset() const;

  /*
   sequence has sub-sequence degrades to a sequence.
   */
  void degradeSequence(const Argument& input);

  /*
   After pooling with stride n (n is smaller than sequence length),
   a long sequence will be shorten.
   This function is invalid for sequence having sub-sequence.
   */
  void poolSequenceWithStride(const Argument& input,
                              size_t stride,
                              ICpuGpuVectorPtr* stridePositions,
                              bool reversed = false);
  /**
   * @brief getValueString will return the argument's output in string. There
   * are several kinds of output. The keys of output dictionary are 'value',
   * 'id', 'sequence pos', 'sub-sequence pos'.
   * @param out [out]: the return values.
   */
  void getValueString(std::unordered_map<std::string, std::string>* out) const;

  /**
   * @brief printValueString will print the argument's output in order of
   * 'value', 'id', 'sequence pos', 'sub-sequence pos'.
   * @param stream: Output stream
   * @param prefix: line prefix for printing.
   */
  void printValueString(std::ostream& stream,
                        const std::string& prefix = "") const;

  /**
   * @brief reorganizeSeqInfo will reorganize sequenceStartPositions and
   * subSequenceStartPositions into a 2 dimensional arrary: reorganizedSeqInfo.
   *
   * @param seqStartPos: sequenceStartPositions of an Argument.
   * @param subSeqStartPos: subSequenceStartPositions of an Argument.
   * @param the reorganized sequence start position information.
   *
   * Examples:
   * seqStartPos: [0, 4, 15, 20, 28]
   * subSeqStartPos: [0, 3, 4, 5, 7, 10, 15, 20, 22, 23, 25, 28]
   * reorganizedSeqInfo:
   *   [
   *     [0,3,4],
   *     [4,5,7,10,15],
   *     [15,20],
   *     [20,22,23,25,28]
   *   ]
   */
  static void reorganizeSeqInfo(
      const ICpuGpuVectorPtr seqStartPos,
      const ICpuGpuVectorPtr subSeqStartPos,
      std::vector<std::vector<int>>& reorganizedSeqInfo);
};
  
typedef enum {
  /// The paddle original basic format
  PARAM_FORMAT_ORIGINAL = 0,

  /// See mkldnn_memory_format_t in
  /// https://github.com/01org/mkl-dnn/blob/master/include/mkldnn_types.h
  /// for a detailed description.
  /// 2D weights tensor in the format (output channels, input channels).
  PARAM_FORMAT_MKLDNN_OI,

  /// The total format items numbers
  PARAM_FORMAT_ITEMS,
} PARAM_FORMAT;

class SparsePrefetchRowCpuMatrix;

class Parameter;
typedef std::function<void(Parameter* param)> UpdateCallback;
typedef std::function<void(int paramId, Parameter* param)> ParamInitCallback;

class Parameter;
typedef std::shared_ptr<Parameter> ParameterPtr;

class Parameter {
public:
  Parameter(const ParameterConfig& config, bool useGpu, bool doInit = true);
  const std::string& getName() const { return config_.name(); }

  size_t getSize() const { return config_.size(); }

  bool isFullSize() const {
    if (bufs_[PARAMETER_VALUE]) {
      return this->getSize() == bufs_[PARAMETER_VALUE]->getSize();
    }
    return false;
  }

  inline bool useGpu() const { return useGpu_; }

  int getDeviceId() const { return deviceId_; }

  void setDevice(int deviceId) { deviceId_ = deviceId; }

  /// The id ranges from 0 to the_total_number_of_parameters - 1
  size_t getID() const { return config_.para_id(); }

  /// ID is a implict value created until neural network is built.
  void setID(size_t id) { config_.set_para_id(id); }

  bool isStatic() const { return config_.is_static(); }

  enum MatType {
    MAT_NORMAL,
    /// both value and grad are shared
    MAT_NORMAL_SHARED,

    /// Now used in BatchNorm in CPU mode
    MAT_VALUE_SHARED,

    /// sparse matrix, which has full size parameter
    MAT_SPARSE_ROW_IDS,
    /// sparse matrix, parameter size scale by sparse rates.
    MAT_SPARSE_ROW_AUTO_GROW,
    MAT_CACHE_ROW,
    MAT_SPARSE_ROW,

    /// sparse matrix for prefetching parameter from pserver
    MAT_SPARSE_ROW_PREFETCH,
    /// same as above, but parameter has full size for saving parameter in local
    MAT_SPARSE_ROW_PREFETCH_FULL_SIZE,
  };

  void enableSparseParameter() {
    if (config_.is_sparse()) {
      if (config_.format() == "csr") {
        size_t height = config_.dims(0);
        size_t nnz = config_.size();
        enableIntType(PARAMETER_ROWS, height + 1);
        enableIntType(PARAMETER_COLS, nnz);
        format_ = SPARSE_CSR;
      } else {
        size_t width = config_.dims(1);
        size_t nnz = config_.size();
        enableIntType(PARAMETER_COLS, width + 1);
        enableIntType(PARAMETER_ROWS, nnz);
        format_ = SPARSE_CSC;
      }
    }
  }

  /// allocate buffer for the give type
  void enableType(ParameterType type, MatType matType = MAT_NORMAL) {
    if (bufs_[type] || mats_[type]) {
      return;
    }
    SetDevice device(deviceId_);
    if (config_.dims_size() == 2) {
      if (matType == MAT_NORMAL || matType == MAT_NORMAL_SHARED ||
          matType == MAT_SPARSE_ROW_PREFETCH_FULL_SIZE ||
          matType == MAT_VALUE_SHARED || matType == MAT_SPARSE_ROW_IDS) {
        bufs_[type] = Vector::createParallelVector(config_.size(), useGpu_);
        bufs_[type]->zeroMem();
      } else {
        CHECK(isGradSparseUpdate());
      }
      if (config_.is_sparse() && type == PARAMETER_VALUE) {
        enableSparseParameter();
      }
      setMat(type, matType);
    } else {
      bufs_[type] = Vector::createParallelVector(config_.size(), useGpu_);
      bufs_[type]->zeroMem();
    }
  }

  void enableBufType(ParameterType type) {
    if (bufs_[type]) return;
    bufs_[type] = Vector::createParallelVector(config_.size(), useGpu_);
    bufs_[type]->zeroMem();
  }

  void enableIntType(ParameterType type, size_t intStoreSize = 0) {
    if (!intBufs_[type]) {
      SetDevice device(deviceId_);
      size_t size = intStoreSize ? intStoreSize : config_.size();
      intBufs_[type] = IVector::create(size, useGpu_);
      intBufs_[type]->zeroMem();
    }
  }

  void enableSharedType(ParameterType type,
                        VectorPtr vec,
                        MatrixPtr mat = nullptr) {
    if (!bufs_[type] && !mats_[type]) {
      bufs_[type] = vec;
      mats_[type] = mat;
    }
  }

  /// for batchGradientMachine: blockNum is number of partitions of the matrix.
  bool isGradShared(size_t* blockNum = NULL);

  bool isValueShared();

  // for AsgdSparseGradientMachine & SgdSparseGradientMachine:
  // and MultiGradientMachine
  bool isGradSparseUpdate() const;

  bool isSparseRemoteUpdate() const {
    return config_.sparse_remote_update() && !useGpu();
  }

  const ParameterConfig& getConfig() const { return config_; }

  ParameterConfig& getConfig() { return config_; }

  bool hasType(ParameterType pType) const {
    return bufs_[pType] || mats_[pType];
  }

  const VectorPtr& getBuf(ParameterType pType) const {
    return this->bufs_[pType];
  }

  const VectorPtr* getBufs() const { return bufs_; }

  const MatrixPtr& getMat(ParameterType pType) const { return mats_[pType]; }

  void setValueUpdated() { updated_ = true; }

  void clearValueUpdated() { updated_ = false; }

  bool isValueUpdated() const { return updated_; }

  /**
   * Save parameter value to a file
   */
  bool save(const std::string& filename) const;

  /**
   * Save parameter to ostream
   */
  bool save(std::ostream& s) const;

  /**
   * Load parameter value from a file
   */
  bool load(const std::string& filename);

  /**
   * Load parameter from istream
   */
  bool load(std::istream& is);

  void incShared() { sharedCount_++; }

  /**
   * After one of the parameter's gradient is merged
   * You should call this function to do some additional processing,
   */
  void incUpdate(const UpdateCallback& callbacks = NULL);

  void clearGradient() {
    auto& mat = getMat(PARAMETER_GRADIENT);
    if (mat) {
      // zeroMem will also clear rows for SparseRowCpuMatrix
      mat->zeroMem();
    } else {
      auto& gradBuf = getBuf(PARAMETER_GRADIENT);
      if (gradBuf) gradBuf->zeroMem();
    }
  }

  void initialize();

  /**
   * Initialize the value according to config_: initial_mean,
   * initial_std and initial_strategy.
   */
  void randomize();
  static void randomize(const VectorPtr& value, const ParameterConfig& config);

  /// Initialize the value to 0
  void zeroMem();

  /// file header structure
  struct Header {
    int32_t format;      // = PARAM_FORMAT
    uint32_t valueSize;  // = sizeof(real)
    uint64_t size;       // = getSize()
  };

  /**
   * @brief Is the header format supported.
   */
  static bool isHeaderFormatSupported(int32_t fmt) {
    return fmt < PARAM_FORMAT_ITEMS;
  }

  /**
   * @brief Get the format in header.
   */
  int getHeaderFormat() { return headerFormat_; }

  /**
   * @brief Set the format in header.
   */
  void setHeaderFormat(int32_t fmt) {
    CHECK(isHeaderFormatSupported(fmt)) << "Unsupported format version: "
                                        << fmt;
    headerFormat_ = fmt;
  }

  /**
   * @brief  Parameter Update Hook.
   *
   * The parameter's update hook before ParameterUpdater::updateImpl
   * It could modify gradient/momentum/etc here. Such as drop some gradient,
   * etc.
   */
  void updateHook() {
    for (auto& hook : updaterHooks_) {
      hook->update(this);
    }
  }

  /**
   * @brief  Initialize all updater hook.
   *
   * This method should be invoked in ParameterUpdater::init() only.
   */
  void initHook() {
    for (auto& hook : updaterHooks_) {
      hook->init(this);
    }
  }

protected:
  /**
   * @brief create matrix to matType.
   *
   * used by gradient machine which needs specify matrix type,
   * instead of creating in weights.cpp.
   *
   * @note  pType should be enabled already.
   */
  void setMat(ParameterType pType, int matType);

  bool isUpdatable() { return (updateCounter_ == sharedCount_); }

  void clearUpdate() { updateCounter_ = 0; }

protected:
  ParameterConfig config_;

  bool useGpu_;

  int deviceId_;

  /**
   * @brief bufs_ stores parameter value and gradient.
   *
   * Layer should use bufs_[PARAMETER_VALUE] to form weight matrix for
   * calculation and stores gradient to bufs_[PARAMETER_GRADIENT].
   */
  VectorPtr bufs_[NUM_PARAMETER_TYPES];

  /**
   * @brief Weight matrix for bufs_.
   *
   * It's helpfull when parameter shared by multi-layers.
   * Caller should check, if mats exist, do not create it again.
   */
  MatrixPtr mats_[NUM_PARAMETER_TYPES];

  /// Int vectors, used in some User defined parameter types
  IVectorPtr intBufs_[NUM_PARAMETER_TYPES];

  int sharedCount_;
  int updateCounter_;

  bool updated_;
  SparseFormat format_;

  /// The header format for saving or loading param
  int32_t headerFormat_;

  std::vector<std::shared_ptr<IParameterUpdaterHook>> updaterHooks_;

public:
  void setSharedCount(int cnt) { sharedCount_ = cnt; }
  int getSharedCount() { return sharedCount_; }

  bool isSparse() { return config_.is_sparse(); }
  SparseFormat getFormat() { return format_; }

  static const std::string kMissParameterFail;
  static const std::string kMissParameterRand;
  static const std::string kMissParameterZero;
};

typedef std::map<std::string, ParameterPtr> ParameterMap;


/**
 * Some member functions are set to const for two reasons:
 *
 * 1. For sparse update thread safe: update(), traverse callback(const this)
 *    may be called many times, each time one row, and these function
 *    can be called parallelly by multi worker, to speed up large block.
 *
 * 2. For predicate functions, needSpecialTraversal(), startCatchUpWith()
 *    may be called many times, should be no state change between calls.
 */
class ParameterOptimizer {
public:
  typedef std::function<void(
      const VectorPtr vecs[], const ParameterConfig& config, size_t sparseId)>
      TraverseCallback;

public:
  explicit ParameterOptimizer(const OptimizationConfig& optConfig)
      : applyDecay_(true),
        optConfig_(optConfig),
        parameterTypes_{PARAMETER_VALUE, PARAMETER_GRADIENT},
        learningRate_(optConfig.learning_rate()),
        learningRateScheduler_(LearningRateScheduler::create(optConfig)),
        pass_(0),
        firstTime_(true) {}

  real calcLearningRate(int64_t numSamplesProcessed, int64_t pass) {
    return learningRateScheduler_->calcLearningRate(numSamplesProcessed, pass);
  }

  virtual ~ParameterOptimizer() {}

  /**
   * For sparse update, optimizer can maintain numRows of timer(t0).
   * Some sparse optimizer depends on parameter config in functions
   * such as startBatch(). Optimizer can get it here. But notice that,
   * not all callers can pass config here, so the optimizer should check
   * config passed in is not null ptr.
   */
  virtual void init(size_t numRows, const ParameterConfig* config) {}

  virtual void startPass() {}
  virtual void finishPass() { ++pass_; }

  /// called by Trainer before forward() of a batch.
  virtual void startBatch(int64_t numSamplesProcessed) {
    (void)numSamplesProcessed;
  }

  /**
   * following hooks useful for sparse update,
   * because the traversal in block costs.
   * called by Trainer after update and before finishBatch
   * e.g. Trainer call like this:
   *
   * @code
   * startBatch();
   * if (dense) {
   *   update(blockVec);
   * } else {//sparse
   *   for (row : rows_in_block) {update(rowVec)}
   * }
   * auto callback = needSpecialTraversal();
   * if (callback) {
   *   // do traverse, maybe multi-thread
   *   if (dense) {
   *     callback();
   *   } else {//sparse
   *     for (row : all_rows_in_block) {callback();}
   *   }
   * }
   * finishBatch();
   * @endcode
   *
   * @return callback if need traverse,
   *         else return nullptr.
   *         It should be no state change.
   */
  virtual TraverseCallback needSpecialTraversal(
      const ParameterConfig& config) const {
    return nullptr;
  }

  /// called by Trainer after backward() of a batch
  virtual void finishBatch() {}

  /**
   * between startBatch() and finishBatch(), update() will be called
   * by the trainer multiple times, each time for updating one Parameter
   * with its gradient in PARAMETER_GRADIENT. sparseId is row id,
   * when sparseId set, update is sparse, each time one row.
   */
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& config,
                      size_t sparseId = -1LU) const = 0;

  /**
   * following hooks catch up with current time for sparse update,
   * In the beginning, call startCatchUpWith() and check return.
   * In the end, call finishCatchUpWith() to finish state.
   * callback do the actual works, can call many times for sparse data.
   * e.g. Trainer call like this:
   *
   * @code
   * auto callback = startCatchUpWith();
   * if (callback) {
   *   // do catch up with, maybe multi-thread
   *   if (dense) {
   *     callback();
   *   } else {//sparse
   *     for (row : rows_in_block) {callback();}
   *   }
   *   // finish catch up with, main thread
   *   finishCatchUpWith();
   * }
   * @endcode
   *
   * @return callback if need catch up with,
   *         else return nullptr.
   *         It should be no state change.
   */
  virtual TraverseCallback startCatchUpWith() const { return nullptr; }
  virtual void finishCatchUpWith() {}

  /**
   * following two hooks used by averager,
   * apply to final parameter value (PARAMETER_VALUE or PARAMETER_APPLY).
   *
   * restore() will restore orginal value if it apply to PARAMETER_VALUE.
   * Caller must ensure it's catched up with current time before apply.
   *
   * Use returned callback same way as callback returned by
   * ParameterOptimizer::needSpecialTraversal()
   */
  virtual TraverseCallback apply() { return nullptr; }
  virtual TraverseCallback restore() { return nullptr; }

  /// return the parameter types used by this updater
  const std::vector<ParameterType>& getParameterTypes() const {
    return parameterTypes_;
  }

  void addParameterType(ParameterType type) {
    for (auto t : parameterTypes_) {
      if (t == type) return;
    }
    parameterTypes_.push_back(type);
  }

  real getLearningRate() const { return learningRate_; }

  virtual void setNoDecay() { applyDecay_ = false; }

  static ParameterOptimizer* create(const OptimizationConfig& optConfig,
                                    bool inPserver = false);

protected:
  typedef std::vector<ParameterOptimizer::TraverseCallback> TraverseCallbackVec;

  static TraverseCallback composeCallbacks(
      const TraverseCallbackVec& callbacks) {
    if (callbacks.size() > 1LU) {
      return [callbacks](const VectorPtr vecs[],
                         const ParameterConfig& config,
                         size_t sparseId) {
        for (auto callback : callbacks) {
          callback(vecs, config, sparseId);
        }
      };
    }
    return (callbacks.size() == 1LU) ? callbacks[0] : nullptr;
  }

  bool applyDecay_;
  const OptimizationConfig& optConfig_;
  std::vector<ParameterType> parameterTypes_;

  /**
   * global learning rate, init value is opt_config.learning_rate,
   * sparse regularizer get this value per batch, after StartBatch() called
   * so, if lr change in StartBatch, please assign to learningRate_
   */
  real learningRate_;

  std::unique_ptr<LearningRateScheduler> learningRateScheduler_;
  int64_t pass_;  // current training pass (starting from 0)
  bool firstTime_;
};

class Weight {
private:
  MatrixPtr weight_;
  MatrixPtr weightGrad_;
  ParameterPtr parameter_;

public:
  Weight(size_t height, size_t width, ParameterPtr parameter);
  Weight(size_t height, size_t width, ParameterPtr parameter, size_t offset);

  const MatrixPtr& getW() { return weight_; }
  const MatrixPtr& getWGrad() { return weightGrad_; }
  const ParameterPtr& getParameterPtr();

  void incUpdate(const UpdateCallback& callback) {
    getParameterPtr()->incUpdate(callback);
  }

  void setParameterPtr(ParameterPtr param);
};

typedef std::vector<std::unique_ptr<Weight>> WeightList;


// Regularizer function for parameter, e.g. L1/L2
class Regularizer {
public:
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,  // learningrate from optimizer
                      int t0,             // last occurence time
                      int t) const = 0;   // current time
  virtual ~Regularizer() {}

  static Regularizer* get(const std::vector<ParameterType>& types,
                          const ParameterConfig& paraConfig);
};

// L1 Regularizer, |w|_1
class L1Regularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL1(learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate_l1() * (t - t0));
  }
};

// L1 Lr Regularizer
class L1LrRegularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL1(*vecs[PARAMETER_LEARNING_RATE],
                                   learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate_l1() * (t - t0));
  }
};

// L2 Regularizer, |w|_2^2
class L2Regularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL2(learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate() * (t - t0));
  }
};

// L2 Lr Regularizer
class L2LrRegularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL2(*vecs[PARAMETER_LEARNING_RATE],
                                   learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate() * (t - t0));
  }
};

// L1 + L2 Regularizer, |w|_1 + |w|_2^2
class L1L2Regularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL1(learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate_l1() * (t - t0));
    vecs[PARAMETER_VALUE]->applyL2(learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate() * (t - t0));
  }
};

// L1 + L2 Lr Regularizer
class L1L2LrRegularizer : public Regularizer {
  virtual void update(const VectorPtr vecs[],
                      const ParameterConfig& paraConfig,
                      real learningRate,
                      int t0,
                      int t) const {
    vecs[PARAMETER_VALUE]->applyL1(*vecs[PARAMETER_LEARNING_RATE],
                                   learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate_l1() * (t - t0));
    vecs[PARAMETER_VALUE]->applyL2(*vecs[PARAMETER_LEARNING_RATE],
                                   learningRate * paraConfig.learning_rate(),
                                   paraConfig.decay_rate() * (t - t0));
  }
};

}  // namespace mypaddle
}  // namespace bubblefs