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

// Paddle/paddle/gserver/dataproviders/DataProvider.h
// Paddle/paddle/gserver/dataproviders/DataProvider.cpp

#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

#include "utils/paddle_proto.h"
#include "utils/paddle_matrix.h"
#include "utils/paddle_sparse_matrix.h"
#include "utils/paddle_vector.h"
#include "utils/paddle_parameter.h"
#include "platform/paddle_locks.h"
#include "utils/paddle_queue.h"
#include "utils/paddle_string_util.h"
#include "platform/paddle_threadlocal.h"

namespace bubblefs {
namespace mypaddle {
/**
 * @def REGISTER_DATA_PROVIDER
 * @brief Macro for registering a data provider. The class type should contain
 *        a consturctor with parameter (DataConfig, bool).
 */
#define REGISTER_DATA_PROVIDER(__type_name, __class_name)                \
  static InitFunction __reg_type_##__type_name([]() {                    \
    DataProvider::registrar_.registerClass(                              \
        #__type_name,                                                    \
        [](DataConfig conf, ModelConfig, bool useGpu) -> DataProvider* { \
          DataProvider* dp = new __class_name(conf, useGpu);             \
          return dp;                                                     \
        });                                                              \
  })

/**
 * @def REGISTER_DATA_PROVIDER_EX
 * @brief Macro for registering a data provider, which contains a constructor
 *        with parameter (DataConfig, ModelConfig, bool).
 */
#define REGISTER_DATA_PROVIDER_EX(__type_name, __class_name)            \
  static InitFunction __reg_type_##__type_name([] {                     \
    DataProvider::registrar_.registerClass<__class_name>(#__type_name); \
  })

class DataBatch;
class BufferBatch;
typedef std::shared_ptr<DataBatch> DataBatchPtr;
typedef std::shared_ptr<BufferBatch> BufferBatchPtr;
/**
 * @brief Data for batch training a neural network
 */
class DataBatch {
public:
  DataBatch() : size_(0) { data_.clear(); }
  /**
   * @brief Get batch size
   * @return batch size
   */
  int64_t getSize() const { return size_; }
  /**
   * @brief Get num of sequences of sequence data
   * @return num of sequences
   */
  int64_t getNumSequences() const {
    if (data_.empty()) return size_;
    return data_[0].sequenceStartPositions
               ? data_[0].sequenceStartPositions->getSize() - 1
               : size_;
  }
  /**
   * @brief Set batch size
   * @param[in] size size
   */
  void setSize(int64_t size) { size_ = size; }
  /**
   * @brief Get size of argument vector
   * @return size of argument vector
   * @note For usual supervised learning, input data and label is needed,
   * then there will be two argument.
   */
  int64_t getNumStreams() const { return data_.size(); }

  /**
   * @brief Get a argument with index i
   * @param[in] i index in argument vector
   * @return a argument with index i
   */
  const Argument& getStream(int i) const { return data_[i]; }
  /**
   * @brief Get all argument
   * @return an argument vector
   */
  std::vector<Argument>& getStreams() { return data_; }
  /**
   * @brief Get all argument const
   * @return an argument vector
   */
  std::vector<Argument> getStreams() const { return data_; }
  /**
   * @brief Clear DataBatch
   */
  void clear() {
    data_.clear();
    size_ = 0;
  }

  /**
   * @brief Append data to DataBatch
   * @param[in] data  matrix data
   * @note The order in which each data stream is appended must match the order
   * specified in stream_names of DataConfig. The stream_names can be obtained
   * using DataProvider::getStreamNames().
   */
  void appendData(MatrixPtr data) {
    Argument argu;
    argu.value = data;
    data_.push_back(argu);
  }

  /**
   * @brief Append sequence data to DataBatch
   * @param[in] data                      matrix data
   * @param[in] sequenceStartPositions    sequence data
   * @note The order in which each data stream is appended must match the order
   * specified in stream_names of DataConfig. The stream_names can be obtained
   * using DataProvider::getStreamNames().
   */
  void appendData(const MatrixPtr& data,
                  const ICpuGpuVectorPtr& sequenceStartPositions) {
    Argument argu;
    argu.value = data;
    argu.sequenceStartPositions = sequenceStartPositions;
    data_.push_back(argu);
  }
  /**
   * @brief Append label data
   * @param[in]  label    label data
   * @param[in]  value    matrix data, default null
   */
  void appendLabel(IVectorPtr label, MatrixPtr value = nullptr) {
    Argument argu;
    argu.ids = label;
    argu.value = value;
    data_.push_back(argu);
  }

  /*
   * @brief Append argument
   * @param[in]  argus   DataBatch.getStreams()
   * @param[in]  size    DataBatch.getSize()
   * @param[in]  dataId  sub dataprovider id (in MultiDataProvider)
   */
  void appendArguments(const std::vector<Argument>& argus,
                       int size,
                       int dataId) {
    size_ += size;
    for (const auto& argu : argus) {
      data_.push_back(argu);
      data_.back().dataId = dataId;
    }
  }

protected:
  /**
   * @brief batch size
   */
  int64_t size_;
  /**
   * @brief A batch data consist of a Argument vector,
   * An argument corresponds to a type of input data.
   */
  std::vector<Argument> data_;
};

class BufferBatch {
public:
  BufferBatch() {
    hlStream_ = HPPL_STREAM_DEFAULT;
    hlEvent_ = NULL;
    batchData_ = NULL;
  }
  ~BufferBatch() {
    if (hlEvent_) {
      hl_destroy_event(hlEvent_);
      hlEvent_ = NULL;
    }
    delete batchData_;
    batchData_ = NULL;
  }

  void setDataBatch(DataBatch* batchData) { batchData_ = batchData; }
  DataBatch* getDataBatch() { return batchData_; }

  void setCuStream(hl_stream_t stream) { hlStream_ = stream; }
  hl_stream_t getCuStream() const { return hlStream_; }

  void setCuEvent(hl_event_t event) { hlEvent_ = event; }

  hl_event_t getCuEvent() const { return hlEvent_; }

  void createCuEvent() {
    if (!hlEvent_) {
      hlStream_ = HPPL_STREAM_1;
      hl_create_event(&hlEvent_);
    }
  }

  void syncEvent() {
    if (hlEvent_) {
      hl_stream_wait_event(hlStream_, hlEvent_);
    }
  }

  void swap(BufferBatch* bufBatch);
  void clone(DataBatch* srcBatch, bool useGpu);

protected:
  DataBatch* batchData_;
  hl_stream_t hlStream_;
  hl_event_t hlEvent_;
};

class DataProvider;
typedef std::shared_ptr<DataProvider> DataProviderPtr;

typedef Queue<BufferBatch*> BufferBatchQueue;

class DoubleBuffer {
public:
  DoubleBuffer(DataProvider* dataPool, bool useGpu, int64_t batchSize = 0);
  virtual ~DoubleBuffer();
  void removeOneBatch(DataBatch* dataBatch);

  void setBatchSize(int64_t newBatchSize) { batchSize_ = newBatchSize; }

  int64_t getBatchSize() { return batchSize_; }

  void startAsyncLoad();
  void finishAsyncLoad() {
    stopping_ = true;
    taskReadySem_.post();
    if (asyncLoader_) {
      asyncLoader_->join();
    }
  }

  void setPending(bool pending) { pending_ = pending; }

protected:
  virtual void asyncLoadBatch();
  void insertOneBatch(DataBatch* batch);

  DataProvider* dataPool_;
  bool useGpu_;
  int32_t batchSize_;
  ThreadLocal<BufferBatchPtr> usingBatch_;
  BufferBatchQueue* dataQueue_;
  BufferBatchQueue* bufferQueue_;
  std::unique_ptr<std::thread> asyncLoader_;
  Semaphore taskReadySem_;
  bool stopping_;
  bool pending_;
};

/**
 * @brief Base class for DataProvider, which supplies data for training
 * @note It can supplies multiple streams of data.
 * For typical supervised training, there are two streams:
 * one is for input, one is for label.
 */
class DataProvider {
public:
  static ClassRegistrar<DataProvider, DataConfig, ModelConfig, bool> registrar_;
  static DataProvider* create(const DataConfig& config,
                              const ModelConfig& modelConfig,
                              bool useGpu = FLAGS_use_gpu);

  /**
   * @brief create only used for unittest.
   */
  inline static DataProvider* create(const DataConfig& config,
                                     bool useGpu = FLAGS_use_gpu) {
    return create(config, ModelConfig(), useGpu);
  }

  DataProvider(const DataConfig& config, bool useGpu)
      : config_(config),
        skipShuffle_(false),
        usageRatio_(config.usage_ratio()),
        useGpu_(useGpu) {
    if (config_.async_load_data()) {
      initAsyncLoader();
    }
  }
  virtual ~DataProvider() {}

  const DataConfig& getConfig() const { return config_; }

  void setSkipShuffle() { skipShuffle_ = true; }

  /**
   * @brief Get next batch of training samples
   * @param[in]    size    size of training samples to get
   * @param[out]   batch   a batch of training samples
   * @return actual size of obtained training samples
   */
  int64_t getNextBatch(int64_t size, DataBatch* batch);

  /**
   * @brief Shuffle the data set
   */
  virtual void shuffle() = 0;

  /**
   * @brief reset all the value of index
   * @note reset() must be called before any calls to getNextBatch()
   * IMPORTANT: subclass reset() should always call the base class reset()
   * at the end of the function
   */
  virtual void reset() {
    if (doubleBuffer_ != nullptr) {
      doubleBuffer_->startAsyncLoad();
    }
  }

  /**
   * @brief Get the size of training samples
   * @return the number of training samples in the data set.
   * @note return -1 to indicate unlimited number of samples.
   */
  virtual int64_t getSize() = 0;

  /**
   * @brief Get next batch training samples internally
   * @param[in]    size      size of training samples to get
   * @param[out]   batch     a batch of training samples
   * @return actual size of obtained training samples
   */
  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch) = 0;

protected:
  DataConfig config_;
  bool skipShuffle_;
  float usageRatio_;
  bool useGpu_;
  std::unique_ptr<DoubleBuffer> doubleBuffer_;
  ThreadLocal<std::vector<MatrixPtr>> constantSlots_;
  /**
   * @@brief Get next batch training samples from buffer
   * @param[in]    size      size of training samples to get
   * @param[out]   batch     a batch of training samples
   * @return actual size of obtained training samples
   */
  int64_t getNextBatchFromBuffer(int64_t size, DataBatch* batch);

  void initAsyncLoader();
};

/**
 * A data provider which does nothing. It only serves as providing
 * necessary configurations such as stream_names
 */
class DummyDataProvider : public DataProvider {
public:
  DummyDataProvider(const DataConfig& config, bool useGpu)
      : DataProvider(config, useGpu) {}
  virtual void shuffle() {}
  virtual void reset() { DataProvider::reset(); }
  virtual int64_t getSize() { return 0; }
  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch) {
    (void)size;
    (void)batch;
    return 0;
  }
};

/**
 * Data provider for one input and one integer label.
 */
class SimpleDataProviderBase : public DataProvider {
protected:
  /// sample feature dimension
  int64_t sampleDim_;
  /// the number of samples
  int64_t bufferCapacity_;
  int64_t sampleNumInBuf_;
  /// next item to read in buffer
  int64_t nextItemIndex_;
  /// some user defined info for validation
  bool withInfo_;

  /// data buffer: bufferCapacity_ * nDataDim_
  CpuMatrixPtr hInputDataBuf_;

  /// label buffer:bufferCapacity_ * 1
  CpuIVectorPtr hInputLabelBuf_;

  /// info buffer:bufferCapacity_ * 1
  CpuIVectorPtr hInputInfoBuf_;

  ThreadLocal<MatrixPtr> dataBatch_;
  ThreadLocal<IVectorPtr> labelBatch_;
  ThreadLocal<IVectorPtr> infoBatch_;

  RWLock lock_;

public:
  SimpleDataProviderBase(const DataConfig& config, bool useGpu, bool withInfo);
  ~SimpleDataProviderBase() {}

  void shuffle();

  virtual void reset();

  virtual int64_t getSize();

  virtual int64_t getNextBatchInternal(int64_t size, DataBatch* batch);

  /// return the number of samples in the buffer
  int64_t fillBuffer();

protected:
  /**
   * @brief Fill at most size samples into data and label.
   *
   * Each input is stored in contiguous memory locations in data.
   *
   * data[n * sampleDim_] .. data[n * sampleDim_ + sampleDim_ - 1] is for
   * the input of the n-th sample.
   *
   * label[n] is the label for the n-th sample.
   */
  virtual int64_t fillBufferImp(real* data,
                                int* label,
                                int* info,
                                int64_t size) = 0;
};

class SimpleDataProvider : public SimpleDataProviderBase {
public:
  SimpleDataProvider(const DataConfig& config, bool useGpu);
  ~SimpleDataProvider();
  virtual void reset();

protected:
  void loadData(const std::string& fileName);
  void loadDataFile(const std::string& fileName);
  virtual int64_t fillBufferImp(real* data,
                                int* label,
                                int* info,
                                int64_t size);

protected:
  size_t currentSampleIndex_;
  std::vector<int> labels_;
  std::vector<real> data_;
};

void BufferBatch::swap(BufferBatch* bufBatch) {
  DataBatch* batchData = bufBatch->getDataBatch();
  hl_event_t hlEvent = bufBatch->getCuEvent();
  hl_stream_t hlStream = bufBatch->getCuStream();
  bufBatch->setDataBatch(batchData_);
  bufBatch->setCuStream(hlStream_);
  bufBatch->setCuEvent(hlEvent_);

  batchData_ = batchData;
  hlEvent_ = hlEvent;
  hlStream_ = hlStream;
}

void BufferBatch::clone(DataBatch* srcBatch, bool useGpu) {
  if (batchData_ == NULL) {
    batchData_ = new DataBatch();
  }
  std::vector<Argument>& destData = batchData_->getStreams();
  int numStreams = srcBatch->getNumStreams();
  destData.resize(numStreams);
  batchData_->setSize(srcBatch->getSize());
  if (useGpu) {
    createCuEvent();
  }

  for (int i = 0; i < numStreams; i++) {
    destData[i].resizeAndCopyFrom(srcBatch->getStream(i), useGpu, hlStream_);
  }
  if (useGpu) {
    hl_stream_record_event(hlStream_, hlEvent_);
  }
}

DoubleBuffer::DoubleBuffer(DataProvider* dataPool,
                           bool useGpu,
                           int64_t batchSize) {
  batchSize_ = batchSize;
  dataPool_ = dataPool;
  useGpu_ = useGpu;
  dataQueue_ = new BufferBatchQueue();
  bufferQueue_ = new BufferBatchQueue();

  // insert a empty buffer
  bufferQueue_->enqueue(new BufferBatch());
  stopping_ = false;
  pending_ = true;
}

DoubleBuffer::~DoubleBuffer() {
  finishAsyncLoad();
  while (dataQueue_->size()) {
    BufferBatch* dataBtch = dataQueue_->dequeue();
    delete dataBtch;
    dataBtch = NULL;
  }
  while (bufferQueue_->size()) {
    BufferBatch* bufBtch = bufferQueue_->dequeue();
    delete bufBtch;
    bufBtch = NULL;
  }
  delete dataQueue_;
  dataQueue_ = NULL;
  delete bufferQueue_;
  bufferQueue_ = NULL;
}

void DoubleBuffer::removeOneBatch(DataBatch* dataBatch) {
  // get data
  BufferBatch* batch = dataQueue_->dequeue();
  batch->syncEvent();  // when use GPU, need synchronized with the cuEvent
  *dataBatch = *(batch->getDataBatch());

  // push anothor buffer
  if (*usingBatch_ == nullptr) {
    *usingBatch_ = std::make_shared<BufferBatch>();
  }

  // Mark the using-batch
  batch->swap((*usingBatch_).get());
  bufferQueue_->enqueue(batch);

  if (0 == dataBatch->getSize()) {
    setPending(true);
  }
}

void DoubleBuffer::insertOneBatch(DataBatch* batch) {
  while (!bufferQueue_->waitNotEmptyFor(2 /* seconds */)) {  // time out
    if (stopping_) return;
  }
  BufferBatch* bufBatch = bufferQueue_->dequeue();
  // clone and copy the data from an Threadlocal Variable
  bufBatch->clone(batch, useGpu_);
  dataQueue_->enqueue(bufBatch);
}

void DoubleBuffer::asyncLoadBatch() {
  int64_t actualSize = 0;
  if (useGpu_) {
    hl_set_device(FLAGS_gpu_id);
  }
  setPending(false);

  while (true) {
    taskReadySem_.wait();
    if (stopping_) break;

    while (batchSize_ == 0 && !stopping_) {
      usleep(5);
    }
    if (stopping_) break;

    do {
      DataBatch newBatch;
      {
        REGISTER_TIMER("getNextBatchInternal");
        actualSize = dataPool_->getNextBatchInternal(batchSize_, &newBatch);
      }
      insertOneBatch(&newBatch);
    } while (actualSize > 0 && !stopping_);
  }
}

void DoubleBuffer::startAsyncLoad() {
  if (asyncLoader_ == nullptr) {
    asyncLoader_.reset(new std::thread([this]() { this->asyncLoadBatch(); }));
  }
  taskReadySem_.post();
}

ClassRegistrar<DataProvider, DataConfig, ModelConfig, bool>
    DataProvider::registrar_;

DataProvider* DataProvider::create(const DataConfig& config,
                                   const ModelConfig& modelConfig,
                                   bool useGpu) {
  return registrar_.createByType(config.type(), config, modelConfig, useGpu);
}

REGISTER_DATA_PROVIDER(simple, SimpleDataProvider);
REGISTER_DATA_PROVIDER(dummy, DummyDataProvider);

int64_t DataProvider::getNextBatch(int64_t size, DataBatch* batch) {
  int64_t batchSize = doubleBuffer_ ? getNextBatchFromBuffer(size, batch)
                                    : getNextBatchInternal(size, batch);

  if (!batchSize) return 0;

  if (!config_.constant_slots_size()) return batchSize;

  auto& constantSlots = *constantSlots_;
  constantSlots.resize(config_.constant_slots_size());

  for (int i = 0; i < config_.constant_slots_size(); ++i) {
    MemoryHandlePtr handle =
        constantSlots[i] ? constantSlots[i]->getMemoryHandle() : nullptr;
    Matrix::resizeOrCreate(constantSlots[i],
                           batchSize,
                           1,         // = width
                           false,     // = trans
                           useGpu_);  // = useGpu
    if (handle != constantSlots[i]->getMemoryHandle()) {
      // memory buf was reallocated. We need to initialize the value
      constantSlots[i]->assign(config_.constant_slots(i));
    }
    batch->appendData(constantSlots[i],
                      batch->getStream(0).sequenceStartPositions);
  }

  return batchSize;
}

int64_t DataProvider::getNextBatchFromBuffer(int64_t size, DataBatch* batch) {
  CHECK(doubleBuffer_ != nullptr);

  if (doubleBuffer_->getBatchSize() != size) {
    doubleBuffer_->setBatchSize(size);
  }

  doubleBuffer_->removeOneBatch(batch);
  return batch->getSize();
}

void DataProvider::initAsyncLoader() {
  if (doubleBuffer_ == nullptr) {
    doubleBuffer_.reset(new DoubleBuffer(this, useGpu_));
  }
  useGpu_ = false;  // Avoid D2D copy, it will delay the computing performance
}

SimpleDataProviderBase::SimpleDataProviderBase(const DataConfig& config,
                                               bool useGpu,
                                               bool withInfo)
    : DataProvider(config, useGpu) {
  /* initialize the size of a sample, and the buffer */
  sampleDim_ = config_.feat_dim() * (2 * config_.context_len() + 1);
  bufferCapacity_ = config_.buffer_capacity();
  withInfo_ = withInfo;
  sampleNumInBuf_ = 0;
  nextItemIndex_ = 0;

  /* malloc buffer in cpu */
  hInputDataBuf_ = std::make_shared<CpuMatrix>(bufferCapacity_, sampleDim_);
  hInputLabelBuf_ = std::make_shared<CpuIVector>(bufferCapacity_);
  hInputInfoBuf_ = std::make_shared<CpuIVector>(bufferCapacity_);
}

void SimpleDataProviderBase::shuffle() {
  int i, t;
  int len = sampleNumInBuf_;
  std::vector<real> temp(sampleDim_);
  real* data = hInputDataBuf_->getData();
  int* label = hInputLabelBuf_->getData();
  int* info = hInputInfoBuf_->getData();
  int sampleSz = sizeof(real) * sampleDim_;
  for (i = 0; i < len; i++) {
    int randNum = rand();  // NOLINT TODO(yuyang18): Use rand_r instead?
    t = randNum % (len - i) + i;
    // swap
    if (i != t) {
      // swap data
      memcpy(&temp[0], &data[i * sampleDim_], sampleSz);
      memcpy(&data[i * sampleDim_], &data[t * sampleDim_], sampleSz);
      memcpy(&data[t * sampleDim_], &temp[0], sampleSz);
      std::swap(label[i], label[t]);
      if (withInfo_) {
        std::swap(info[i], info[t]);
      }
    }
  }
}

int64_t SimpleDataProviderBase::getNextBatchInternal(int64_t size,
                                                     DataBatch* batch) {
  CHECK(batch != NULL);
  batch->clear();

  int64_t startIndex;
  int64_t cpySize;

  std::lock_guard<RWLock> guard(lock_);
  if (sampleNumInBuf_ - nextItemIndex_ < size) {
    int64_t n = fillBuffer();
    VLOG(1) << "fillBuffer return " << n << " samples.\n";
  }

  startIndex = nextItemIndex_;
  cpySize = std::min(size, sampleNumInBuf_ - nextItemIndex_);
  nextItemIndex_ += cpySize;

  if (cpySize > 0) {
    real* data = hInputDataBuf_->getData() + startIndex * sampleDim_;
    int* label = hInputLabelBuf_->getData() + startIndex;
    int* info = hInputInfoBuf_->getData() + startIndex;

    MatrixPtr& dataBatch = *dataBatch_;     // get the thread local object
    IVectorPtr& labelBatch = *labelBatch_;  // get the thread local object
    IVectorPtr& infoBatch = *infoBatch_;    // get the thread local object
    if (!dataBatch) {
      dataBatch = Matrix::create(cpySize, sampleDim_, false, useGpu_);
      labelBatch = IVector::create(cpySize, useGpu_);
      if (withInfo_) {
        infoBatch = IVector::create(cpySize, 0);
      }
    } else {
      dataBatch->resize(cpySize, sampleDim_);
      labelBatch->resize(cpySize);
      if (withInfo_) {
        infoBatch->resize(cpySize);
      }
    }
    dataBatch->copyFrom(data, cpySize * sampleDim_);
    labelBatch->copyFrom(label, cpySize);
    batch->appendData(dataBatch);
    batch->appendLabel(labelBatch);
    if (withInfo_) {
      infoBatch->copyFrom(info, cpySize);
      batch->appendLabel(infoBatch);
    }
  }

  batch->setSize(cpySize);
  return cpySize;
}

void SimpleDataProviderBase::reset() {
  sampleNumInBuf_ = 0;
  nextItemIndex_ = 0;
  DataProvider::reset();
}

int64_t SimpleDataProviderBase::getSize() {
  LOG(FATAL) << "Currently, not implemented";
  return 0;
}

int64_t SimpleDataProviderBase::fillBuffer() {
  int64_t n = sampleNumInBuf_ - nextItemIndex_;

  /* flash the remaining data to the beginning of the buffer */
  if (n > 0) {
    hInputDataBuf_->copyFrom(
        hInputDataBuf_->getData() + nextItemIndex_ * sampleDim_,
        n * sampleDim_);
    hInputLabelBuf_->copyFrom(hInputLabelBuf_->getData() + nextItemIndex_, n);
    if (withInfo_) {
      hInputInfoBuf_->copyFrom(hInputInfoBuf_->getData() + nextItemIndex_, n);
    }
  }

  sampleNumInBuf_ =
      n + fillBufferImp(hInputDataBuf_->getData() + n * sampleDim_,
                        hInputLabelBuf_->getData() + n,
                        hInputInfoBuf_->getData() + n,
                        bufferCapacity_ - n);

  /* for stachastic gradient training */
  if (!skipShuffle_) {
    shuffle();
  }

  nextItemIndex_ = 0;

  return sampleNumInBuf_;
}

SimpleDataProvider::SimpleDataProvider(const DataConfig& config, bool useGpu)
    : SimpleDataProviderBase(config, useGpu, /* withInfo= */ false),
      currentSampleIndex_(0) {
  loadData(config_.files());
}

SimpleDataProvider::~SimpleDataProvider() {}

int64_t SimpleDataProvider::fillBufferImp(real* data,
                                          int* label,
                                          int* info,
                                          int64_t size) {
  (void)info;
  int64_t n = std::min<int64_t>(labels_.size() - currentSampleIndex_, size);
  memcpy(data,
         &data_[currentSampleIndex_ * sampleDim_],
         n * sampleDim_ * sizeof(real));
  memcpy(label, &labels_[currentSampleIndex_], sizeof(int) * n);
  currentSampleIndex_ += n;

  return n;
}

void SimpleDataProvider::reset() {
  currentSampleIndex_ = 0;
  SimpleDataProviderBase::reset();
}

void SimpleDataProvider::loadData(const std::string& fileName) {
  std::ifstream is(fileName);
  CHECK(is) << "Fail to open " << fileName;
  std::string line;
  while (is) {
    if (!getline(is, line)) break;
    LOG(INFO) << "load data file " << line;
    loadDataFile(line);
  }
  LOG(INFO) << "read done, num of instance=" << labels_.size()
            << " data size=" << data_.size();
}

void SimpleDataProvider::loadDataFile(const std::string& fileName) {
  std::ifstream is(fileName);
  std::string line;
  std::vector<std::string> pieces;
  while (is) {
    if (!getline(is, line)) break;
    str::split(line, ' ', &pieces);
    CHECK_EQ((uint64_t)(sampleDim_ + 1), pieces.size())
        << " Dimension mismatch, " << pieces.size() - 1 << " in " << fileName
        << " " << sampleDim_ << " from config";
    labels_.push_back(atoi(pieces[0].c_str()));
    for (int i = 0; i < sampleDim_; ++i) {
      data_.push_back(atof(pieces[i + 1].c_str()));
    }
  }
}

}  // namespace mypaddle
}  // namespace bubblefs