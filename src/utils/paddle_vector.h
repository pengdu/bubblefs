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

// Paddle/paddle/math/Vector.h
// Paddle/paddle/math/Vector.cpp

#pragma once

#include <cmath>
#include <memory>

#include "utils/paddle_base_matrix.h"
#include "utils/paddle_memory_handle.h"
#include "utils/paddle_thread.h"
#include "platform/paddle_threadlocal.h"

#include "hl_gpu.h"
#include "hl_matrix.h"
#include "hl_table_apply.h"

namespace bubblefs {
namespace mypaddle {

template <class T>
class GpuVectorT;
template <class T>
class CpuVectorT;

template <class T>
class BaseVector;

class SyncThreadPool;

class Matrix;

template <class T>
class BaseVector : public BaseMatrixT<T> {
public:
  BaseVector(size_t size, T* data, bool useGpu)
      : BaseMatrixT<T>(1, size, data, false, useGpu), size_(this->width_) {}

  ~BaseVector() {}

protected:
  size_t& size_;
};

/**
 * Copy or assignemnt constructor will share the data as opposed to making a
 * copy of the original data. To make a copy of the orinal data, use copyFrom()
 * instead.
 */
template <class T>
class VectorT : public BaseVector<T> {
protected:
  VectorT(size_t size, MemoryHandlePtr memoryHandle, size_t offset, bool useGpu)
      : BaseVector<T>(size,
                      reinterpret_cast<T*>(memoryHandle->getBuf()) + offset,
                      useGpu) {
    memoryHandle_ = memoryHandle;
  }

  // data is still owned by the caller.
  // data should be valid during the life of this vector.
  // Caller is responsible for release the memory.
  VectorT(size_t size, T* data, bool useGpu)
      : BaseVector<T>(size, data, useGpu) {}

public:
  virtual ~VectorT() {}

  static std::shared_ptr<VectorT<T>> create(size_t size, bool useGpu);

  static std::shared_ptr<VectorT<T>> create(T* data, size_t size, bool useGpu);

  static std::shared_ptr<VectorT<T>> create(size_t size,
                                            MemoryHandlePtr memoryHandle,
                                            size_t offset = 0);

  // owner can set SyncThreadPool,
  // if not set, will use globalSyncThreadPool,
  // which can be used in main thread only.
  static std::shared_ptr<VectorT<T>> createParallelVector(
      size_t size, bool useGpu, SyncThreadPool* pool = nullptr);

  size_t getSize() const { return this->size_; }
  const T* getData() const { return this->data_; }
  T* getData() { return this->data_; }

  virtual void zeroMem() = 0;
  // set all elements to value
  virtual void reset(const T& value) = 0;
  // fill data by 0, 1, 2, ...
  virtual void fillSequence() = 0;

  MemoryHandlePtr getMemoryHandle() const { return memoryHandle_; }

  /**
   * resizing to a big vector will not preserve old values.
   */
  void resize(size_t newSize) {
    if (!memoryHandle_ || newSize * sizeof(T) > memoryHandle_->getAllocSize()) {
      memoryHandle_ = newMemory(newSize * sizeof(T));
      this->data_ = reinterpret_cast<T*>(memoryHandle_->getBuf());
    }
    this->size_ = newSize;
  }

  static void resizeOrCreate(std::shared_ptr<VectorT<T>>& vec,
                             size_t size,
                             bool useGpu) {
    if (vec) {
      vec->resize(size);
    } else {
      vec = create(size, useGpu);
    }
  }

  virtual MemoryHandlePtr newMemory(size_t size) = 0;

  /**
   * form sub vector from *src*, shallow copy
   */
  void subVecFrom(const VectorT<T>& src, size_t start, size_t size) {
    CHECK_EQ(BaseVector<T>::useGpu_, src.useGpu_);
    CHECK_LT(start, src.size_);
    CHECK_LE(start + size, src.size_);

    BaseVector<T>::size_ = size;
    BaseVector<T>::data_ = const_cast<T*>(src.data_) + start;
  }

  std::shared_ptr<VectorT<T>> subVec(size_t start, size_t size) {
    CHECK_LE(start + size, static_cast<size_t>(getSize()));
    return VectorT<T>::create(getData() + start, size, BaseVector<T>::useGpu_);
  }

  /**
   * form sub vector from *src*, shallow copy
   */
  void subVecFrom(const T* src, size_t start, size_t size) {
    BaseVector<T>::size_ = size;
    BaseVector<T>::data_ = const_cast<T*>(src) + start;
  }

  /**
   * form sub vector from *src*, shallow copy
   * in *interval* [interval.first, interval.second)
   */
  void subVecFrom(const VectorT<T>& src, std::pair<size_t, size_t> interval) {
    subVecFrom(src, interval.first, interval.second - interval.first);
  }

  /**
   * convert the vector to a sparse one_hot matrix of width idRange
   * only applies to IVector
   */
  std::shared_ptr<Matrix> toOneHotSparseMatrix(size_t idRange, bool useGpu);

  /**
   * @brief cast vector of "real" elements to "int" elements.
   *
   * @note: float -> int must be casted, or you'll get wrong data.
   */
  std::shared_ptr<VectorT<int>> castToInt();

  /**
   * This function will crash if the size of src and dest is different.
   */
  virtual void copyFrom(const VectorT<T>& src) = 0;

  /**
   * If GpuVector, this function is an asynchronous interface,
   * will push the copy-task to the specifed-stream and return immediately.
   *
   * If CpuVector, this function is an synchronous interface,
   * same as the copyFrom(const VectorT<T>& src).
   */
  virtual void copyFrom(const VectorT<T>& src, hl_stream_t stream) = 0;

  /**
   * copy size elements from src
   *
   * If this is GpuVector, src can be cpu or gpu memory
   *
   * If this is CpuVector, src is assumed to be cpu memory
   */
  virtual void copyFrom(const T* src, size_t size) = 0;

  /**
   * copy size elements from src
   *
   * If this is GpuVector, src can be cpu or gpu memory
   *
   * If this is CpuVector, src is assumed to be cpu memory,
   */
  virtual void copyFrom(const T* src, size_t size, hl_stream_t stream) = 0;

  /**
   * exec a func in single/multi thread
   */
  virtual void exec(SyncThreadPool::JobFunc func) { func(0, 1); }

  /// Get the buffer point with beginPos
  virtual T* getPoint(const uint64_t beginPos) = 0;

  /// Get the value for the i'th element
  virtual T getElement(size_t i) const = 0;
  virtual void setElement(size_t i, const T& value) = 0;

  //----------  math operations ----------------

  // sum of the absolute value of each elements
  virtual T getAbsSum() = 0;

  virtual T getSum() = 0;
  virtual T getMax() = 0;
  virtual T getAbsMax() = 0;
  virtual T getMin() = 0;

  /// element-wise calc:  this = (b == value)
  virtual void isEqualTo(const VectorT<T>& b, const T& value) = 0;

  /// select elements indexed by *ids* from vector *src*
  virtual void selectFrom(const VectorT<T>& src, const VectorT<int>& ids) = 0;

  enum HistogramType {
    HISTOGRAM_EXPONENT = 0,
  };

  /**
   * @brief  print histogram of vector values
   *
   * @note   only exponent histogram supported currently
   */
  virtual void histogram(std::ostream& os, int type = HISTOGRAM_EXPONENT) = 0;

  /// generate uniform random value for each element
  virtual void rand() = 0;
  /**
   * generate uniform random value for each element,
   * data range is from 0 to (classes - 1).
   */
  virtual void rand(size_t classes) = 0;

  /**
   * Debug use only. Very inefficient for GPU vector.
   * get the value at pos.
   */
  virtual T get(size_t pos) = 0;

  /**
   * generate univariate Gaussian distributed random numbers
   * with given mean and standardDeviation.
   */
  virtual void randnorm(real mean, real standardDeviation) = 0;

  /**
   * generate uniform distributed random numbers
   * with given range.
   */
  virtual void uniform(real left, real right) = 0;

  /// print the first "num" elements of the Vector
  virtual void print(std::ostream& os, size_t num) const = 0;

  /// print the "idx" element of the Vector
  virtual void printOneElement(std::ostream& os, size_t idx) const = 0;

  template <typename ExpressionType>
  void operator=(const ExpressionType& expr) {
    if (BaseVector<T>::useGpu_) {
      TensorGpuApply<T>(*this, expr);
    } else {
      TensorCpuApply<T>(*this, expr);
    }
  }

protected:
  friend class GpuVectorT<T>;
  friend class CpuVectorT<T>;
  virtual void copyTo(CpuVectorT<T>* dest) const = 0;
  virtual void copyTo(GpuVectorT<T>* dest) const = 0;
  MemoryHandlePtr memoryHandle_;
};

template <class T>
std::ostream& operator<<(std::ostream& os, const VectorT<T>& vec) {
  vec.print(os, vec.getSize());
  return os;
}

template <class T>
class GpuVectorT : public VectorT<T> {
public:
  explicit GpuVectorT(size_t size);
  GpuVectorT(size_t size, GpuMemHandlePtr memHandle, size_t offset)
      : VectorT<T>(size, memHandle, offset, true) {}

  // data is still owned by the caller.
  // data should be valid during the life of this vector.
  // Caller is responsible for release the memory.
  GpuVectorT(size_t size, T* data) : VectorT<T>(size, data, true) {}

  virtual MemoryHandlePtr newMemory(size_t size) {
    return std::make_shared<GpuMemoryHandle>(size);
  }
  virtual void zeroMem();
  virtual void reset(const T& value);
  virtual void fillSequence();

  virtual void copyFrom(const T* src, size_t size);
  virtual void copyFrom(const T* src, size_t size, hl_stream_t stream);
  virtual void copyFrom(const VectorT<T>& src);
  virtual void copyFrom(const VectorT<T>& src, hl_stream_t stream);
  virtual T getElement(size_t i) const;
  virtual void setElement(size_t i, const T& value);
  virtual T* getPoint(const uint64_t beginPos);

  virtual T getAbsSum();
  virtual T getSum();
  virtual T getMax();
  virtual T getAbsMax();
  virtual T getMin();
  virtual void isEqualTo(const VectorT<T>& b, const T& value);
  virtual void selectFrom(const VectorT<T>& src, const VectorT<int>& ids);
  virtual void histogram(std::ostream& os, int type);
  virtual void rand();
  virtual void rand(size_t classes);
  virtual void randnorm(real mean, real standardDeviation);
  virtual void uniform(real left, real right);
  virtual T get(size_t pos);
  virtual void print(std::ostream& os, size_t num) const;
  virtual void printOneElement(std::ostream& os, size_t idx) const;

  template <typename ExpressionType>
  void operator=(const ExpressionType& expr) {
    TensorGpuApply<T>(*this, expr);
  }

protected:
  virtual void copyTo(CpuVectorT<T>* dest) const;
  virtual void copyTo(GpuVectorT<T>* dest) const;
};

template <class T>
class CpuVectorT : public VectorT<T> {
public:
  explicit CpuVectorT(size_t size);
  CpuVectorT(size_t size, MemoryHandlePtr memoryHandle, size_t offset)
      : VectorT<T>(size, memoryHandle, offset, false) {}

  // data is still owned by the caller.
  // data should be valid during the life of this vector.
  // Caller is responsible for release the memory.
  CpuVectorT(size_t size, T* data) : VectorT<T>(size, data, false) {}

  /**
   * If src is a CpuVector, the new CpuVector will share the data with src
   *
   * If src is a GpuVector, the new CpuVector will copy data from src
   */
  explicit CpuVectorT(const VectorT<T>& src);

  virtual MemoryHandlePtr newMemory(size_t size) {
    return std::make_shared<CpuMemoryHandle>(size);
  }

  virtual void zeroMem();
  virtual void reset(const T& value);
  virtual void fillSequence();
  virtual void copyFrom(const T* src, size_t size);
  virtual void copyFrom(const T* src, size_t size, hl_stream_t stream);
  virtual void copyFrom(const VectorT<T>& src);
  virtual void copyFrom(const VectorT<T>& src, hl_stream_t stream);
  virtual void copyTo(CpuVectorT<T>* dest) const;
  virtual void copyTo(GpuVectorT<T>* dest) const;

  /// Get the buffer point with beginPos
  virtual T* getPoint(const uint64_t beginPos) {
    return this->getData() + beginPos;
  }

  virtual T getElement(size_t i) const { return this->getData()[i]; }
  virtual void setElement(size_t i, const T& value) {
    this->getData()[i] = value;
  }

  virtual T getAbsSum();
  virtual T getSum();
  virtual T getMax();
  virtual T getAbsMax();
  virtual T getMin();
  virtual void isEqualTo(const VectorT<T>& b, const T& value);
  virtual void selectFrom(const VectorT<T>& src, const VectorT<int>& ids);
  virtual void histogram(std::ostream& os, int type);
  virtual void rand();
  virtual void rand(size_t classes);
  virtual void randnorm(real mean, real standardDeviation);
  virtual void uniform(real left, real right);
  virtual T get(size_t pos);
  virtual void print(std::ostream& os, size_t num) const;
  virtual void printOneElement(std::ostream& os, size_t idx) const;

  template <typename ExpressionType>
  void operator=(const ExpressionType& expr) {
    TensorCpuApply<T>(*this, expr);
  }
};

template <class T>
class ParallelCpuVectorT : public CpuVectorT<T> {
public:
  ParallelCpuVectorT(size_t size, SyncThreadPool* pool)
      : CpuVectorT<T>(size), pool_(pool) {}

  virtual void zeroMem() {
    parallelExec([](CpuVectorT<T>& vec) { vec.CpuVectorT<T>::zeroMem(); });
  }
  virtual void randnorm(real mean, real standardDeviation) {
    parallelExec([=](CpuVectorT<T>& vec) {
      vec.CpuVectorT<T>::randnorm(mean, standardDeviation);
    });
  }
  virtual void uniform(real left, real right) {
    parallelExec(
        [=](CpuVectorT<T>& vec) { vec.CpuVectorT<T>::uniform(left, right); });
  }

  virtual void exec(SyncThreadPool::JobFunc jobFunc);

private:
  typedef std::function<void(CpuVectorT<T>& vec)> ExecFunc;
  void parallelExec(ExecFunc func);
  SyncThreadPool* pool_;
};

/**
 * A class to do conversion between CpuVector and GpuVector automatically.
 */
template <class T>
class CpuGpuVectorT {
public:
  /**
   * @brief An enum type of SyncedFlag using to
   *        mark data memory is in CPU or GPU.
   *
   * DATA_AT_CPU: data is located in CPU.
   *
   * DATA_AT_GPU: data is located in GPU.
   *
   * SYNCED: data is located in CPU and GPU simultaneously.
   */
  enum SyncedFlag { DATA_AT_CPU = 0, DATA_AT_GPU = 1, SYNCED = 2 };

  /**
   * @brief A constructor, create cpuVectorT_ or gpuVectorT_.
   *
   * @param[in] size    data size.
   * @param[in] useGpu  use gpu or not.
   */
  explicit CpuGpuVectorT(size_t size, bool useGpu);

  /**
   * @brief A constructor, create CpuGpuVectorT by VectorT.
   *
   * If src is CpuVector, cpuVectorT_ is shared data with src.
   *
   * If src is GpuVector, gpuVectorT_ is shared data with src.
   */
  explicit CpuGpuVectorT(const std::shared_ptr<VectorT<T>>& src);

  /**
   * @brief A constructor.
   *
   * If useGpu is true, data should be located in device and
   * create gpuVectorT_ with data.
   *
   * If useGpu is false, data should be located in host and
   * create cpuVectorT_ with data.
   *
   * @note Data is owned by the caller and should be valid during
   *       the life of this vector.
   *       Caller is responsible for release the memory.
   */
  CpuGpuVectorT(size_t size, T* data, bool useGpu);

  CpuGpuVectorT(CpuGpuVectorT<T>& src, size_t offset, size_t size);

  virtual ~CpuGpuVectorT() {}

  static std::shared_ptr<CpuGpuVectorT<T>> create(size_t size, bool useGpu);

  /**
   * @brief resize vector.
   *
   * If useGpu is true, resize gpuVectorT_ and set syncFlag_ to DATA_AT_GPU,
   *
   * otherwise resize cpuVectorT_ and set syncFlag_ to DATA_AT_CPU.
   */
  void resize(size_t size, bool useGpu);

  /**
   * @brief resize or create CpuGpuVectorT.
   */
  static void resizeOrCreate(std::shared_ptr<CpuGpuVectorT<T>>& vec,
                             size_t size,
                             bool useGpu);

  /**
   * @brief return a const cpuVectorT_ or gpuVectorT_.
   *
   * If useGpu is true, return gpuVectorT_.
   *
   * If useGpu is false, return cpuVectorT_.
   *
   * @note Caller should not change the data.
   *       If caller changes const attribute,
   *       should set syncFlag_.
   */
  std::shared_ptr<const VectorT<T>> getVector(bool useGpu) const;

  /**
   * @brief return a const cpuVectorT_ or gpuVectorT_.
   *
   * @note: This interface will change syncFlag_, so if you will
   *        not change the data, you should call getVector.
   */
  std::shared_ptr<VectorT<T>>& getMutableVector(bool useGpu);

  /**
   * @brief return const T* data.
   *
   * If useGpu is true, return device data.
   *
   * If useGpu is false, return host data.
   */
  const T* getData(bool useGpu) const;

  // TODO(yuyang18): Make getData more c++ style.
  //  inline T* getData(bool useGpu) {
  //    return getMutableData(useGpu);
  //  }

  T* getMutableData(bool useGpu);

  /**
   * If useGpu is true, gpuVectorT_->Op().
   *
   * If useGpu is false, cpuVectorT_->Op().
   *
   * Op is zeroMem, fillSequence, ...
   */
  void zeroMem(bool useGpu);
  void fillSequence(bool useGpu);
  void setElement(size_t i, const T& value, bool useGpu);

  /**
   * @brief return i-th element.
   */
  T getElement(size_t i) const;

  /**
   * @brief return vector size.
   */
  size_t getSize() const {
    size_t size = 0;
    switch (*sync_) {
      case SYNCED:
      case DATA_AT_CPU:
        size = cpuVectorT_->getSize();
        break;
      case DATA_AT_GPU:
        size = gpuVectorT_->getSize();
        break;
      default:
        LOG(FATAL) << "Not support";
        break;
    }
    return size;
  }

  /// copy data to cpuVectorT_.
  inline void copyToCpu(const T* data, size_t size) {
    this->resizeOrCreate(size, false);
    cpuVectorT_->copyFrom(data, size);
    setSync(DATA_AT_CPU);
  }
  /// copy data to cpuVectorT_ using specifed-stream.
  inline void copyToCpu(const T* data, size_t size, hl_stream_t stream) {
    this->resizeOrCreate(size, false);
    cpuVectorT_->copyFrom(data, size, stream);
    setSync(DATA_AT_CPU);
  }

  /// copy data to gpuVectorT_.
  inline void copyToGpu(const T* data, size_t size) {
    this->resizeOrCreate(size, true);
    gpuVectorT_->copyFrom(data, size);
    setSync(DATA_AT_GPU);
  }
  /// copy data to gpuVectorT_ using specifed-stream.
  inline void copyToGpu(const T* data, size_t size, hl_stream_t stream) {
    this->resizeOrCreate(size, true);
    gpuVectorT_->copyFrom(data, size, stream);
    setSync(DATA_AT_GPU);
  }

  /**
   * @brief copy from src using specifed-stream.
   *
   * If src is CpuVectorT, copy to cpuVectorT_.
   *
   * If src is GpuVectorT, copy to gpuVectorT_.
   */
  void copyFrom(const VectorT<T>& src, hl_stream_t stream);

  /**
   * @brief copy data.
   *
   * If useGpu is false, copy host data to cpuVectorT_.
   *
   * If useGpu is true, copy device data to gpuVectorT_.
   *
   * @note  data address should consistent with useGpu.
   */
  void copyFrom(const T* data, size_t size, bool useGpu);
  void copyFrom(const T* data, size_t size, hl_stream_t stream, bool useGpu);

  /**
   * @brief copy from (src + offset) using specifed-stream.
   */
  void copyFrom(CpuGpuVectorT<T>& src,
                size_t offset,
                size_t size,
                bool useGpu,
                hl_stream_t stream);

  /**
   * @brief copy from src using specifed-stream.
   */
  void copyFrom(CpuGpuVectorT<T>& src, hl_stream_t stream);

  /**
   * @brief return sync_.
   */
  inline SyncedFlag* getSync() const { return sync_; }

  /**
   * @brief set sync_.
   */
  inline void setSync(SyncedFlag* sync) { sync_ = sync; }

  inline void setSync(SyncedFlag syncFlag) {
    if (sync_) {
      *sync_ = syncFlag;
    } else {
      syncFlag_ = syncFlag;
      sync_ = &syncFlag_;
    }
  }

  inline void setSync(bool useGpu) {
    SyncedFlag flag = useGpu ? DATA_AT_GPU : DATA_AT_CPU;
    setSync(flag);
  }

protected:
  void resizeOrCreate(size_t size, bool useGpu);

  /**
   * @brief copy between cpuVectorT_ and gpuVectorT_.
   *
   * If syncFlag_ is DATA_AT_CPU and SYNCED, do nothing.
   *
   * If syncFlag_ is DATA_AT_GPU, copy gpuVectorT_ to cpuVectorT_
   *   and set syncFlag_ to SYNCED.
   */
  void copyToCpu();

  /**
   * @brief copy between cpuVectorT_ and gpuVectorT_.
   *
   * If syncFlag_ is DATA_AT_GPU and SYNCED, do nothing.
   *
   * If syncFlag_ is DATA_AT_CPU, copy cpuVectorT_ to gpuVectorT_
   *   and set syncFlag_ to SYNCED.
   */
  void copyToGpu();

  /// host pointer.
  std::shared_ptr<VectorT<T>> cpuVectorT_;
  /// device pointer.
  std::shared_ptr<VectorT<T>> gpuVectorT_;
  /// specify current data address.
  SyncedFlag syncFlag_;
  SyncedFlag* sync_;
};

typedef VectorT<real> Vector;
typedef CpuVectorT<real> CpuVector;
typedef GpuVectorT<real> GpuVector;

typedef VectorT<int> IVector;
typedef CpuVectorT<int> CpuIVector;
typedef GpuVectorT<int> GpuIVector;

typedef std::shared_ptr<Vector> VectorPtr;
typedef std::shared_ptr<CpuVector> CpuVectorPtr;
typedef std::shared_ptr<GpuVector> GpuVectorPtr;

typedef std::shared_ptr<IVector> IVectorPtr;
typedef std::shared_ptr<CpuIVector> CpuIVectorPtr;
typedef std::shared_ptr<GpuIVector> GpuIVectorPtr;

typedef CpuGpuVectorT<real> CpuGpuVector;
typedef CpuGpuVectorT<int> ICpuGpuVector;
typedef std::shared_ptr<CpuGpuVector> CpuGpuVectorPtr;
typedef std::shared_ptr<ICpuGpuVector> ICpuGpuVectorPtr;

template <class T>
std::shared_ptr<VectorT<T>> VectorT<T>::create(size_t size, bool useGpu) {
  if (useGpu) {
    return std::make_shared<GpuVectorT<T>>(size);
  } else {
    return std::make_shared<CpuVectorT<T>>(size);
  }
}

template <class T>
std::shared_ptr<VectorT<T>> VectorT<T>::createParallelVector(
    size_t size, bool useGpu, SyncThreadPool* pool) {
  if (!useGpu && FLAGS_trainer_count > 1 && FLAGS_enable_parallel_vector &&
      size >= (size_t)FLAGS_enable_parallel_vector) {
    return std::make_shared<ParallelCpuVectorT<T>>(
        size, pool ? pool : getGlobalSyncThreadPool());
  } else {
    return create(size, useGpu);
  }
}

template <class T>
std::shared_ptr<VectorT<T>> VectorT<T>::create(T* data,
                                               size_t size,
                                               bool useGpu) {
  if (useGpu) {
    return std::make_shared<GpuVectorT<T>>(size, data);
  } else {
    return std::make_shared<CpuVectorT<T>>(size, data);
  }
}

template <class T>
std::shared_ptr<VectorT<T>> VectorT<T>::create(size_t size,
                                               MemoryHandlePtr memoryHandle,
                                               size_t offset) {
  if (auto cpuMemHandle =
          std::dynamic_pointer_cast<CpuMemoryHandle>(memoryHandle)) {
    return std::make_shared<CpuVectorT<T>>(size, cpuMemHandle, offset);
  } else if (auto gpuMemHandle =
                 std::dynamic_pointer_cast<GpuMemoryHandle>(memoryHandle)) {
    return std::make_shared<GpuVectorT<T>>(size, gpuMemHandle, offset);
  } else {
    LOG(FATAL) << "Wrong";
    return NULL;
  }
}

template <>
MatrixPtr VectorT<real>::toOneHotSparseMatrix(size_t idRange, bool useGpu) {
  LOG(FATAL) << "Wrong for real vector";
  return nullptr;
}

template <>
MatrixPtr VectorT<int>::toOneHotSparseMatrix(size_t idRange, bool useGpu) {
  size_t height = getSize();
  size_t width = idRange;
  MatrixPtr mat = Matrix::createSparseMatrix(
      height, idRange, height, NO_VALUE, SPARSE_CSR, false, useGpu);

  CpuIVector cpuIds(height);
  cpuIds.copyFrom(*this);
  int* idData = cpuIds.getData();

  for (decltype(height) i = 0; i < height; i++) {
    const unsigned int id = idData[i];
    CHECK_LT(id, width);
    mat->setRow(i, 1, &id, nullptr);
  }
  return mat;
}

template <>
std::shared_ptr<VectorT<int>> VectorT<real>::castToInt() {
  std::shared_ptr<VectorT<int>> ret = IVector::create(this->getSize(), useGpu_);
  if (useGpu_) {
    hl_vector_cast2int(ret->getData(), this->getData(), this->getSize());
  } else {
    for (size_t i = 0; i < getSize(); ++i) {
      ret->getData()[i] = int(this->getData()[i]);
    }
  }
  return ret;
}

template <class T>
GpuVectorT<T>::GpuVectorT(size_t size)
    : VectorT<T>(size,
                 std::make_shared<GpuMemoryHandle>(sizeof(T) * size),
                 0, /* offset = 0 */
                 true /* useGpu = true */) {}

template <class T>
T GpuVectorT<T>::getElement(size_t i) const {
  T elem = 0;
  hl_memcpy_device2host(&elem, const_cast<T*>(&this->getData()[i]), sizeof(T));
  return elem;
}
template <class T>
void GpuVectorT<T>::setElement(size_t i, const T& value) {
  hl_memcpy_host2device(&this->getData()[i], const_cast<T*>(&value), sizeof(T));
}

template <class T>
T* GpuVectorT<T>::getPoint(const uint64_t beginPos) {
  LOG(FATAL) << "Not implemented" << beginPos;
  return NULL;
}

template <>
int GpuVectorT<int>::getAbsSum() {
  LOG(FATAL) << "Not implemented";
  return 0;
}

template <>
int GpuVectorT<int>::getSum() {
  LOG(FATAL) << "Not implemented";
  return 0;
}

template <>
real GpuVectorT<real>::getAbsSum() {
  real* A = this->getData();
  real sum = 0;
  hl_vector_abs_sum(A, &sum, this->getSize());
  return sum;
}

template <>
real GpuVectorT<real>::getSum() {
  real* A = this->getData();
  real sum = 0;
  hl_vector_sum(A, &sum, this->getSize());
  return sum;
}

template <>
int GpuVectorT<int>::getMax() {
  CpuIVector cpuIVec = CpuIVector(this->getSize());
  copyTo(&cpuIVec);
  return cpuIVec.getMax();
}

template <>
int GpuVectorT<int>::getAbsMax() {
  CpuIVector cpuIVec = CpuIVector(this->getSize());
  copyTo(&cpuIVec);
  return cpuIVec.getAbsMax();
}

template <class T>
void GpuVectorT<T>::isEqualTo(const VectorT<T>& b, const T& value) {
  BaseMatrixT<T>::isEqualTo((BaseMatrixT<T>&)b, value);
}

template <class T>
void GpuVectorT<T>::selectFrom(const VectorT<T>& src, const VectorT<int>& ids) {
#ifdef PADDLE_WITH_CUDA
  hl_vector_select_from<T>(this->getData(),
                           this->getSize(),
                           src.getData(),
                           src.getSize(),
                           ids.getData(),
                           ids.getSize());
#endif
}

template <class Func>
real gpuRowFunc(Func f, GpuVector& v) {
  static ThreadLocal<std::unique_ptr<CpuVectorT<real>>> local;
  if (!*local) {
    (*local).reset(new CpuVector(1));
  }
  real* A = v.getData();
  f(A, (*local)->getData(), 1, v.getSize());
  return (*local)->getData()[0];
}

template <>
real GpuVectorT<real>::getMax() {
  return gpuRowFunc(hl_matrix_row_max, *this);
}

template <>
real GpuVectorT<real>::getAbsMax() {
  return std::max(gpuRowFunc(hl_matrix_row_max, *this),
                  -gpuRowFunc(hl_matrix_row_min, *this));
}

template <>
int GpuVectorT<int>::getMin() {
  LOG(FATAL) << "Not implemented";
  return 0;
}

template <>
real GpuVectorT<real>::getMin() {
  return gpuRowFunc(hl_matrix_row_min, *this);
}

template <class T>
T GpuVectorT<T>::get(size_t pos) {
  T val = (T)0;
  hl_memcpy_device2host((void*)&val, (void*)(this->getData() + pos), sizeof(T));
  return val;
}

template <class T>
void GpuVectorT<T>::histogram(std::ostream& os, int type) {
  LOG(FATAL) << "Not implemented";
}

template <class T>
void GpuVectorT<T>::zeroMem() {
  BaseMatrixT<T>::zero();
}

template <class T>
void GpuVectorT<T>::reset(const T& value) {
  BaseMatrixT<T>::assign(value);
}

template <class T>
void GpuVectorT<T>::fillSequence() {
  LOG(FATAL) << "not implemented";
}

template <class T>
void GpuVectorT<T>::copyFrom(const VectorT<T>& src) {
  src.copyTo(this);
}

template <class T>
void GpuVectorT<T>::copyFrom(const VectorT<T>& src, hl_stream_t stream) {
  CHECK_EQ(src.getSize(), this->getSize());
  hl_memcpy_async((void*)this->getData(),
                  (void*)src.getData(),
                  sizeof(T) * this->getSize(),
                  stream);
}

template <class T>
void GpuVectorT<T>::copyFrom(const T* gpuSrc, size_t size) {
  CHECK(gpuSrc != NULL);
  CHECK_LE(size, this->size_);

  hl_memcpy((void*)this->getData(), (void*)gpuSrc, sizeof(T) * size);
}

template <class T>
void GpuVectorT<T>::copyFrom(const T* gpuSrc, size_t size, hl_stream_t stream) {
  CHECK(gpuSrc != NULL);
  CHECK_LE(size, this->size_);

  hl_memcpy_async(
      (void*)this->getData(), (void*)gpuSrc, sizeof(T) * size, stream);
}

template <class T>
void GpuVectorT<T>::copyTo(CpuVectorT<T>* dest) const {
  CHECK_EQ(this->getSize(), dest->getSize());

  hl_memcpy_device2host((void*)dest->getData(),
                        (void*)this->getData(),
                        sizeof(T) * this->getSize());
}

template <class T>
void GpuVectorT<T>::copyTo(GpuVectorT<T>* dest) const {
  CHECK_EQ(this->getSize(), dest->getSize());

  hl_memcpy_device2device((void*)dest->getData(),
                          (void*)this->getData(),
                          sizeof(T) * this->getSize());
}

template <>
void GpuVectorT<int>::rand() {
  LOG(FATAL) << "Not implemented";
}

template <>
void GpuVectorT<int>::print(std::ostream& os, size_t num) const {
  IVectorPtr dest = IVector::create(this->size_, false);
  hl_memcpy_device2host((void*)dest->getData(),
                        (void*)this->getData(),
                        sizeof(int) * this->getSize());
  dest->print(os, num);
}

template <>
void GpuVectorT<real>::print(std::ostream& os, size_t num) const {
  VectorPtr dest = Vector::create(this->size_, false);
  hl_memcpy_device2host((void*)dest->getData(),
                        (void*)this->getData(),
                        sizeof(int) * this->getSize());
  dest->print(os, num);
}

template <>
void GpuVectorT<int>::printOneElement(std::ostream& os, size_t idx) const {
  LOG(FATAL) << "Not implemented";
}

template <>
void GpuVectorT<real>::printOneElement(std::ostream& os, size_t idx) const {
  LOG(FATAL) << "Not implemented";
}

template <>
void CpuVectorT<int>::rand() {
  LOG(FATAL) << "Not implemented";
}
template <>
void GpuVectorT<real>::rand(size_t classNum) {
  LOG(FATAL) << "Not implemented";
}

template <>
void CpuVectorT<real>::rand(size_t classNum) {
  LOG(FATAL) << "Not implemented";
}

template <>
void GpuVectorT<real>::rand() {
  VectorPtr cPtr = Vector::create(this->size_, false);
  cPtr->rand();

  hl_memcpy_host2device(data_, cPtr->getData(), this->size_ * sizeof(real));
}

template <>
void GpuVectorT<int>::rand(size_t classNum) {
  IVectorPtr cPtr = IVector::create(this->size_, false);
  cPtr->rand(classNum);

  hl_memcpy_host2device(data_, cPtr->getData(), this->size_ * sizeof(int));
}

template <>
void CpuVectorT<int>::rand(size_t classNum) {
  size_t size = this->getSize();
  int* data = this->getData();
  for (size_t i = 0; i < size; i++) {
    data[i] =
        std::min(classNum - 1,
                 size_t(::rand() * (1. / ((double)RAND_MAX + 1)) * classNum));
  }
}

template <>
void CpuVectorT<real>::rand() {
  size_t size = this->getSize();
  real* data = this->getData();
  for (size_t i = 0; i < size; i++) {
    data[i] = ::rand() * (1. / (double)RAND_MAX);
    // data[ii] = ((temp > RAND_MAX/2)? 1 : -1) *
    // sqrt( abs((temp-RAND_MAX/2))/(double(RAND_MAX))/2048 );
  }
}

template <class T>
void CpuVectorT<T>::randnorm(real, real) {
  LOG(FATAL) << "Not implemented";
}

template <class T>
void CpuVectorT<T>::uniform(real, real) {
  LOG(FATAL) << "Not implemented";
}

template <class T>
void GpuVectorT<T>::randnorm(real, real) {
  LOG(FATAL) << "Not implemented";
}

template <class T>
void GpuVectorT<T>::uniform(real, real) {
  LOG(FATAL) << "Not implemented";
}

template <>
void CpuVectorT<real>::randnorm(real mean, real std) {
  size_t size = this->getSize();
  real* data = this->getData();
  unsigned int* seed = ThreadLocalRand::getSeed();
  auto rand1 = [&]() { return (1. + ::rand_r(seed)) * (1. / (1. + RAND_MAX)); };
  for (size_t i = 0; i < size - 1; i += 2) {
    real r1 = rand1();
    r1 = std::sqrt(-2 * std::log(r1));
    real r2 = rand1();
    data[i] = mean + std * r1 * cos(2 * M_PI * r2);
    data[i + 1] = mean + std * r1 * sin(2 * M_PI * r2);
  }
  real r1 = rand1();
  r1 = std::sqrt(-2 * std::log(r1));
  real r2 = rand1();
  data[size - 1] = mean + std * r1 * cos(2 * M_PI * r2);
}

template <>
void CpuVectorT<real>::uniform(real left, real right) {
  size_t size = this->getSize();
  real* data = this->getData();
  real range = right - left;
  unsigned int* seed = ThreadLocalRand::getSeed();
  auto rand1 = [&]() { return ::rand_r(seed) * (1. / (1. + RAND_MAX)); };
  for (size_t i = 0; i < size; ++i) {
    data[i] = rand1() * range + left;
  }
}

template <>
void GpuVectorT<real>::randnorm(real mean, real std) {
  CpuVector cpuVec = CpuVector(this->getSize());
  cpuVec.randnorm(mean, std);

  hl_memcpy_host2device(
      data_, cpuVec.getData(), this->getSize() * sizeof(real));
}

template <>
void GpuVectorT<real>::uniform(real left, real right) {
  CpuVector cpuVec = CpuVector(this->getSize());
  cpuVec.uniform(left, right);

  hl_memcpy_host2device(
      data_, cpuVec.getData(), this->getSize() * sizeof(real));
}

template <class T>
CpuVectorT<T>::CpuVectorT(size_t size)
    : VectorT<T>(size,
                 std::make_shared<CpuMemoryHandle>(sizeof(T) * size),
                 0, /* offset = 0 */
                 false /* useGpu = false */) {}

template <class T>
CpuVectorT<T>::CpuVectorT(const VectorT<T>& src)
    : VectorT<T>(src.getSize(),
                 src.getMemoryHandle(),
                 0, /* offset = 0 */
                 false /* useGpu = false */) {
  if (typeid(*this->memoryHandle_.get()) != typeid(CpuMemoryHandle)) {
    this->memoryHandle_ =
        std::make_shared<CpuMemoryHandle>(sizeof(T) * this->getSize());
    this->data_ = reinterpret_cast<T*>(this->memoryHandle_->getBuf());
  }
  src.copyTo(this);
}

template <class T>
T CpuVectorT<T>::getAbsSum() {
  const T* A = this->getData();
  size_t size = this->getSize();
  T sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += (A[i] > 0) ? A[i] : -A[i];
  }
  return sum;
}

// cannot use above version, due to precision issue of float
template <>
real CpuVectorT<real>::getAbsSum() {
  const real* A = this->getData();
  size_t size = this->getSize();
  double sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += (A[i] > 0) ? A[i] : -A[i];
  }
  return sum;
}

template <class T>
T CpuVectorT<T>::getSum() {
  const T* A = this->getData();
  size_t size = this->getSize();
  T sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += A[i];
  }
  return sum;
}

template <>
real CpuVectorT<real>::getSum() {
  const real* A = this->getData();
  size_t size = this->getSize();
  double sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += A[i];
  }
  return sum;
}

template <class T>
T CpuVectorT<T>::get(size_t pos) {
  return this->getData()[pos];
}

template <class T>
T CpuVectorT<T>::getMax() {
  const T* A = this->getData();
  size_t size = this->getSize();
  T res = A[0];
  for (size_t i = 1; i < size; i++) {
    if (res < A[i]) res = A[i];
  }
  return res;
}

template <class T>
T CpuVectorT<T>::getAbsMax() {
  const T* A = this->getData();
  size_t size = this->getSize();
  T res = std::abs(A[0]);
  for (size_t i = 1; i < size; i++) {
    if (res < std::abs(A[i])) res = std::abs(A[i]);
  }
  return res;
}

template <class T>
T CpuVectorT<T>::getMin() {
  const T* A = this->getData();
  size_t size = this->getSize();
  T res = A[0];
  for (size_t i = 1; i < size; i++) {
    if (res > A[i]) res = A[i];
  }
  return res;
}

template <class T>
void CpuVectorT<T>::isEqualTo(const VectorT<T>& b, const T& value) {
  size_t size = this->getSize();
  CHECK_EQ(b.getSize(), size);

  const T* B = b.getData();
  T* A = this->getData();
  for (size_t i = 0; i < size; i++) {
    A[i] = (B[i] == value);
  }
}

template <class T>
void CpuVectorT<T>::selectFrom(const VectorT<T>& src, const VectorT<int>& ids) {
  size_t size = this->getSize();
  CHECK_EQ(ids.getSize(), size);

  const int* indices = ids.getData();
  const T* B = src.getData();
  T* A = this->getData();
  for (size_t i = 0; i < size; i++) {
    int index = indices[i];
    CHECK_LT(index, (int)src.getSize());
    A[i] = B[index];
  }
}

static int getSignAndExponentOfFloat(float a) {
  uint32_t* pa = reinterpret_cast<uint32_t*>(&a);
  return *pa >> 23;
}

template <class T>
void CpuVectorT<T>::histogram(std::ostream& os, int type) {
  LOG(FATAL) << "Not implemented";
}

template <>
void CpuVectorT<real>::histogram(std::ostream& os, int type) {
  int counters[512];
  memset(counters, 0, sizeof(counters));
  int counterZero = 0;

  const real* A = this->getData();
  size_t size = this->getSize();
  for (size_t i = 0; i < size; i++) {
    if (A[i] == 0.0f) {
      ++counterZero;
    } else {
      ++counters[getSignAndExponentOfFloat(A[i])];
    }
  }

  int64_t sum = 0;
  float sizeNonZero = size - counterZero;
  os << "zero:" << counterZero;
  for (int i = 0; i < 256; i++) {
    int counter = counters[i];
    if (counter) {
      os << " 2^" << i - 127 << ":" << counter / sizeNonZero * 100 << "%";
      sum += counter * (i - 127);
    }
  }
  for (int i = 0; i < 256; i++) {
    int counter = counters[i + 256];
    if (counter) {
      os << " -2^" << i - 127 << ":" << counter / sizeNonZero * 100 << "%";
      sum += counter * (i - 127);
    }
  }
  os << ", nonzero_exponent_avg=" << sum / sizeNonZero;
}

template <class T>
void CpuVectorT<T>::zeroMem() {
  memset(this->getData(), 0, sizeof(T) * this->getSize());
}

template <class T>
void CpuVectorT<T>::reset(const T& value) {
  T* A = this->getData();
  size_t size = this->getSize();
  for (size_t i = 0; i < size; i++) {
    A[i] = value;
  }
}

template <class T>
void CpuVectorT<T>::fillSequence() {
  T* A = this->getData();
  size_t size = this->getSize();
  for (size_t i = 0; i < size; i++) {
    A[i] = i;
  }
}

template <class T>
void CpuVectorT<T>::copyFrom(const VectorT<T>& src) {
  src.copyTo(this);
}

template <class T>
void CpuVectorT<T>::copyFrom(const VectorT<T>& src, hl_stream_t stream) {
  if (typeid(src) == typeid(GpuVectorT<T>)) {
    hl_memcpy_async((void*)this->getData(),
                    (void*)src.getData(),
                    sizeof(T) * this->getSize(),
                    stream);
    // There is a need to add synchronization to ensure that the data is copied.
    hl_stream_synchronize(stream);
  } else {
    src.copyTo(this);
  }
}

template <class T>
void CpuVectorT<T>::copyFrom(const T* hostSrc, size_t size) {
  CHECK(hostSrc != NULL);
  CHECK_LE(size, this->size_);
  memcpy(this->data_, hostSrc, sizeof(T) * size);
}

template <class T>
void CpuVectorT<T>::copyFrom(const T* hostSrc,
                             size_t size,
                             hl_stream_t stream) {
  (void)stream;

  CHECK(hostSrc != NULL);
  CHECK_LE(size, this->size_);
  memcpy(this->data_, hostSrc, sizeof(T) * size);
}

template <class T>
void CpuVectorT<T>::copyTo(CpuVectorT<T>* dest) const {
  CHECK_EQ(this->getSize(), dest->getSize());
  memcpy(dest->getData(), this->getData(), sizeof(T) * this->getSize());
}

template <class T>
void CpuVectorT<T>::copyTo(GpuVectorT<T>* dest) const {
  CHECK_EQ(this->getSize(), dest->getSize());
  hl_memcpy_host2device((void*)dest->getData(),
                        (void*)this->getData(),
                        sizeof(T) * this->getSize());
}

template <>
void CpuVectorT<real>::print(std::ostream& os, size_t num) const {
  size_t w = size_ < num ? size_ : num;
  os << "[";
  for (size_t i = 0; i < w; ++i) {
    os << data_[i] << " ";
  }
  os << "]" << std::endl;
}

template <>
void CpuVectorT<int>::print(std::ostream& os, size_t num) const {
  size_t w = size_ < num ? size_ : num;
  os << "[";
  for (size_t i = 0; i < w; ++i) {
    os << (int)data_[i] << " ";
  }
  os << "]" << std::endl;
}

template <>
void CpuVectorT<real>::printOneElement(std::ostream& os, size_t idx) const {
  CHECK_LT(idx, size_);
  os << data_[idx] << ";";
}

template <>
void CpuVectorT<int>::printOneElement(std::ostream& os, size_t idx) const {
  CHECK_LT(idx, size_);
  os << (int)data_[idx] << ";";
}

template <class T>
void ParallelCpuVectorT<T>::parallelExec(ExecFunc func) {
  LOG(FATAL) << "Not implemented";
}

template <>
void ParallelCpuVectorT<real>::parallelExec(ExecFunc func) {
  pool_->exec([this, func](int tid, size_t numThreads) {
    auto interval = calcSplitArrayInterval(
        this->getSize(), (size_t)tid, numThreads, 8LU /*for avx*/);
    // setup sub bufs
    CpuVector subVec(0, nullptr);
    subVec.subVecFrom(*this, interval);
    func(subVec);
  });
}

template <class T>
void ParallelCpuVectorT<T>::exec(SyncThreadPool::JobFunc func) {
  LOG(FATAL) << "Not implemented";
}

template <>
void ParallelCpuVectorT<real>::exec(SyncThreadPool::JobFunc func) {
  pool_->exec(func);
}

template <class T>
CpuGpuVectorT<T>::CpuGpuVectorT(size_t size, bool useGpu) : sync_(nullptr) {
  if (!useGpu) {
    cpuVectorT_ = std::make_shared<CpuVectorT<T>>(size);
  } else {
    gpuVectorT_ = std::make_shared<GpuVectorT<T>>(size);
  }
  setSync(useGpu);
}

template <class T>
CpuGpuVectorT<T>::CpuGpuVectorT(const std::shared_ptr<VectorT<T>>& src)
    : sync_(nullptr) {
  bool useGpu = src->useGpu();
  if (useGpu) {
    gpuVectorT_ = src;
  } else {
    cpuVectorT_ = src;
  }
  setSync(useGpu);
}

template <class T>
CpuGpuVectorT<T>::CpuGpuVectorT(size_t size, T* data, bool useGpu)
    : sync_(nullptr) {
  if (!useGpu) {
    cpuVectorT_ = std::make_shared<CpuVectorT<T>>(size, data);
    setSync(DATA_AT_CPU);
  } else {
    gpuVectorT_ = std::make_shared<GpuVectorT<T>>(size, data);
    setSync(DATA_AT_GPU);
  }
}

template <class T>
std::shared_ptr<CpuGpuVectorT<T>> CpuGpuVectorT<T>::create(size_t size,
                                                           bool useGpu) {
  return std::make_shared<CpuGpuVectorT<T>>(size, useGpu);
}

template <class T>
void CpuGpuVectorT<T>::resize(size_t size, bool useGpu) {
  if (useGpu) {
    CHECK(gpuVectorT_) << "gpuVectorT_ is null";
    // If memoryHandle_ is nullptr,
    // the data may be owned by the caller when it was constructed.
    // It should not resize for this case.
    if (gpuVectorT_->getMemoryHandle()) {
      gpuVectorT_->resize(size);
    } else {
      CHECK_EQ(gpuVectorT_->getSize(), size);
    }
  } else {
    CHECK(cpuVectorT_) << "cpuVectorT_ is null";
    // If memoryHandle_ is nullptr,
    // the data may be owned by the caller when it was constructed.
    // It should not resize for this case.
    if (cpuVectorT_->getMemoryHandle()) {
      cpuVectorT_->resize(size);
    } else {
      CHECK_EQ(cpuVectorT_->getSize(), size);
    }
  }
  setSync(useGpu);
}

template <class T>
void CpuGpuVectorT<T>::resizeOrCreate(std::shared_ptr<CpuGpuVectorT<T>>& vec,
                                      size_t size,
                                      bool useGpu) {
  if (vec) {
    vec->resize(size, useGpu);
  } else {
    vec = create(size, useGpu);
  }
}

template <class T>
void CpuGpuVectorT<T>::resizeOrCreate(size_t size, bool useGpu) {
  if (useGpu && (!gpuVectorT_)) {
    gpuVectorT_ = VectorT<T>::create(size, true);
  } else if ((!useGpu) && (!cpuVectorT_)) {
    cpuVectorT_ = VectorT<T>::create(size, false);
  } else {
    CHECK((useGpu && gpuVectorT_) || (!useGpu && cpuVectorT_));
    this->resize(size, useGpu);
  }
}

template <class T>
CpuGpuVectorT<T>::CpuGpuVectorT(CpuGpuVectorT<T>& src,
                                size_t offset,
                                size_t size)
    : sync_(nullptr) {
  CHECK_LE(offset + size, static_cast<size_t>(src.getSize()));
#ifdef PADDLE_WITH_CUDA
  SyncedFlag* flag = src.getSync();
  if (*flag == DATA_AT_CPU) {
    src.copyToGpu();  // will set synchronous data between CPU and GPU
  } else if (*flag == DATA_AT_GPU) {
    src.copyToCpu();  // will set synchronous data between CPU and GPU
  }
#endif
  auto cMemHandle = (src.getVector(false))->getMemoryHandle();
  cpuVectorT_ = std::make_shared<CpuVectorT<T>>(
      size, std::dynamic_pointer_cast<CpuMemoryHandle>(cMemHandle), offset);
#ifdef PADDLE_WITH_CUDA
  auto gMemHandle = (src.getVector(true))->getMemoryHandle();
  gpuVectorT_ = std::make_shared<GpuVectorT<T>>(
      size, std::dynamic_pointer_cast<GpuMemoryHandle>(gMemHandle), offset);
  src.setSync(SYNCED);
#endif
  setSync(src.getSync());
}

template <class T>
std::shared_ptr<const VectorT<T>> CpuGpuVectorT<T>::getVector(
    bool useGpu) const {
  auto* self = const_cast<CpuGpuVectorT<T>*>(this);
  if (useGpu) {
    self->copyToGpu();
    return std::const_pointer_cast<const VectorT<T>>(gpuVectorT_);
  } else {
    self->copyToCpu();
    return std::const_pointer_cast<const VectorT<T>>(cpuVectorT_);
  }
}

template <class T>
std::shared_ptr<VectorT<T>>& CpuGpuVectorT<T>::getMutableVector(bool useGpu) {
  setSync(useGpu);
  if (useGpu) {
    copyToGpu();
    return gpuVectorT_;
  } else {
    copyToCpu();
    return cpuVectorT_;
  }
}

template <class T>
const T* CpuGpuVectorT<T>::getData(bool useGpu) const {
  auto self = const_cast<CpuGpuVectorT<T>*>(this);
  if (useGpu) {
    self->copyToGpu();
    return gpuVectorT_->getData();
  } else {
    self->copyToCpu();
    return cpuVectorT_->getData();
  }
}

// Operation will change data and need to reset sync_ & syncFlag_.
#define MUTABLE_VECTOR_OP(OP, useGpu, args...) \
  do {                                         \
    if (useGpu) {                              \
      copyToGpu();                             \
      setSync(useGpu);                         \
      return gpuVectorT_->OP(args);            \
    } else {                                   \
      copyToCpu();                             \
      setSync(useGpu);                         \
      return cpuVectorT_->OP(args);            \
    }                                          \
  } while (0)

template <class T>
T* CpuGpuVectorT<T>::getMutableData(bool useGpu) {
  MUTABLE_VECTOR_OP(getData, useGpu);
}

template <class T>
void CpuGpuVectorT<T>::zeroMem(bool useGpu) {
  MUTABLE_VECTOR_OP(zeroMem, useGpu);
}

template <class T>
void CpuGpuVectorT<T>::fillSequence(bool useGpu) {
  MUTABLE_VECTOR_OP(fillSequence, useGpu);
}

template <class T>
void CpuGpuVectorT<T>::setElement(size_t i, const T& value, bool useGpu) {
  MUTABLE_VECTOR_OP(setElement, useGpu, i, value);
}

template <class T>
T CpuGpuVectorT<T>::getElement(size_t i) const {
  switch (*this->getSync()) {
    case SYNCED:
    case DATA_AT_CPU:
      return cpuVectorT_->getElement(i);
      break;
    case DATA_AT_GPU:
      return gpuVectorT_->getElement(i);
      break;
    default:
      LOG(FATAL) << "Not support";
      break;
  }
}

template <class T>
void CpuGpuVectorT<T>::copyFrom(const VectorT<T>& src, hl_stream_t stream) {
  auto cVec = dynamic_cast<const CpuVectorT<T>*>(&src);
  auto gVec = dynamic_cast<const GpuVectorT<T>*>(&src);
  if (cVec) {
    copyToCpu(cVec->getData(), cVec->getSize(), stream);
  } else if (gVec) {
    copyToGpu(gVec->getData(), gVec->getSize(), stream);
  } else {
    LOG(FATAL) << "Invalid type of src";
  }
}

template <class T>
void CpuGpuVectorT<T>::copyFrom(const T* data, size_t size, bool useGpu) {
  if (useGpu) {
    copyToGpu(data, size);
  } else {
    copyToCpu(data, size);
  }
}

template <class T>
void CpuGpuVectorT<T>::copyFrom(const T* data,
                                size_t size,
                                hl_stream_t stream,
                                bool useGpu) {
  if (useGpu) {
    copyToGpu(data, size, stream);
  } else {
    copyToCpu(data, size, stream);
  }
}

template <class T>
void CpuGpuVectorT<T>::copyFrom(CpuGpuVectorT<T>& src,
                                size_t offset,
                                size_t size,
                                bool useGpu,
                                hl_stream_t stream) {
  if (useGpu) {
    VectorT<T>::resizeOrCreate(gpuVectorT_, size, true);
    gpuVectorT_->copyFrom(src.getData(true) + offset, size, stream);
  } else {
    VectorT<T>::resizeOrCreate(cpuVectorT_, size, false);
    cpuVectorT_->copyFrom(src.getData(false) + offset, size, stream);
  }
  setSync(useGpu);
}

template <class T>
void CpuGpuVectorT<T>::copyFrom(CpuGpuVectorT<T>& src, hl_stream_t stream) {
  switch (*src.getSync()) {
    case DATA_AT_CPU:
      copyFrom(*(src.getVector(false)), stream);
      break;
    case DATA_AT_GPU:
      copyFrom(*(src.getVector(true)), stream);
      break;
    case SYNCED:
      copyFrom(*(src.getVector(false)), stream);
      copyFrom(*(src.getVector(true)), stream);
      setSync(SYNCED);
      break;
    default:
      LOG(FATAL) << "Not support";
      break;
  }
}

template <class T>
void CpuGpuVectorT<T>::copyToCpu() {
  switch (*this->getSync()) {
    case DATA_AT_GPU:
      CHECK(gpuVectorT_);
      this->resizeOrCreate(gpuVectorT_->getSize(), false);
      cpuVectorT_->copyFrom(*gpuVectorT_);
      setSync(SYNCED);
      break;
    case DATA_AT_CPU:
    case SYNCED:
      CHECK(cpuVectorT_);
      break;
    default:
      LOG(FATAL) << "Not support";
      break;
  }
}

template <class T>
void CpuGpuVectorT<T>::copyToGpu() {
  switch (*this->getSync()) {
    case DATA_AT_CPU:
      CHECK(cpuVectorT_);
      this->resizeOrCreate(cpuVectorT_->getSize(), true);
      gpuVectorT_->copyFrom(*cpuVectorT_);
      setSync(SYNCED);
      break;
    case DATA_AT_GPU:
    case SYNCED:
      CHECK(gpuVectorT_);
      break;
    default:
      LOG(FATAL) << "Not support";
      break;
  }
}

template class VectorT<real>;
template class VectorT<int>;
template class CpuVectorT<real>;
template class CpuVectorT<int>;
template class GpuVectorT<real>;
template class GpuVectorT<int>;
template class CpuGpuVectorT<real>;
template class CpuGpuVectorT<int>;

}  // namespace mypaddle
}  // namespace bubblefs