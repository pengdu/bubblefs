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

// Paddle/paddle/math/Storage.h
// Paddle/paddle/math/Storage.cpp
// Paddle/paddle/math/MemoryHandle.h
// Paddle/paddle/math/MemoryHandle.cpp
// Paddle/paddle/math/RowBuffer.h

#pragma once

#include <cmath>
#include <memory>
#include <mutex>
#include <vector>
#include "platform/paddle_locks.h"
#include "utils/paddle_pool_allocator.h"

#ifndef PADDLE_MOBILE_INFERENCE
DEFINE_int32(pool_limit_size,
             536870912,
             "maximum memory size managed by a memory pool, default is 512M");
#else
DEFINE_int32(pool_limit_size, 0, "default is 0");
#endif

namespace bubblefs {
namespace mypaddle {

/**
 * @brief Storage manager for multiple devices.
 */
class StorageEngine {
public:
  /**
   * @return Storage singleton
   */
  static StorageEngine* singleton();

  /**
   * @return return one gpu allocator by deviceId
   */
  PoolAllocator* getGpuAllocator(int deviceId);

  /**
   * @return return cpu allocator
   */
  PoolAllocator* getCpuAllocator();

protected:
  StorageEngine();
  ~StorageEngine();
  RWLock lock_;
  std::vector<PoolAllocator*> gpuAllocator_;
  PoolAllocator* cpuAllocator_;
};  
  
class MemoryHandle {
protected:
  explicit MemoryHandle(size_t size);
  virtual ~MemoryHandle() {}

public:
  void* getBuf() const { return buf_; }
  size_t getSize() const { return size_; }
  size_t getAllocSize() const { return allocSize_; }

protected:
  PoolAllocator* allocator_;
  size_t size_;       // the requested size
  size_t allocSize_;  // the allocated size
  int deviceId_;      // the device id of memory if gpu memory
  void* buf_;
};

/**
 * Wrapper class for raw gpu memory handle.
 *
 * The raw handle will be released at destructor
 */
class GpuMemoryHandle : public MemoryHandle {
public:
  explicit GpuMemoryHandle(size_t size);
  virtual ~GpuMemoryHandle();
};

/**
 * Wrapper class for raw cpu memory handle.
 *
 * The raw handle will be released at destructor
 */
class CpuMemoryHandle : public MemoryHandle {
public:
  explicit CpuMemoryHandle(size_t size);
  virtual ~CpuMemoryHandle();
};

typedef std::shared_ptr<MemoryHandle> MemoryHandlePtr;
typedef std::shared_ptr<CpuMemoryHandle> CpuMemHandlePtr;
typedef std::shared_ptr<GpuMemoryHandle> GpuMemHandlePtr;

/**
 * @brief The RowBuffer class
 * Represent the SparseRow Matrix Data.
 *
 * If not set memory handler, then the data could be auto growth.
 */
class RowBuffer {
public:
  /**
   * @brief RowBuffer create a auto-growth row buffer. The row length is width.
   * @param width the length of each row, a.k.a matrix width.
   */
  explicit RowBuffer(size_t width) : width_(width) {}

  /**
   * @brief RowBuffer create a row buffer, which cannot be auto-growth.
   * @param mem the pre-allocated memory.
   * @param width the length of each row, a.k.a matrix width.
   */
  RowBuffer(const CpuMemHandlePtr& mem, size_t width)
      : preallocatedBuf_(mem), width_(width) {}

  /**
   * @brief resize resize the buffer with rowCount
   * @param rowCnt number of row. matrix height.
   */
  inline void resize(int rowCnt) {
    if (preallocatedBuf_) {
      CHECK(preallocatedBuf_->getSize() >= rowCnt * width_ * sizeof(real));
    } else {
      rowStore_.resize(rowCnt * width_);
    }
  }

  /**
   * @brief get a row buffer with row index.
   * @param row the index of row.
   * @return row buffer.
   */
  inline real* get(int row) const {
    if (preallocatedBuf_) {
      CHECK_LE((row)*width_ * sizeof(real), preallocatedBuf_->getSize());
      return reinterpret_cast<real*>(preallocatedBuf_->getBuf()) + row * width_;
    } else {
      CHECK_LE((row + 1) * width_, rowStore_.size());
      return const_cast<real*>(rowStore_.data() + row * width_);
    }
  }

  /**
   * @brief get a row buffer with row index. If row index is larger than local
   *        buffer, the size of local buffer will grow.
   * @param row the index of row.
   * @return row buffer.
   */
  inline real* getWithAutoGrowth(int row) {
    if (preallocatedBuf_) {
      return get(row);
    } else {
      if ((rowStore_.size() <= row * width_)) {
        rowStore_.resize((row + 1) * width_);
      }
      return rowStore_.data() + row * width_;
    }
  }

  /**
   * @return raw data buffer.
   */
  inline real* data() {
    if (preallocatedBuf_) {
      return reinterpret_cast<real*>(preallocatedBuf_->getBuf());
    } else {
      return rowStore_.data();
    }
  }

  /**
   * @brief clear local buffer. It only affect auto-growth buffer.
   */
  inline void clear() {
    // swap an empty vector to it to free the memory.
    std::vector<real, AlignedAllocator<real, 32>> empty;
    rowStore_.swap(empty);
  }

  /**
   * @brief get current number of rows.
   * @return number of rows.
   */
  inline size_t getRowCount() const {
    if (preallocatedBuf_) {
      return preallocatedBuf_->getSize() / sizeof(real) / width_;
    } else {
      return rowStore_.size() / width_;
    }
  }

  /**
   * @brief get is this buffer can automatically grow or not.
   * @return ture if can automacitally grow.
   */
  inline bool isAutoGrowth() const { return !preallocatedBuf_; }

  /**
   * @brief return the width of matrix. a.k.a length of row.
   * @return width of matrix
   */
  inline size_t getWidth() const { return width_; }

private:
  //! TODO(yuyang18): Add resize method to CpuMemHandlePtr, then we can get rid
  //! of std::vector here.
  CpuMemHandlePtr preallocatedBuf_;
  std::vector<real, AlignedAllocator<real, 32>> rowStore_;
  size_t width_;
};

// Initialization StorageEngine singleton.
// Other modules may rely on storage management,
// so StorageEngine need to be initialized before other modules.
static InitFunction __init_storage_engine([]() { StorageEngine::singleton(); },
                                          std::numeric_limits<int>::max());

StorageEngine::StorageEngine() : cpuAllocator_(nullptr) {}

StorageEngine::~StorageEngine() {
  delete cpuAllocator_;
  for (auto it : gpuAllocator_) {
    delete it;
  }
}

StorageEngine* StorageEngine::singleton() {
  static StorageEngine storage;
  return &storage;
}

PoolAllocator* StorageEngine::getGpuAllocator(int deviceId) {
  {
    // if gpuAllocator_ has been constructed
    ReadLockGuard guard(lock_);
    if (deviceId < static_cast<int>(gpuAllocator_.size()) &&
        (gpuAllocator_[deviceId] != nullptr)) {
      return gpuAllocator_[deviceId];
    }
  }

  {
    // Construct gpuAllocator_
    std::lock_guard<RWLock> guard(lock_);
    if (deviceId >= static_cast<int>(gpuAllocator_.size())) {
      gpuAllocator_.resize(deviceId + 1);
    }
    if (gpuAllocator_[deviceId] == nullptr) {
      std::string name =
          "gpu" + str::to_string(deviceId) + std::string("_pool");
      gpuAllocator_[deviceId] =
          new PoolAllocator(new GpuAllocator(), FLAGS_pool_limit_size, name);
    }
    return gpuAllocator_[deviceId];
  }
}

PoolAllocator* StorageEngine::getCpuAllocator() {
  {
    // if cpuAllocator_ has been constructed
    ReadLockGuard guard(lock_);
    if (cpuAllocator_ != nullptr) {
      return cpuAllocator_;
    }
  }

  {
    // Construct cpuAllocator_
    std::lock_guard<RWLock> guard(lock_);
    if (cpuAllocator_ == nullptr) {
      if (FLAGS_use_gpu) {
        cpuAllocator_ = new PoolAllocator(
            new CudaHostAllocator(), FLAGS_pool_limit_size, "cuda_host_pool");
      } else {
        cpuAllocator_ = new PoolAllocator(
            new CpuAllocator(), FLAGS_pool_limit_size, "cpu_pool");
      }
    }
    return cpuAllocator_;
  }
}


/**
 * Calculate the actual allocation size according to the required size.
 */
MemoryHandle::MemoryHandle(size_t size) : size_(size), buf_(nullptr) {
  if (size_ <= 256) {
    // Memory allocation in cuda is always aligned to at least 256 bytes.
    // In many cases it is 512 bytes.
    allocSize_ = 256;
  } else if (size_ <= 512) {
    allocSize_ = 512;
  } else if (size_ <= (1 << 16)) {
    // Allocate multiple of 1024 bytes.
    allocSize_ = (size + 1023) & ~(1023);
  } else {
    allocSize_ = size_;
  }
}

GpuMemoryHandle::GpuMemoryHandle(size_t size) : MemoryHandle(size) {
  CHECK(size != 0) << " allocate 0 bytes";
  deviceId_ = hl_get_device();
  allocator_ = StorageEngine::singleton()->getGpuAllocator(deviceId_);
  buf_ = allocator_->alloc(allocSize_);
}

GpuMemoryHandle::~GpuMemoryHandle() { allocator_->free(buf_, allocSize_); }

CpuMemoryHandle::CpuMemoryHandle(size_t size) : MemoryHandle(size) {
  CHECK(size != 0) << " allocate 0 bytes";
  allocator_ = StorageEngine::singleton()->getCpuAllocator();
  buf_ = allocator_->alloc(allocSize_);
}

CpuMemoryHandle::~CpuMemoryHandle() { allocator_->free(buf_, allocSize_); }

}  // namespace mypaddle
}  // namespace bubblefs