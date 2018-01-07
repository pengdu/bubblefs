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

// Paddle/paddle/math/Allocator.h
// Paddle/paddle/math/PoolAllocator.h

#ifndef BUBBLEFS_UTILS_PADDLE_POOL_ALLOCATOR_H_
#define BUBBLEFS_UTILS_PADDLE_POOL_ALLOCATOR_H_

#include <stdlib.h>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "platform/base_error.h"

//#include "hl_gpu.h"

namespace bubblefs {
namespace mypaddle {

/**
 * @brief Allocator base class.
 *
 * This is the base class of all Allocator class.
 */
class Allocator {
public:
  virtual ~Allocator() {}
  virtual void* alloc(size_t size) = 0;
  virtual void free(void* ptr) = 0;
  virtual std::string getName() = 0;
};

/**
 * @brief CPU allocator implementation.
 */
class CpuAllocator : public Allocator {
public:
  ~CpuAllocator() {}

  /**
   * @brief Aligned allocation on CPU.
   * @param size Size to be allocated.
   * @return Pointer to the allocated memory
   */
  virtual void* alloc(size_t size) {
    void* ptr;
#ifdef PADDLE_WITH_MKLDNN
    // refer to https://github.com/01org/mkl-dnn/blob/master/include/mkldnn.hpp
    // memory alignment
    PANIC_ENFORCE_EQ(posix_memalign(&ptr, 4096ul, size), 0);
#else
    PANIC_ENFORCE_EQ(posix_memalign(&ptr, 32ul, size), 0);
#endif
    PRINTF_CHECK(ptr, "Fail to allocate CPU memory: size=%zu", size);
    return ptr;
  }

  /**
   * @brief Free the memory space.
   * @param ptr  Pointer to be free.
   */
  virtual void free(void* ptr) {
    if (ptr) {
      ::free(ptr);
    }
  }

  virtual std::string getName() { return "cpu_alloc"; }
};

/**
 * @brief GPU allocator implementation.
 */
class GpuAllocator : public Allocator {
public:
  ~GpuAllocator() {}

  /**
   * @brief Allocate GPU memory.
   * @param size Size to be allocated.
   * @return Pointer to the allocated memory
   */
  virtual void* alloc(size_t size) {
    void* ptr = NULL; // hl_malloc_device(size);
    //CHECK(ptr) << "Fail to allocate GPU memory " << size << " bytes";
    return ptr;
  }

  /**
   * @brief Free the GPU memory.
   * @param ptr  Pointer to be free.
   */
  virtual void free(void* ptr) {
    if (ptr) {
      //hl_free_mem_device(ptr);
    }
  }

  virtual std::string getName() { return "gpu_alloc"; }
};

/**
 * @brief CPU pinned memory allocator implementation.
 */
class CudaHostAllocator : public Allocator {
public:
  ~CudaHostAllocator() {}

  /**
   * @brief Allocate pinned memory.
   * @param size Size to be allocated.
   * @return Pointer to the allocated memory
   */
  virtual void* alloc(size_t size) {
    void* ptr = NULL; // hl_malloc_host(size);
    //CHECK(ptr) << "Fail to allocate pinned memory " << size << " bytes";
    return ptr;
  }

  /**
   * @brief Free the pinned memory.
   * @param ptr  Pointer to be free.
   */
  virtual void free(void* ptr) {
    if (ptr) {
      //hl_free_mem_host(ptr);
    }
  }

  virtual std::string getName() { return "cuda_host_alloc"; }
};

/**
 * @brief Memory pool allocator implementation.
 */
class PoolAllocator {
public:
  /**
   * @brief constructor.
   * @param allocator a Allocator object.
   * @param sizeLimit The maximum size memory can be managed,
   * if sizeLimit == 0, the pool allocator is a simple wrapper of allocator.
   */
  PoolAllocator(Allocator* allocator,
                size_t sizeLimit = 0,
                const std::string& name = "pool");

  /**
   * @brief destructor.
   */
  ~PoolAllocator();

  void* alloc(size_t size);
  void free(void* ptr, size_t size);
  std::string getName() { return name_; }

private:
  void freeAll();
  void printAll();
  std::unique_ptr<Allocator> allocator_;
  std::mutex mutex_;
  std::unordered_map<size_t, std::vector<void*>> pool_;
  size_t sizeLimit_;
  size_t poolMemorySize_;
  std::string name_;
};

}  // namespace paddle
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PADDLE_POOL_ALLOCATOR_H_