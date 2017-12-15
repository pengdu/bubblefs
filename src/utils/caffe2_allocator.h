/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/core/allocator.h

#ifndef BUBBLEFS_UTILS_CAFFE2_ALLOCATOR_H_
#define BUBBLEFS_UTILS_CAFFE2_ALLOCATOR_H_

#include <memory>
#include <mutex>
#include <unordered_map>

#include "platform/base_error.h"

namespace bubblefs {
namespace mycaffe2 {

extern bool FLAGS_caffe2_report_cpu_memory_usage;
extern bool FLAGS_caffe2_cpu_allocator_do_zero_fill;  
  
// Use 32-byte alignment should be enough for computation up to AVX512.
constexpr size_t gCaffe2Alignment = 32;

using MemoryDeleter = void (*)(void*);

// A helper function that is basically doing nothing.
void NoDelete(void*);

// A virtual allocator class to do memory allocation and deallocation.
struct CPUAllocator {
  CPUAllocator() {}
  virtual ~CPUAllocator() noexcept {}
  virtual std::pair<void*, MemoryDeleter> New(size_t nbytes) = 0;
  virtual MemoryDeleter GetDeleter() = 0;
};

// A virtual struct that is used to report Caffe2's memory allocation and
// deallocation status
class MemoryAllocationReporter {
 public:
  MemoryAllocationReporter() : allocated_(0) {}
  void New(void* ptr, size_t nbytes);
  void Delete(void* ptr);

 private:
  std::mutex mutex_;
  std::unordered_map<void*, size_t> size_table_;
  size_t allocated_;
};

struct DefaultCPUAllocator final : CPUAllocator {
  DefaultCPUAllocator() {}
  ~DefaultCPUAllocator() override {}
  std::pair<void*, MemoryDeleter> New(size_t nbytes) override {
    void* data = nullptr;
#ifdef __ANDROID__
    data = memalign(gCaffe2Alignment, nbytes);
#elif defined(_MSC_VER)
    data = _aligned_malloc(nbytes, gCaffe2Alignment);
#else
    PANIC_ENFORCE_EQ(posix_memalign(&data, gCaffe2Alignment, nbytes), 0);
#endif
    PANIC_ENFORCE(data, "data is NULL");
    if (FLAGS_caffe2_cpu_allocator_do_zero_fill) {
      memset(data, 0, nbytes);
    }
    return {data, Delete};
  }

#ifdef _MSC_VER
  static void Delete(void* data) {
    _aligned_free(data);
  }
#else
  static void Delete(void* data) {
    free(data);
  }
#endif

  MemoryDeleter GetDeleter() override {
    return Delete;
  }
};

// Get the CPU Alloctor.
CPUAllocator* GetCPUAllocator();
// Sets the CPU allocator to the given allocator: the caller gives away the
// ownership of the pointer.
void SetCPUAllocator(CPUAllocator* alloc);

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_ALLOCATOR_H_