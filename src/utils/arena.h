/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tensorflow/tensorflow/core/lib/core/arena.h

// TODO(vrv): Switch this to an open-sourced version of Arena.

#ifndef BUBBLEFS_UTILS_ARENA_H_
#define BUBBLEFS_UTILS_ARENA_H_

#include <assert.h>
#include <vector>
#include "platform/logging.h"
#include "platform/macros.h"
#include "platform/types.h"
#include "utils/allocator.h"

namespace bubblefs {
namespace core {
  
// This class is "thread-compatible": different threads can access the
// arena at the same time without locking, as long as they use only
// const methods.
class Arena : public Allocator {
 public:
  // Allocates a thread-compatible arena with the specified block size.
  explicit Arena(const size_t block_size = kMinBlockSize);
  virtual ~Arena();
  
  static const size_t kInlineSize = 2048;
  static const size_t kMinBlockSize;
  static const size_t kMaxBlockSize;

  virtual char* Alloc(const size_t size) override {
    return reinterpret_cast<char*>(GetMemory(size, 1));
  }

  virtual char* AllocAligned(const size_t size, const size_t alignment) override {
    return reinterpret_cast<char*>(GetMemory(size, alignment));
  }
  
  virtual size_t BlockSize() const override { return block_size_; }

  void Reset();

 protected:
  bool SatisfyAlignment(const size_t alignment);
  void MakeNewBlock(const uint32 alignment);
  void* GetMemoryFallback(const size_t size, const int align);
  void* GetMemory(const size_t size, const int align) {
    assert(remaining_ <= block_size_);                  // an invariant
    if (size > 0 && size < remaining_ && align == 1) {  // common case
      void* result = freestart_;
      freestart_ += size;
      remaining_ -= size;
      return result;
    }
    return GetMemoryFallback(size, align);
  }

  size_t remaining_;

 private:
  struct AllocatedBlock {
    char* mem;
    size_t size;
  };

  // Allocate new block of at least block_size, with the specified
  // alignment.
  // The returned AllocatedBlock* is valid until the next call to AllocNewBlock
  // or Reset (i.e. anything that might affect overflow_blocks_).
  AllocatedBlock* AllocNewBlock(const size_t block_size,
                                const uint32 alignment);

  const size_t block_size_;
  char* freestart_;  // beginning of the free space in most recent block
  char* freestart_when_empty_;  // beginning of the free space when we're empty
  // STL vector isn't as efficient as it could be, so we use an array at first
  size_t blocks_alloced_;  // how many of the first_blocks_ have been alloced
  AllocatedBlock first_blocks_[16];  // the length of this array is arbitrary
  // if the first_blocks_ aren't enough, expand into overflow_blocks_.
  std::vector<AllocatedBlock>* overflow_blocks_;

  void FreeBlocks();  // Frees all except first block

  DISALLOW_COPY_AND_ASSIGN(Arena);
};

}  // namespace core
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_ARENA_H_
