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

// Paddle/paddle/utils/Util.h
// tensorflow/tensorflow/core/platform/mem.h

#ifndef BUBBLEFS_PLATFORM_MEM_H_
#define BUBBLEFS_PLATFORM_MEM_H_

// TODO(cwhipkey): remove this when callers use annotations directly.
#include <limits>
#include "platform/macros.h"
#include "platform/platform.h"
#include "platform/types.h"

namespace bubblefs {
namespace port {

// Aligned allocation/deallocation. `minimum_alignment` must be a power of 2
// and a multiple of sizeof(void*).
void* AlignedMalloc(size_t size, int minimum_alignment);
void AlignedFree(void* aligned_memory);

void* Malloc(size_t size);
void* Realloc(void* ptr, size_t size);
void Free(void* ptr);

// Tries to release num_bytes of free memory back to the operating
// system for reuse.  Use this routine with caution -- to get this
// memory back may require faulting pages back in by the OS, and
// that may be slow.
//
// Currently, if a malloc implementation does not support this
// routine, this routine is a no-op.
void MallocExtension_ReleaseToSystem(std::size_t num_bytes);

// Returns the actual number N of bytes reserved by the malloc for the
// pointer p.  This number may be equal to or greater than the number
// of bytes requested when p was allocated.
//
// This routine is just useful for statistics collection.  The
// client must *not* read or write from the extra bytes that are
// indicated by this call.
//
// Example, suppose the client gets memory by calling
//    p = malloc(10)
// and GetAllocatedSize(p) may return 16.  The client must only use the
// first 10 bytes p[0..9], and not attempt to read or write p[10..15].
//
// Currently, if a malloc implementation does not support this
// routine, this routine returns 0.
std::size_t MallocExtension_GetAllocatedSize(const void* p);

/**
 * Return value: memory usage ratio (from 0-1)
 */
double GetMemoryUsage();

/**
 * std compatible allocator with memory alignment.
 * @tparam T type of allocator elements.
 * @tparam Alignment the alignment in bytes.
 */
template <typename T, size_t Alignment>
class AlignedAllocator {
public:
  /// std campatible typedefs.
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;

  T* address(T& r) const { return &r; }

  const T* address(const T& r) const { return &r; }

  size_t max_size() const {
    return std::numeric_limits<size_t>::max() / sizeof(T);
  }

  template <typename U>
  struct rebind {
    typedef AlignedAllocator<U, Alignment> other;
  };

  bool operator==(const AlignedAllocator& other) const { return true; }

  bool operator!=(const AlignedAllocator& other) const {
    return !(*this == &other);
  }

  void construct(const T* p, const T& t) const {
    void* pv = const_cast<T*>(p);
    new (pv) T(t);
  }

  void deallocate(const T* p, const size_type n) const {
    (void)(n);  // UNUSED n
    free(const_cast<T*>(p));
  }

  void destroy(const T* p) const { p->~T(); }

  AlignedAllocator() {}
  ~AlignedAllocator() {}

  AlignedAllocator(const AlignedAllocator&) {}
  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

  /**
   * @brief allocate n elements of type T, the first address is aligned by
   *        Alignment bytes.
   * @param n element count.
   * @return begin address of allocated buffer
   * @throw std::length_error for n * sizeof(T) is overflowed.
   * @throw std::bad_alloc
   */
  T* allocate(const size_type n) const;
  
  template <typename U>
  T* allocate(const std::size_t n, const U* /* const hint */) const {
    return this->allocate(n);
  }

private:
  AlignedAllocator& operator=(const AlignedAllocator&);  // disable
};

}  // namespace port
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_MEM_H_