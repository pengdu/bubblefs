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

// Paddle/paddle/memory/memory.h
// Paddle/paddle/memory/memory.cc
// Paddle/paddle/memory/memcpy.h
// Paddle/paddle/memory/memcpy.cc

#pragma once

#include <cstring>
#include "utils/paddle_device_context.h"
#include "utils/paddle_buddy_allocator.h"

// Region-based Heterogeneous Memory Management
// Usage:
// To allocate 4KB CPU memory:
// p = memory::Alloc(platform::CPUPlace(), 4*1024);
// To allocate 4KB memory on the 3rd GPU:
// p = memory::Alloc(platform::CUDAPlace(2), 4*1024);
// To free memory and check the so-far used amount of memory on a place:
// auto pl = platform::CUDAPlace(0);
// p = memory::Alloc(pl, 4*1024);
// cout << memory::Used(pl);
// memory::Free(pl, p);

DECLARE_double(fraction_of_gpu_memory_to_use);

namespace bubblefs {
namespace mypaddle {
namespace memory {

/**
 * \brief   Allocate memory block in one place.
 *
 * \param[in]  place  Allocation place (CPU or GPU).
 * \param[in]  size   Allocation size.
 *
 * \return  Allocated memory block address.
 *
 * \note    If return nullptr, it indicates memory allocation failed
 *          because insufficient memory in current system. When Alloc
 *          function is invoked, you must check the returned memory
 *          address is valid or not.
 */
template <typename Place>
void* Alloc(Place place, size_t size);

/**
 * \brief   Free memory block in one place.
 *
 * \param[in]  place  Allocation place (CPU or GPU).
 * \param[in]  ptr    Memory block address to free.
 *
 */
template <typename Place>
void Free(Place place, void* ptr);

/**
 * \brief   Total size of used memory in one place.
 *
 * \param[in]  place  Allocation place (CPU or GPU).
 *
 */
template <typename Place>
size_t Used(Place place);

/**
 * \brief   Free memory block in one place.
 *
 * \note    In some cases, custom deleter is used to
 *          deallocate the memory automatically for
 *          std::unique_ptr<T> in tensor.h.
 *
 */
template <typename T, typename Place>
class PODDeleter {
  static_assert(std::is_pod<T>::value, "T must be POD");

 public:
  explicit PODDeleter(Place place) : place_(place) {}
  void operator()(T* ptr) { Free(place_, static_cast<void*>(ptr)); }

 private:
  Place place_;
};

/**
 * \brief   Copy memory from one place to another place.
 *
 * \param[in]  DstPlace Destination allocation place (CPU).
 * \param[in]  dst      Destination memory address.
 * \param[in]  SrcPlace Source allocation place (CPU).
 * \param[in]  src      Source memory address.
 * \param[in]  num      memory size in bytes to copy.
 *
 */
template <typename DstPlace, typename SrcPlace>
void Copy(DstPlace, void* dst, SrcPlace, const void* src, size_t num);

#ifdef PADDLE_WITH_CUDA

/**
 * \brief   Copy memory from one place to another place.
 *
 * \param[in]  DstPlace Destination allocation place (CPU or GPU).
 * \param[in]  dst      Destination memory address.
 * \param[in]  SrcPlace Source allocation place (CPU or GPU).
 * \param[in]  src      Source memory address.
 * \param[in]  num      memory size in bytes to copy.
 * \param[in]  stream   CUDA stream.
 *
 * \note    For GPU memory copy, CUDA stream need to be specified
 *          for asynchronously memory copy.
 *
 */
template <typename DstPlace, typename SrcPlace>
void Copy(DstPlace, void* dst, SrcPlace, const void* src, size_t num,
          cudaStream_t stream);

#endif

using BuddyAllocator = detail::BuddyAllocator;

BuddyAllocator* GetCPUBuddyAllocator() {
  static detail::BuddyAllocator* a = nullptr;
  if (a == nullptr) {
    a = new detail::BuddyAllocator(new detail::CPUAllocator,
                                   platform::CpuMinChunkSize(),
                                   platform::CpuMaxChunkSize());
  }
  return a;
}

template <>
void* Alloc<platform::CPUPlace>(platform::CPUPlace place, size_t size) {
  VLOG(10) << "Allocate " << size << " bytes on " << platform::Place(place);
  void* p = GetCPUBuddyAllocator()->Alloc(size);
  VLOG(10) << "  pointer=" << p;
  return p;
}

template <>
void Free<platform::CPUPlace>(platform::CPUPlace place, void* p) {
  VLOG(10) << "Free pointer=" << p << " on " << platform::Place(place);
  GetCPUBuddyAllocator()->Free(p);
}

template <>
size_t Used<platform::CPUPlace>(platform::CPUPlace place) {
  return GetCPUBuddyAllocator()->Used();
}

#ifdef PADDLE_WITH_CUDA

BuddyAllocator* GetGPUBuddyAllocator(int gpu_id) {
  static BuddyAllocator** as = NULL;
  if (as == NULL) {
    int gpu_num = platform::GetCUDADeviceCount();
    as = new BuddyAllocator*[gpu_num];
    for (int gpu = 0; gpu < gpu_num; gpu++) {
      as[gpu] = nullptr;
    }
  }
  platform::SetDeviceId(gpu_id);
  if (!as[gpu_id]) {
    as[gpu_id] = new BuddyAllocator(new detail::GPUAllocator,
                                    platform::GpuMinChunkSize(),
                                    platform::GpuMaxChunkSize());
    VLOG(10) << "\n\nNOTE: each GPU device use "
             << FLAGS_fraction_of_gpu_memory_to_use * 100
             << "% of GPU memory.\n"
             << "You can set GFlags environment variable '"
             << "FLAGS_fraction_of_gpu_memory_to_use"
             << "' to change the fraction of GPU usage.\n\n";
  }
  return as[gpu_id];
}

template <>
size_t Used<platform::CUDAPlace>(platform::CUDAPlace place) {
  return GetGPUBuddyAllocator(place.device)->Used();
}

template <>
void* Alloc<platform::CUDAPlace>(platform::CUDAPlace place, size_t size) {
  auto* buddy_allocator = GetGPUBuddyAllocator(place.device);
  auto* ptr = buddy_allocator->Alloc(size);
  if (ptr == nullptr) {
    int cur_dev = platform::GetCurrentDeviceId();
    platform::SetDeviceId(place.device);
    size_t avail, total;
    platform::GpuMemoryUsage(avail, total);
    LOG(WARNING) << "Cannot allocate " << size << " bytes in GPU "
                 << place.device << ", available " << avail << " bytes";
    LOG(WARNING) << "total " << total;
    LOG(WARNING) << "GpuMinChunkSize " << platform::GpuMinChunkSize();
    LOG(WARNING) << "GpuMaxChunkSize " << platform::GpuMaxChunkSize();
    LOG(WARNING) << "GPU memory used: " << Used<platform::CUDAPlace>(place);
    platform::SetDeviceId(cur_dev);
  }
  return ptr;
}

template <>
void Free<platform::CUDAPlace>(platform::CUDAPlace place, void* p) {
  GetGPUBuddyAllocator(place.device)->Free(p);
}

#endif

template <>
void Copy<platform::CPUPlace, platform::CPUPlace>(platform::CPUPlace, void* dst,
                                                  platform::CPUPlace,
                                                  const void* src, size_t num) {
  std::memcpy(dst, src, num);
}

#ifdef PADDLE_WITH_CUDA
template <>
void Copy<platform::CPUPlace, platform::CUDAPlace>(
    platform::CPUPlace dst_place, void* dst, platform::CUDAPlace src_place,
    const void* src, size_t num, cudaStream_t stream) {
  platform::SetDeviceId(src_place.device);
  platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToHost, stream);
}

template <>
void Copy<platform::CUDAPlace, platform::CPUPlace>(
    platform::CUDAPlace dst_place, void* dst, platform::CPUPlace src_place,
    const void* src, size_t num, cudaStream_t stream) {
  platform::SetDeviceId(dst_place.device);
  platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyHostToDevice, stream);
}

template <>
void Copy<platform::CUDAPlace, platform::CUDAPlace>(
    platform::CUDAPlace dst_place, void* dst, platform::CUDAPlace src_place,
    const void* src, size_t num, cudaStream_t stream) {
  if (dst_place == src_place) {
    platform::SetDeviceId(src_place.device);
    platform::GpuMemcpyAsync(dst, src, num, cudaMemcpyDeviceToDevice, stream);
  } else {
    platform::GpuMemcpyPeer(dst, dst_place.device, src, src_place.device, num,
                            stream);
  }
}

#endif

}  // namespace memory
}  // namespace mypaddle
}  // namespace bubblefs