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

// Paddle/paddle/platform/cpu_info.cc

#include "platform/paddle_cpu_info.h"
#include <unistd.h>

#include "gflags/gflags.h"

DEFINE_double(fraction_of_cpu_memory_to_use, 1,
              "Default use 100% of CPU memory for PaddlePaddle,"
              "reserve the rest for page tables, etc");

namespace bubblefs {
namespace mypaddle {
namespace platform {

inline size_t CpuTotalPhysicalMemory() {
  int64_t pages = sysconf(_SC_PHYS_PAGES);
  int64_t page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}

size_t CpuMaxAllocSize() {
  // For distributed systems, it requires configuring and limiting
  // the fraction of memory to use.
  return FLAGS_fraction_of_cpu_memory_to_use * CpuTotalPhysicalMemory();
}

size_t CpuMinChunkSize() {
  // Allow to allocate the minimum chunk size is 4 KB.
  return 1 << 12;
}

size_t CpuMaxChunkSize() {
  // Allow to allocate the maximum chunk size is roughly 3% of CPU memory.
  return CpuMaxAllocSize() / 32;
}

}  // namespace platform
}  // namespace mypaddle
}  // namespace bubblefs