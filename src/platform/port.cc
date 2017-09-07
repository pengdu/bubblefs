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
//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// rocksdb/port/port_posix.cc
// tensorflow/tensorflow/core/platform/posix/port.cc

#include "platform/port.h"
#include <assert.h>
#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif
#include <malloc.h>
#if defined(__linux__) && !defined(__ANDROID__)
#include <sched.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdexcept>
#include <thread>
#include "platform/base.h"
#include "platform/cpu_info.h"
#include "platform/logging.h"
#include "platform/mem.h"
#include "platform/platform.h"
#include "platform/snappy_wrapper.h"
#include "platform/types.h"
#if TF_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif
#if TF_USE_SNAPPY
#include "snappy.h"
#endif

namespace bubblefs {
namespace port {

void InitMain(const char* usage, int* argc, char*** argv) {}

std::string Hostname() {
  char hostname[1024];
  gethostname(hostname, sizeof hostname);
  hostname[sizeof hostname - 1] = 0;
  return std::string(hostname);
}

int NumSchedulableCPUs() {
#if defined(__linux__) && !defined(__ANDROID__)
  cpu_set_t cpuset;
  if (sched_getaffinity(0, sizeof(cpu_set_t), &cpuset) == 0) {
    return CPU_COUNT(&cpuset);
  }
  perror("sched_getaffinity");
#endif
#if (defined(__APPLE__) && defined(__MACH__)) || defined(__FreeBSD__)
  unsigned int count = std::thread::hardware_concurrency();
  if (count > 0) return static_cast<int>(count);
#endif
  const int kDefaultCores = 4;  // Semi-conservative guess
  fprintf(stderr, "can't determine number of CPU cores: assuming %d\n",
          kDefaultCores);
  return kDefaultCores;
}

void* AlignedMalloc(size_t size, int minimum_alignment) {
#if defined(__ANDROID__)
  return memalign(minimum_alignment, size);
#else  // !defined(__ANDROID__)
  void* ptr = nullptr;
  // posix_memalign requires that the requested alignment be at least
  // sizeof(void*). In this case, fall back on malloc which should return
  // memory aligned to at least the size of a pointer.
  const int required_alignment = sizeof(void*);
  if (minimum_alignment < required_alignment) return Malloc(size);
#if TF_USE_JEMALLOC
  int err = jemalloc_posix_memalign(&ptr, minimum_alignment, size);
#else
  int err = posix_memalign(&ptr, minimum_alignment, size);
#endif
  if (err != 0) {
    return nullptr;
  } else {
    return ptr;
  }
#endif
}

void AlignedFree(void* aligned_memory) { Free(aligned_memory); }

void* Malloc(size_t size) {
#if TF_USE_JEMALLOC
  return jemalloc_malloc(size);
#else
  return malloc(size);
#endif
}

void* Realloc(void* ptr, size_t size) {
#ifdef TENSORFLOW_USE_JEMALLOC
  return jemalloc_realloc(ptr, size);
#else
  return realloc(ptr, size);
#endif
}

void Free(void* ptr) {
#if TF_USE_JEMALLOC
  jemalloc_free(ptr);
#else
  free(ptr);
#endif
}

void MallocExtension_ReleaseToSystem(std::size_t num_bytes) {
  // No-op.
}

std::size_t MallocExtension_GetAllocatedSize(const void* p) { return 0; }


double GetMemoryUsage() {
  FILE* fp = fopen("/proc/meminfo", "r");
  CHECK(fp) << "failed to fopen /proc/meminfo";
  size_t bufsize = 256 * sizeof(char);
  char* buf = new (std::nothrow) char[bufsize];
  CHECK(buf);
  int totalMem = -1;
  int freeMem = -1;
  int bufMem = -1;
  int cacheMem = -1;
  while (getline(&buf, &bufsize, fp) >= 0) {
    if (0 == strncmp(buf, "MemTotal", 8)) {
      if (1 != sscanf(buf, "%*s%d", &totalMem)) {
        LOG(FATAL) << "failed to get MemTotal from string: [" << buf << "]";
      }
    } else if (0 == strncmp(buf, "MemFree", 7)) {
      if (1 != sscanf(buf, "%*s%d", &freeMem)) {
        LOG(FATAL) << "failed to get MemFree from string: [" << buf << "]";
      }
    } else if (0 == strncmp(buf, "Buffers", 7)) {
      if (1 != sscanf(buf, "%*s%d", &bufMem)) {
        LOG(FATAL) << "failed to get Buffers from string: [" << buf << "]";
      }
    } else if (0 == strncmp(buf, "Cached", 6)) {
      if (1 != sscanf(buf, "%*s%d", &cacheMem)) {
        LOG(FATAL) << "failed to get Cached from string: [" << buf << "]";
      }
    }
    if (totalMem != -1 && freeMem != -1 && bufMem != -1 && cacheMem != -1) {
      break;
    }
  }
  CHECK(totalMem != -1 && freeMem != -1 && bufMem != -1 && cacheMem != -1)
      << "failed to get all information";
  fclose(fp);
  delete[] buf;
  double usedMem = 1.0 - 1.0 * (freeMem + bufMem + cacheMem) / totalMem;
  return usedMem;
}

T* AlignedAllocator::allocate(const size_type n) const {
  if (n == 0) {
    return nullptr;
  }
  if (n > max_size()) {
    throw std::length_error("AlignAllocator<T>::allocate() - Int Overflow.");
  }
  void* r = nullptr;
  CHECK_EQ(posix_memalign(&r, Alignment * 8, sizeof(T) * n), 0);
  if (r == nullptr) {
    throw std::bad_alloc();
  } else {
    return static_cast<T*>(r);
  }
}

void AdjustFilenameForLogging(std::string* filename) {
  // Nothing to do
}

bool Snappy_Compress(const char* input, size_t length, std::string* output) {
#if TF_USE_SNAPPY
  output->resize(snappy::MaxCompressedLength(length));
  size_t outlen;
  snappy::RawCompress(input, length, &(*output)[0], &outlen);
  output->resize(outlen);
  return true;
#else
  return false;
#endif
}

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result) {
#if TF_USE_SNAPPY
  return snappy::GetUncompressedLength(input, length, result);
#else
  return false;
#endif
}

bool Snappy_Uncompress(const char* input, size_t length, char* output) {
#if TF_USE_SNAPPY
  return snappy::RawUncompress(input, length, output);
#else
  return false;
#endif
}

}  // namespace port
}  // namespace bubblefs