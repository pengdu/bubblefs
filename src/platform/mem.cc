//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

// tensorflow/tensorflow/core/platform/posix/port.cc

#include "platform/mem.h"
#include "platform/base.h"
#ifdef TF_USE_JEMALLOC
#include "jemalloc/jemalloc.h"
#endif
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include "platform/logging.h"

namespace bubblefs {
namespace port {

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
  int err = je_posix_memalign(&ptr, minimum_alignment, size);
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
  return je_malloc(size);
#else
  return malloc(size);
#endif
}

void* Realloc(void* ptr, size_t size) {
#ifdef TF_USE_JEMALLOC
  return je_realloc(ptr, size);
#else
  return realloc(ptr, size);
#endif
}

void Free(void* ptr) {
#if TF_USE_JEMALLOC
  je_free(ptr);
#else
  free(ptr);
#endif
}

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

}  // namespace port
}  // namespace bubblefs