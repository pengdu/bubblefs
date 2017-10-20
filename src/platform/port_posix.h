//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// See port_example.h for documentation for the following types/functions.
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

// tensorflow/tensorflow/core/platform/net.h
// rocksdb/port/port_posix.h

#ifndef BUBBLEFS_PLATFORM_PORT_POSIX_H_
#define BUBBLEFS_PLATFORM_PORT_POSIX_H_

#include <sys/types.h>
#include <assert.h>
#include <byteswap.h>
#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <limits>
#include <string>
#include <thread>
#include "platform/cpu_info.h"
#include "platform/macros.h"
#include "platform/mutex.h"

namespace bubblefs {
  
namespace internal {
int PickUnusedPortOrDie();
}  // namespace internal

namespace port {

static const int kMaxInt32 = std::numeric_limits<int32_t>::max();
static const uint64_t kMaxUint64 = std::numeric_limits<uint64_t>::max();
static const int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
static const size_t kMaxSizet = std::numeric_limits<size_t>::max();

extern unsigned page_size;
extern unsigned long page_mask;
extern unsigned page_shift;

using Thread = std::thread;

inline void AsmVolatilePause() {
#if defined(__i386__) || defined(__x86_64__)
  asm volatile("pause");
#elif defined(__aarch64__)
  asm volatile("wfe");
#elif defined(__powerpc64__)
  asm volatile("or 27,27,27");
#endif
  // it's okay for other platforms to be no-ops
}

// Pause instruction to prevent excess processor bus usage, only works in GCC
inline void AsmVolatileCpuRelax() {
  asm volatile("pause\n": : :"memory");
}

// Compile read-write barrier
inline void AsmVolatileBarrier() {
  asm volatile("": : :"memory");
}

// Make blocking ops in the pthread returns -1 and EINTR.
// Returns what pthread_kill returns.
int interrupt_pthread(pthread_t th);

// Returns -1 if not available on this platform
extern int PhysicalCoreID();

extern void *cacheline_aligned_alloc(size_t size);

extern void cacheline_aligned_free(void *memblock);

extern void Crash(const std::string& srcfile, int srcline);

extern pid_t Gettid();

extern bool get_env_bool(const char *key);
extern int get_env_int(const char *key);

extern int GetMaxOpenFiles();

/**
 * Return value: memory usage ratio (from 0-1)
 */
extern double GetMemoryUsage();

// Return the number of bytes of physical memory on the current machine.
extern int64_t AmountOfPhysicalMemory();

// Return the number of bytes of virtual memory of this process. A return
// value of zero means that there is no limit on the available virtual memory.
extern int64_t AmountOfVirtualMemory();

// Return the number of megabytes of available virtual memory, or zero if it
// is unlimited.
inline int AmountOfVirtualMemoryMB() {
  return static_cast<int>(AmountOfVirtualMemory() / 1024 / 1024);
}

// Return the number of logical processors/cores on the current machine.
int NumberOfProcessors();

// Returns the name of the host operating system.
extern std::string OperatingSystemName();

  // Returns the version of the host operating system.
extern std::string OperatingSystemVersion();

// Returns the architecture of the running operating system.
// Exact return value may differ across platforms.
// e.g. a 32-bit x86 kernel on a 64-bit capable CPU will return "x86",
//      whereas a x86-64 kernel on the same CPU will return "x86_64"
extern std::string OperatingSystemArchitecture();

// Return the hostname of the machine on which this process is running
extern std::string Hostname();

} // namespace port

} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_PORT_POSIX_H_