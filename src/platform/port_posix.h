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
#include <dirent.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <limits>
#include <string>
#include "platform/base.h"
#include "platform/macros.h"
#include "platform/cpu_info.h"

// size_t printf formatting named in the manner of C99 standard formatting
// strings such as PRIu64
// in fact, we could use that one
#define TF_PRIszt "zu"

#define __declspec(S)

namespace bubblefs {
namespace internal {

int PickUnusedPortOrDie();

}  // namespace internal
}  // namespace bubblefs

namespace bubblefs {
namespace port {

static const int kMaxInt32 = std::numeric_limits<int32_t>::max();
static const uint64_t kMaxUint64 = std::numeric_limits<uint64_t>::max();
static const int64_t kMaxInt64 = std::numeric_limits<int64_t>::max();
static const size_t kMaxSizet = std::numeric_limits<size_t>::max();  

class CondVar;

class Mutex {
 public:
// We want to give users opportunity to default all the mutexes to adaptive if
// not specified otherwise. This enables a quick way to conduct various
// performance related experiements.
//
// NB! Support for adaptive mutexes is turned on by definining
// ROCKSDB_PTHREAD_ADAPTIVE_MUTEX during the compilation. If you use RocksDB
// build environment then this happens automatically; otherwise it's up to the
// consumer to define the identifier.
#ifdef TF_USE_ADAPTIVE_MUTEX
  explicit Mutex(bool adaptive = true);
#else
  explicit Mutex(bool adaptive = false);
#endif
  ~Mutex();

  void Lock();
  void Unlock();
  // this will assert if the mutex is not locked
  // it does NOT verify that mutex is held by a calling thread
  void AssertHeld();

 private:
  friend class CondVar;
  pthread_mutex_t mu_;
#ifndef NDEBUG
  bool locked_;
#endif
  TF_DISALLOW_COPY_AND_ASSIGN(Mutex);
};

class RWMutex {
 public:
  RWMutex();
  ~RWMutex();

  void ReadLock();
  void WriteLock();
  void ReadUnlock();
  void WriteUnlock();
  void AssertHeld() { }

 private:
  pthread_rwlock_t mu_; // the underlying platform mutex

  TF_DISALLOW_COPY_AND_ASSIGN(RWMutex);
};

class CondVar {
 public:
  explicit CondVar(Mutex* mu);
  ~CondVar();
  void Wait();
  // Timed condition wait.  Returns true if timeout occurred.
  bool TimedWait(uint64_t abs_time_us);
  void Signal();
  void SignalAll();
  void Broadcast();
 private:
  pthread_cond_t cv_;
  Mutex* mu_;
};

static inline void AsmVolatilePause() {
#if defined(__i386__) || defined(__x86_64__)
  asm volatile("pause");
#elif defined(__aarch64__)
  asm volatile("wfe");
#elif defined(__powerpc64__)
  asm volatile("or 27,27,27");
#endif
  // it's okay for other platforms to be no-ops
}

// Returns -1 if not available on this platform
extern int PhysicalCoreID();

typedef pthread_once_t OnceType;
extern void InitOnce(OnceType* once, void (*initializer)());

#ifndef CACHE_LINE_SIZE
  #if defined(__s390__)
    #define CACHE_LINE_SIZE 256U
  #elif defined(__powerpc__) || defined(__aarch64__)
    #define CACHE_LINE_SIZE 128U
  #else
    #define CACHE_LINE_SIZE 64U
  #endif
#endif

extern void *cacheline_aligned_alloc(size_t size);

extern void cacheline_aligned_free(void *memblock);

#define ALIGN_AS(n) alignas(n)

#define PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)

extern void Crash(const std::string& srcfile, int srcline);

extern int GetMaxOpenFiles();

int ExecuteCMD(const char* cmd, char* result);

// Return the hostname of the machine on which this process is running
std::string Hostname();

} // namespace port
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_PORT_POSIX_H_