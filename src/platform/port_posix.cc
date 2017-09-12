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

// rocksdb/port/port_posix.cc
// tensorflow/tensorflow/core/platform/posix/net.cc
// tensorflow/tensorflow/core/platform/posix/port.cc

#include "platform/port_posix.h"
#include <netinet/in.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <assert.h>
#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif
#include <errno.h>
#include <malloc.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <unordered_set>
#include "platform/base.h"
#include "platform/logging.h"
#include "platform/mem.h"
#include "utils/strcat.h"

namespace bubblefs {
namespace internal {

namespace { // namespace anonymous
bool IsPortAvailable(int* port, bool is_tcp) {
  const int protocol = is_tcp ? IPPROTO_TCP : 0;
  const int fd = socket(AF_INET, is_tcp ? SOCK_STREAM : SOCK_DGRAM, protocol);

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);
  int actual_port;

  CHECK_GE(*port, 0);
  CHECK_LE(*port, 65535);
  if (fd < 0) {
    LOG(ERROR) << "socket() failed: " << strerror(errno);
    return false;
  }

  // SO_REUSEADDR lets us start up a server immediately after it exists.
  int one = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
    LOG(ERROR) << "setsockopt() failed: " << strerror(errno);
    close(fd);
    return false;
  }

  // Try binding to port.
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(*port));
  if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    LOG(WARNING) << "bind(port=" << *port << ") failed: " << strerror(errno);
    close(fd);
    return false;
  }

  // Get the bound port number.
  if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&addr), &addr_len) <
      0) {
    LOG(WARNING) << "getsockname() failed: " << strerror(errno);
    close(fd);
    return false;
  }
  CHECK_LE(addr_len, sizeof(addr));
  actual_port = ntohs(addr.sin_port);
  CHECK_GT(actual_port, 0);
  if (*port == 0) {
    *port = actual_port;
  } else {
    CHECK_EQ(*port, actual_port);
  }
  close(fd);
  return true;
}

const int kNumRandomPortsToPick = 100;
const int kMaximumTrials = 1000;

}  // namespace anonymous

int PickUnusedPortOrDie() {
  static std::unordered_set<int> chosen_ports;

  // Type of port to first pick in the next iteration.
  bool is_tcp = true;
  int trial = 0;
  while (true) {
    int port;
    trial++;
    CHECK_LE(trial, kMaximumTrials)
        << "Failed to pick an unused port for testing.";
    if (trial == 1) {
      port = getpid() % (65536 - 30000) + 30000;
    } else if (trial <= kNumRandomPortsToPick) {
      port = rand() % (65536 - 30000) + 30000;
    } else {
      port = 0;
    }

    if (chosen_ports.find(port) != chosen_ports.end()) {
      continue;
    }
    if (!IsPortAvailable(&port, is_tcp)) {
      continue;
    }

    CHECK_GT(port, 0);
    if (!IsPortAvailable(&port, !is_tcp)) {
      is_tcp = !is_tcp;
      continue;
    }

    chosen_ports.insert(port);
    return port;
  }

  return 0;
}

}  // namespace internal
}  // namespace bubblefs

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
  
static int PthreadCall(const char* label, int result) {
  if (result != 0 && result != ETIMEDOUT) {
    fprintf(stderr, "pthread %s: %s\n", label, strerror(result));
    abort();
  }
  return result;
}

Mutex::Mutex(bool adaptive) {
#ifdef TF_USE_ADAPTIVE_MUTEX
  if (!adaptive) {
    PthreadCall("init mutex", pthread_mutex_init(&mu_, nullptr));
  } else {
    pthread_mutexattr_t mutex_attr;
    PthreadCall("init mutex attr", pthread_mutexattr_init(&mutex_attr));
    PthreadCall("set mutex attr",
                pthread_mutexattr_settype(&mutex_attr,
                                          PTHREAD_MUTEX_ADAPTIVE_NP));
    PthreadCall("init mutex", pthread_mutex_init(&mu_, &mutex_attr));
    PthreadCall("destroy mutex attr",
                pthread_mutexattr_destroy(&mutex_attr));
  }
#else
  PthreadCall("init mutex", pthread_mutex_init(&mu_, nullptr));
#endif // TF_USE_ADAPTIVE_MUTEX
}

Mutex::~Mutex() { PthreadCall("destroy mutex", pthread_mutex_destroy(&mu_)); }

void Mutex::Lock() {
  PthreadCall("lock", pthread_mutex_lock(&mu_));
#ifndef NDEBUG
  locked_ = true;
#endif
}

void Mutex::Unlock() {
#ifndef NDEBUG
  locked_ = false;
#endif
  PthreadCall("unlock", pthread_mutex_unlock(&mu_));
}

void Mutex::AssertHeld() {
#ifndef NDEBUG
  assert(locked_);
#endif
}

CondVar::CondVar(Mutex* mu)
    : mu_(mu) {
    PthreadCall("init cv", pthread_cond_init(&cv_, nullptr));
}

CondVar::~CondVar() { PthreadCall("destroy cv", pthread_cond_destroy(&cv_)); }

void CondVar::Wait() {
#ifndef NDEBUG
  mu_->locked_ = false;
#endif
  PthreadCall("wait", pthread_cond_wait(&cv_, &mu_->mu_));
#ifndef NDEBUG
  mu_->locked_ = true;
#endif
}

bool CondVar::TimedWait(uint64_t abs_time_us) {
  struct timespec ts;
  ts.tv_sec = static_cast<time_t>(abs_time_us / 1000000);
  ts.tv_nsec = static_cast<suseconds_t>((abs_time_us % 1000000) * 1000);

#ifndef NDEBUG
  mu_->locked_ = false;
#endif
  int err = pthread_cond_timedwait(&cv_, &mu_->mu_, &ts);
#ifndef NDEBUG
  mu_->locked_ = true;
#endif
  if (err == ETIMEDOUT) {
    return true;
  }
  if (err != 0) {
    PthreadCall("timedwait", err);
  }
  return false;
}

void CondVar::Signal() {
  PthreadCall("signal", pthread_cond_signal(&cv_));
}

void CondVar::SignalAll() {
  PthreadCall("broadcast", pthread_cond_broadcast(&cv_));
}

void CondVar::Broadcast() {
  PthreadCall("broadcast", pthread_cond_broadcast(&cv_));
}

RWMutex::RWMutex() {
  PthreadCall("init rwmutex", pthread_rwlock_init(&mu_, nullptr));
}

RWMutex::~RWMutex() { PthreadCall("destroy rwmutex", pthread_rwlock_destroy(&mu_)); }

void RWMutex::ReadLock() { PthreadCall("read lock", pthread_rwlock_rdlock(&mu_)); }

void RWMutex::WriteLock() { PthreadCall("write lock", pthread_rwlock_wrlock(&mu_)); }

void RWMutex::ReadUnlock() { PthreadCall("read unlock", pthread_rwlock_unlock(&mu_)); }

void RWMutex::WriteUnlock() { PthreadCall("write unlock", pthread_rwlock_unlock(&mu_)); }

int PhysicalCoreID() {
#if defined(__x86_64__) && (__GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ >= 22))
  // sched_getcpu uses VDSO getcpu() syscall since 2.22. I believe Linux offers VDSO
  // support only on x86_64. This is the fastest/preferred method if available.
  int cpuno = sched_getcpu();
  if (cpuno < 0) {
    return -1;
  }
  return cpuno;
#elif defined(__x86_64__) || defined(__i386__)
  // clang/gcc both provide cpuid.h, which defines __get_cpuid(), for x86_64 and i386.
  unsigned eax, ebx = 0, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return -1;
  }
  return ebx >> 24;
#else
  // give up, the caller can generate a random number or something.
  return -1;
#endif
}

void InitOnce(OnceType* once, void (*initializer)()) {
  PthreadCall("once", pthread_once(once, initializer));
}

void Crash(const std::string& srcfile, int srcline) {
  fprintf(stdout, "Crashing at %s:%d\n", srcfile.c_str(), srcline);
  fflush(stdout);
  kill(getpid(), SIGTERM);
}

int GetMaxOpenFiles() {
#if defined(RLIMIT_NOFILE)
  struct rlimit no_files_limit;
  if (getrlimit(RLIMIT_NOFILE, &no_files_limit) != 0) {
    return -1;
  }
  // protect against overflow
  if (no_files_limit.rlim_cur >= std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(no_files_limit.rlim_cur);
#endif
  return -1;
}

void *cacheline_aligned_alloc(size_t size) {
#if __GNUC__ < 5 && defined(__SANITIZE_ADDRESS__)
  return malloc(size);
#elif defined(_ISOC11_SOURCE)
  return aligned_alloc(CACHE_LINE_SIZE, size);
#elif ( _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600 || defined(__APPLE__))
  void *m;
  errno = posix_memalign(&m, CACHE_LINE_SIZE, size);
  return errno ? NULL : m;
#else
  return malloc(size);
#endif
}

void cacheline_aligned_free(void *memblock) {
  free(memblock);
}

constexpr int kExecuteCMDBufLength = 204800;  
  
int ExecuteCMD(const char* cmd, char* result) {
  char bufPs[kExecuteCMDBufLength];
  char ps[kExecuteCMDBufLength] = {0};
  FILE* ptr = nullptr;
  strncpy(ps, cmd, kExecuteCMDBufLength);
  if ((ptr = popen(ps, "r")) != nullptr) {
    size_t count = fread(bufPs, 1, kExecuteCMDBufLength, ptr);
    memcpy(result,
           bufPs,
           count - 1);  // why count-1: remove the '\n' at the end
    result[count] = 0;
    pclose(ptr);
    ptr = nullptr;
    return count - 1;
  } else {
    LOG(FATAL) << "ExecuteCMD popen failed";
    return -1;
  }
}

std::string Hostname() {
  char hostname[1024];
  gethostname(hostname, sizeof hostname);
  hostname[sizeof hostname - 1] = 0;
  return std::string(hostname);
}

}  // namespace port
}  // namespace bubblefs