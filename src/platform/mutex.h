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

// tensorflow/tensorflow/core/platform/mutex.h
// tensorflow/tensorflow/core/platform/default/mutex.h

#ifndef BUBBLEFS_PLATFORM_MUTEX_H_
#define BUBBLEFS_PLATFORM_MUTEX_H_

#include <pthread.h>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include "platform/platform.h"
#include "platform/thread_annotations.h"
#include "platform/types.h"

namespace bubblefs {

#undef mutex_lock

enum LinkerInitialized { LINKER_INITIALIZED };

// A class that wraps around the std::mutex implementation, only adding an
// additional LinkerInitialized constructor interface.
class LOCKABLE mutex : public std::mutex {
 public:
  mutex() {}
  // The default implementation of std::mutex is safe to use after the linker
  // initializations
  explicit mutex(LinkerInitialized x) {}

  void lock() ACQUIRE() { std::mutex::lock(); }
  bool try_lock() EXCLUSIVE_TRYLOCK_FUNCTION(true) {
    return std::mutex::try_lock();
  };
  void unlock() RELEASE() { std::mutex::unlock(); }
};

class SCOPED_LOCKABLE mutex_lock : public std::unique_lock<std::mutex> {
 public:
  mutex_lock(class mutex& m) ACQUIRE(m) : std::unique_lock<std::mutex>(m) {}
  mutex_lock(class mutex& m, std::try_to_lock_t t) ACQUIRE(m)
      : std::unique_lock<std::mutex>(m, t) {}
  mutex_lock(mutex_lock&& ml) noexcept
      : std::unique_lock<std::mutex>(std::move(ml)) {}
  ~mutex_lock() RELEASE() {}
};

// Catch bug where variable name is omitted, e.g. mutex_lock (mu);
#define mutex_lock(x) static_assert(0, "mutex_lock_decl_missing_var_name");

namespace port { 

class CondVar;

// A Mutex represents an exclusive lock.
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
  Mutex();
  Mutex(bool adaptive);
  ~Mutex();

  void Lock(const char* msg = nullptr, int64_t msg_threshold = 5000);
  bool TryLock();
  bool TimedLock(long _millisecond);
  void Unlock();
  bool IsLocked();
  // this will assert if the mutex is not locked
  // it does NOT verify that mutex is held by a calling thread
  void AssertHeld();

 private:
  void AfterLock(const char* msg = nullptr, int64_t msg_threshold = 5000);
  void BeforeUnlock(const char* msg = nullptr);
   
  friend class CondVar;
  pthread_mutex_t mu_;
  pthread_t owner_;
#ifndef NDEBUG
  bool locked_;
#endif
  DISALLOW_COPY_AND_ASSIGN(Mutex);
};

class CondVar {
 public:
  explicit CondVar(Mutex* mu);
  ~CondVar();
  void Wait(const char* msg = nullptr);
  // Timed condition wait.  Returns true if timeout occurred.
  bool TimedWait(uint64_t abs_time_us, const char* msg = nullptr);
  // Time wait in timeout ms, return true if signalled
  bool IntervalWait(uint64_t timeout_interval, const char* msg = nullptr);
  void Signal();
  void SignalAll();
  void Broadcast();
 private:
  pthread_cond_t cv_;
  Mutex* mu_;
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
  DISALLOW_COPY_AND_ASSIGN(RWMutex);
};
  
} // namespace port

}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_MUTEX_H_