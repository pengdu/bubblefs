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
// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.
//

// tensorflow/tensorflow/core/platform/mutex.h
// tensorflow/tensorflow/core/platform/default/mutex.h
// slash/slash/include/slash_mutex.h
// slash/slash/include/cond_lock.h

#ifndef BUBBLEFS_PLATFORM_MUTEX_H_
#define BUBBLEFS_PLATFORM_MUTEX_H_

#include <pthread.h>
#include <unistd.h>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
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

typedef pthread_once_t OnceType;
extern void InitOnce(OnceType* once, void (*initializer)());  
  
class CondVar;

//#define MUTEX_DEBUG

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
  // use when MUTEX_DEBUG is defined
  void AssertHeld();
  pthread_mutex_t* GetMutex() {
    return &mu_;
  }

 private:
  void AfterLock(const char* msg = nullptr, int64_t msg_threshold = 5000);
  void BeforeUnlock(const char* msg = nullptr);
   
  friend class CondVar;
  pthread_mutex_t mu_;
  
#ifdef MUTEX_DEBUG
  pthread_t owner_;
  const char* msg_;
  int64_t msg_threshold_;
  int64_t lock_time_;
#endif
  DISALLOW_COPY_AND_ASSIGN(Mutex);
};

class CondVar {
 public:
  explicit CondVar(Mutex* mu);
  ~CondVar();
  void Wait();
  // Timed condition wait.  Returns true if timeout occurred.
  bool TimedWaitAbsolute(uint64_t abs_time_us);
  bool TimedWaitAbsolute(const struct timespec& absolute_time);
  // Time wait in timeout ms, return true if timeout
  bool TimedWait(uint64_t timeout);
  bool TimedwaitRelative(const struct timespec& relative_time);
  void Signal();
  void SignalAll();
  void Broadcast();
 private:
  pthread_cond_t cv_;
  Mutex* mu_;
  
  DISALLOW_COPY_AND_ASSIGN(CondVar);
};

class ConditionVariable
{
public:
    ConditionVariable();
    ~ConditionVariable();

    void Signal();
    void Broadcast();
    void Wait(Mutex* mutex);
    // If timeout_in_ms < 0, it means infinite waiting until condition is signaled
    // by another thread
    int TimedWait(Mutex* mutex, int timeout_in_ms = -1);
private:
    void CheckValid() const;
private:
    pthread_cond_t m_hCondition;
};

class RWMutex {
 public:
  RWMutex();
  ~RWMutex();

  void ReadLock();
  bool TryReadLock();
  void WriteLock();
  bool TryWriteLock();
  void Unlock();
  void ReadUnlock();
  void WriteUnlock();
  void AssertHeld() { }

 private:
  pthread_rwlock_t mu_; // the underlying platform mutex
  DISALLOW_COPY_AND_ASSIGN(RWMutex);
};

class RefMutex {
 public:
  RefMutex();
  ~RefMutex();

  // Lock and Unlock will increase and decrease refs_,
  // should check refs before Unlock
  void Lock();
  void Unlock();

  void Ref();
  void Unref();
  bool IsLastRef() {
    return refs_ == 1;
  }

 private:
  pthread_mutex_t mu_;
  int refs_;
  DISALLOW_COPY_AND_ASSIGN(RefMutex);
};

class RecordMutex {
public:
  RecordMutex() {};
  ~RecordMutex();

  void Lock(const std::string &key);
  void Unlock(const std::string &key);

private:

  Mutex mutex_;
  std::unordered_map<std::string, RefMutex *> records_;
  
  DISALLOW_COPY_AND_ASSIGN(RecordMutex);
};

class RecordLock {
 public:
  RecordLock(RecordMutex *mu, const std::string &key)
      : mu_(mu), key_(key) {
        mu_->Lock(key_);
      }
  ~RecordLock() { mu_->Unlock(key_); }

 private:
  RecordMutex *const mu_;
  std::string key_;

  DISALLOW_COPY_AND_ASSIGN(RecordLock);
};

/*
 * CondLock is a wrapper for condition variable.
 * It contain a mutex in it's class, so we don't need other to protect the 
 * condition variable.
 */
class CondLock {
 public:
  CondLock();
  ~CondLock();

  void Lock();
  void Unlock();

  void Wait();
  
  /*
   * timeout is millisecond
   */
  void TimedWait(uint64_t timeout);
  void Signal();
  void Broadcast();

 private:
  pthread_mutex_t mutex_;
  pthread_cond_t cond_;

  DISALLOW_COPY_AND_ASSIGN(CondLock);
};
  
} // namespace port

}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_MUTEX_H_