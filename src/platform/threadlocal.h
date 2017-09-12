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
//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// Paddle/paddle/utils/ThreadLocal.h
// rocksdb/util/thread_local.h

#ifndef BUBBLEFS_PLATFORM_THREADLOCAL_H_
#define BUBBLEFS_PLATFORM_THREADLOCAL_H_

#include <sys/syscall.h>
#include <sys/types.h>
#include <pthread.h>
#include <unistd.h>
#include <atomic>
#include <functional>
#include <memory>
#include <map>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>
#include "platform/base.h"
#include "platform/logging.h"
#include "platform/port.h"
#include "utils/autovector.h"

namespace bubblefs {
  
// Try to come up with a portable implementation of thread local variables
#ifdef TF_SUPPORT_THREAD_LOCAL
#define TF_STATIC_THREAD_LOCAL static __thread
#define TF_THREAD_LOCAL __thread
#else
#define TF_STATIC_THREAD_LOCAL static thread_local
#define TF_THREAD_LOCAL thread_local
#endif
  
namespace concurrent {
  
/**
 * Thread local storage for object.
 * Example:
 *
 * Declarartion:
 * ThreadLocal<vector<int>> vec_;
 *
 * Use in thread:
 * vector<int>& vec = *vec; // obtain the thread specific object
 * vec.resize(100);
 *
 * Note that this ThreadLocal will desconstruct all internal data when thread
 * exits
 * This class is suitable for cases when frequently creating and deleting
 * threads.
 *
 * Consider implementing a new ThreadLocal if one needs to frequently create
 * both instances and threads.
 *
 * see also ThreadLocalD
 */
template <class T>
class ThreadLocal {
public:
  ThreadLocal() {
    CHECK(pthread_key_create(&threadSpecificKey_, dataDestructor) == 0);
  }
  ~ThreadLocal() { pthread_key_delete(threadSpecificKey_); }

  /**
   * @brief get thread local object.
   * @param if createLocal is true and thread local object is never created,
   * return a new object. Otherwise, return nullptr.
   */
  T* get(bool createLocal = true) {
    T* p = (T*)pthread_getspecific(threadSpecificKey_);
    if (!p && createLocal) {
      p = new T();
      int ret = pthread_setspecific(threadSpecificKey_, p);
      CHECK(ret == 0);
    }
    return p;
  }

  /**
   * @brief set (overwrite) thread local object. If there is a thread local
   * object before, the previous object will be destructed before.
   *
   */
  void set(T* p) {
    if (T* q = get(false)) {
      dataDestructor(q);
    }
    CHECK(pthread_setspecific(threadSpecificKey_, p) == 0);
  }

  /**
   * return reference.
   */
  T& operator*() { return *get(); }

  /**
   * Implicit conversion to T*
   */
  operator T*() { return get(); }

private:
  static void dataDestructor(void* p) { delete (T*)p; }

  pthread_key_t threadSpecificKey_;
};

/**
 * Almost the same as ThreadLocal, but note that this ThreadLocalD will
 * destruct all internal data when ThreadLocalD instance destructs.
 *
 * This class is suitable for cases when frequently creating and deleting
 * objects.
 *
 * see also ThreadLocal
 *
 * @note The type T must implemented default constructor.
 */
template <class T>
class ThreadLocalD {
public:
  ThreadLocalD() { CHECK(pthread_key_create(&threadSpecificKey_, NULL) == 0); }
  ~ThreadLocalD() {
    pthread_key_delete(threadSpecificKey_);
    for (auto t : threadMap_) {
      dataDestructor(t.second);
    }
  }

  /**
   * @brief Get thread local object. If not exists, create new one.
   */
  T* get() {
    T* p = (T*)pthread_getspecific(threadSpecificKey_);
    if (!p) {
      p = new T();
      CHECK(pthread_setspecific(threadSpecificKey_, p) == 0);
      updateMap(p);
    }
    return p;
  }

  /**
   * @brief Set thread local object. If there is an object create before, the
   * old object will be destructed.
   */
  void set(T* p) {
    if (T* q = (T*)pthread_getspecific(threadSpecificKey_)) {
      dataDestructor(q);
    }
    CHECK(pthread_setspecific(threadSpecificKey_, p) == 0);
    updateMap(p);
  }

  /**
   * @brief Get reference of the thread local object.
   */
  T& operator*() { return *get(); }

private:
  static void dataDestructor(void* p) { delete (T*)p; }

  void updateMap(T* p) {
    pid_t tid = gettid();
    CHECK_NE(tid, -1);
    std::lock_guard<std::mutex> guard(mutex_);
    auto ret = threadMap_.insert(std::make_pair(tid, p));
    if (!ret.second) {
      ret.first->second = p;
    }
  }

  pthread_key_t threadSpecificKey_;
  std::mutex mutex_;
  std::map<pid_t, T*> threadMap_;
};

/**
 * @brief Thread-safe C-style random API.
 */
class ThreadLocalRand {
public:
  /**
   * initSeed just like srand,
   * called by main thread,
   * init defaultSeed for all thread
   */
  static void initSeed(unsigned int seed) { defaultSeed_ = seed; }

  /**
   * initThreadSeed called by each thread,
   * init seed to defaultSeed + *tid*
   * It should be called after main initSeed and before using rand()
   * It's optional, getSeed will init seed if it's not initialized.
   */
  static void initThreadSeed(int tid) {
    seed_.set(new unsigned int(defaultSeed_ + tid));
  }

  /// thread get seed, then can call rand_r many times.
  /// Caller thread can modify the seed value if it's necessary.
  ///
  /// if flag thread_local_rand_use_global_seed set,
  /// the seed will be set to defaultSeed in thread's first call.
  static unsigned int* getSeed();

  /// like ::rand
  static int rand() { return rand_r(getSeed()); }

  /**
   * Get defaultSeed for all thread.
   */
  static int getDefaultSeed() { return defaultSeed_; }

protected:
  static unsigned int defaultSeed_;
  static ThreadLocal<unsigned int> seed_;
};

/**
 * @brief Thread-safe C++ style random engine.
 */
class ThreadLocalRandomEngine {
public:
  /**
   * get random_engine for each thread.
   *
   * Engine's seed will be initialized by ThreadLocalRand.
   */
  static std::default_random_engine& get();

protected:
  static ThreadLocal<std::default_random_engine> engine_;
};

}  // namespace concurrent

// Cleanup function that will be called for a stored thread local
// pointer (if not NULL) when one of the following happens:
// (1) a thread terminates
// (2) a ThreadLocalPtr is destroyed
typedef void (*UnrefHandler)(void* ptr);

// ThreadLocalPtr stores only values of pointer type.  Different from
// the usual thread-local-storage, ThreadLocalPtr has the ability to
// distinguish data coming from different threads and different
// ThreadLocalPtr instances.  For example, if a regular thread_local
// variable A is declared in DBImpl, two DBImpl objects would share
// the same A.  However, a ThreadLocalPtr that is defined under the
// scope of DBImpl can avoid such confliction.  As a result, its memory
// usage would be O(# of threads * # of ThreadLocalPtr instances).
class ThreadLocalPtr {
 public:
  explicit ThreadLocalPtr(UnrefHandler handler = nullptr);

  ~ThreadLocalPtr();

  // Return the current pointer stored in thread local
  void* Get() const;

  // Set a new pointer value to the thread local storage.
  void Reset(void* ptr);

  // Atomically swap the supplied ptr and return the previous value
  void* Swap(void* ptr);

  // Atomically compare the stored value with expected. Set the new
  // pointer value to thread local only if the comparison is true.
  // Otherwise, expected returns the stored value.
  // Return true on success, false on failure
  bool CompareAndSwap(void* ptr, void*& expected);

  // Reset all thread local data to replacement, and return non-nullptr
  // data for all existing threads
  void Scrape(core::autovector<void*>* ptrs, void* const replacement);

  typedef std::function<void(void*, void*)> FoldFunc;
  // Update res by applying func on each thread-local value. Holds a lock that
  // prevents unref handler from running during this call, but clients must
  // still provide external synchronization since the owning thread can
  // access the values without internal locking, e.g., via Get() and Reset().
  void Fold(FoldFunc func, void* res);

  // Add here for testing
  // Return the next available Id without claiming it
  static uint32_t TEST_PeekId();

  // Initialize the static singletons of the ThreadLocalPtr.
  //
  // If this function is not called, then the singletons will be
  // automatically initialized when they are used.
  //
  // Calling this function twice or after the singletons have been
  // initialized will be no-op.
  static void InitSingletons();

  class StaticMeta;

private:

  static StaticMeta* Instance();

  const uint32_t id_;
};

}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_THREADLOCAL_H_