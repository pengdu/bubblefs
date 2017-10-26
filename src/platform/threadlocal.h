// Copyright (c) 2014 Baidu, Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Ge,Jun (gejun@baidu.com)
// Date: Tue Sep 16 12:39:12 CST 2014

// protobuf/src/google/protobuf/stubs/mutex.h
// brpc/src/butil/thread_local.h
// dmlc-core/include/dmlc/thread_local.h

#ifndef BUBBLEFS_PLATFORM_THREADLOCAL_H_
#define BUBBLEFS_PLATFORM_THREADLOCAL_H_

#include <sys/syscall.h>
#include <sys/types.h>
#include <stddef.h>
#include <pthread.h>
#include <unistd.h>
#include <atomic>
#include <functional>
#include <memory>
#include <map>
#include <mutex>
#include <new>
#include <random>
#include <unordered_map>
#include <vector>
#include "platform/logging.h"
#include "platform/macros.h"
#include "platform/port.h"

// Try to come up with a portable implementation of thread local variables
// Provide thread_local keyword (for primitive types) before C++11
// DEPRECATED: define this keyword before C++11 might make the variable ABI
// incompatible between C++11 and C++03
#if !defined(thread_local) &&                                           \
    (__cplusplus < 201103L ||                                           \
     (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) < 40800)
// GCC supports thread_local keyword of C++11 since 4.8.0
#ifdef _MSC_VER
// WARNING: don't use this macro in C++03
#define thread_local __declspec(thread)
#else
// WARNING: don't use this macro in C++03
#define thread_local __thread
#endif  // _MSC_VER
#endif  // thread_local

#ifdef TF_SUPPORT_THREAD_LOCAL
#define STATIC_THREAD_LOCAL static __thread
#define THREAD_LOCAL __thread
#else
#define STATIC_THREAD_LOCAL static thread_local
#define THREAD_LOCAL thread_local
#endif

namespace bubblefs { 
  
namespace internal {
  
template<typename T>
class ThreadLocalStorage {
 public:
  ThreadLocalStorage() {
    pthread_key_create(&key_, &ThreadLocalStorage::Delete);
  }
  ~ThreadLocalStorage() {
    pthread_key_delete(key_);
  }
  T* Get() {
    T* result = static_cast<T*>(pthread_getspecific(key_));
    if (result == NULL) {
      result = new T();
      pthread_setspecific(key_, result);
    }
    return result;
  }
 private:
  static void Delete(void* value) {
    delete static_cast<T*>(value);
  }
  pthread_key_t key_;

  DISALLOW_COPY_AND_ASSIGN(ThreadLocalStorage);
};  
  
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
    pid_t tid = syscall(SYS_gettid);
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

}  // namespace internal

namespace base {

// Get a thread-local object typed T. The object will be default-constructed
// at the first call to this function, and will be deleted when thread
// exits.
template <typename T> inline T* get_thread_local();

// |fn| or |fn(arg)| will be called at caller's exit. If caller is not a 
// thread, fn will be called at program termination. Calling sequence is LIFO:
// last registered function will be called first. Duplication of functions 
// are not checked. This function is often used for releasing thread-local
// resources declared with __thread (or thread_local defined in 
// butil/thread_local.h) which is much faster than pthread_getspecific or
// boost::thread_specific_ptr.
// Returns 0 on success, -1 otherwise and errno is set.
int thread_atexit(void (*fn)());
int thread_atexit(void (*fn)(void*), void* arg);

// Remove registered function, matched functions will not be called.
void thread_atexit_cancel(void (*fn)());
void thread_atexit_cancel(void (*fn)(void*), void* arg);

// Delete the typed-T object whose address is `arg'. This is a common function
// to thread_atexit.
template <typename T> void delete_object(void* arg) {
    delete static_cast<T*>(arg);
}

namespace detail {

template <typename T>
class ThreadLocalHelper {
public:
    inline static T* get() {
        if (__builtin_expect(value != nullptr, 1)) {
            return value;
        }
        value = new (std::nothrow) T;
        if (value != nullptr) {
            base::thread_atexit(delete_object<T>, value);
        }
        return value;
    }
    static THREAD_LOCAL T* value;
};

template <typename T> THREAD_LOCAL T* ThreadLocalHelper<T>::value = nullptr;

}  // namespace detail

template <typename T> inline T* get_thread_local() {
    return detail::ThreadLocalHelper<T>::get();
}

/*!
 * \brief A threadlocal store to store threadlocal variables.
 *  Will return a thread local singleton of type T
 * \tparam T the type we like to store
 */
template<typename T>
class ThreadLocalStore {
 public:
  /*! \return get a thread local singleton */
  static T* Get() {
#if TF_SUPPORT_THREAD_LOCAL
    static __thread T* ptr = nullptr;
    if (ptr == nullptr) {
      ptr = new T();
      Singleton()->RegisterDelete(ptr);
    }
    return ptr;
#else
    static thread_local T inst;
    return &inst;
#endif
  }

 private:
  /*! \brief constructor */
  ThreadLocalStore() {}
  /*! \brief destructor */
  ~ThreadLocalStore() {
    for (size_t i = 0; i < data_.size(); ++i) {
      delete data_[i];
    }
  }
  /*! \return singleton of the store */
  static ThreadLocalStore<T> *Singleton() {
    static ThreadLocalStore<T> inst;
    return &inst;
  }
  /*!
   * \brief register str for internal deletion
   * \param str the string pointer
   */
  void RegisterDelete(T *str) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_.push_back(str);
    lock.unlock();
  }
  /*! \brief internal mutex */
  std::mutex mutex_;
  /*!\brief internal data */
  std::vector<T*> data_;
};

}  // namespace base

}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_THREADLOCAL_H_