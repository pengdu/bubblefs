// Copyright (c) 2015 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef BUBBLEFS_UTILS_SIMPLE_THREADLOCAL_STORAGE_H_
#define BUBBLEFS_UTILS_SIMPLE_THREADLOCAL_STORAGE_H_

#include <pthread.h>

namespace bubblefs {
namespace simple {

template <typename T>
class ThreadLocalStorage {
 public:
  ThreadLocalStorage() {
    pthread_key_create(&key_, &ThreadLocalStorage::Deleter);
  }
  ~ThreadLocalStorage() { pthread_key_delete(key_); }

  T* Get() {
    T* result = reinterpret_cast<T*>(pthread_getspecific(key_));
    if (result == NULL) {
      result = new T();
      pthread_setspecific(key_, result);
    }
    return result;
  }

 private:
  static void Deleter(void* value) { delete static_cast<T*>(value); }
  pthread_key_t key_;

  // No copying allowed
  ThreadLocalStorage(const ThreadLocalStorage&);
  void operator=(const ThreadLocalStorage&);
};

}  // namespace simple
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SIMPLE_THREADLOCAL_STORAGE_H_