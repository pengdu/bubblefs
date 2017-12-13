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

// brpc/src/butil/thread_local.h

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

#include "platform/base_error.h"
#include "platform/macros.h"

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
  
}  // namespace internal
}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_THREADLOCAL_H_