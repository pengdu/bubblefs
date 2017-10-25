// Copyright (c) 2015 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// thread/mythread/include/mythread/singleton.h

#ifndef BUBBLEFS_UTILS_SIMPLE_SINGLETON_H_
#define BUBBLEFS_UTILS_SIMPLE_SINGLETON_H_

#include <pthread.h>

namespace bubblefs {
namespace simple {

template <typename T>
class Singleton {
 public:
  static T* Instance() {
    pthread_once(&once_, &Singleton<T>::Init);
    return instance_;
  }

  static void ShutDown() {
    delete instance_;
    instance_ = NULL;
  }

 private:
  static void Init() { instance_ = new T(); }

  static pthread_once_t once_;
  static T* instance_;
};

template <typename T>
pthread_once_t Singleton<T>::once_ = PTHREAD_ONCE_INIT;

template <typename T>
T* Singleton<T>::instance_ = NULL;

}  // namespace simple
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SIMPLE_SINGLETON_H_