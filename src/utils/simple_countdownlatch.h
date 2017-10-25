// Copyright (c) 2015 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// thread/mythread/include/mythread/countdownlatch.h

#ifndef BUBBLEFS_UTILS_SIMPLE_COUNTDOWNLATCH_H_
#define BUBBLEFS_UTILS_SIMPLE_COUNTDOWNLATCH_H_

#include "platform/mutexlock.h"

namespace bubblefs {
namespace simple {

class CountDownLatch {
 public:
  explicit CountDownLatch(int count)
      : mutex_(), cond_(&mutex_), count_(count) {}

  void Wait() {
    MutexLock lock(&mutex_);
    while (count_ > 0) {
      cond_.Wait();
    }
  }

  void CountDown() {
    MutexLock lock(&mutex_);
    --count_;
    if (count_ == 0) {
      cond_.Signal();
    }
  }

  int GetCount() const {
    MutexLock lock(&mutex_);
    return count_;
  }

 private:
  mutable port::Mutex mutex_;
  port::CondVar cond_;
  int count_;
};

}  // namespace simple
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SIMPLE_COUNTDOWNLATCH_H_