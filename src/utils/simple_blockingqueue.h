// Copyright (c) 2015 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// thread/mythread/include/mythread/blockingqueue.h

#ifndef BUBBLEFS_UTILS_SIMPLE_BLOCKINGQUEUE_H_
#define BUBBLEFS_UTILS_SIMPLE_BLOCKINGQUEUE_H_

#include <assert.h>
#include <queue>
#include "platform/mutexlock.h"

namespace bubblefs {
namespace mysimple {

template <typename T>
class BlockingQueue {
 public:
  BlockingQueue() : mutex_(), not_empty_(&mutex_) {}

  void push(const T& t) {
    MutexLock lock(&mutex_);
    if (queue_.empty()) {
      not_empty_.Signal();
    }
    queue_.push(t);
  }

  void pop() {
    MutexLock lock(&mutex_);
    while (queue_.empty()) {
      not_empty_.Wait();
    }
    assert(!queue_.empty());
    queue_.pop();
  }

  T take() {
    MutexLock lock(&mutex_);
    while (queue_.empty()) {
      not_empty_.Wait();
    }
    assert(!queue_.empty());
    T t(queue_.front());
    queue_.pop();
    return t;
  }

  bool empty() const {
    MutexLock lock(&mutex_);
    return queue_.empty();
  }

  size_t size() const {
    MutexLock lock(&mutex_);
    return queue_.size();
  }

 private:
  mutable port::Mutex mutex_;
  port::CondVar not_empty_;
  std::queue<T> queue_;

  // No copying allowed
  BlockingQueue(const BlockingQueue&);
  void operator=(const BlockingQueue&);
};

}  // namespace mysimple
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SIMPLE_BLOCKINGQUEUE_H_