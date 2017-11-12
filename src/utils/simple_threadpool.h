// Copyright (c) 2015 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// thread/mythread/include/mythread/threadpool.h

#ifndef BUBBLEFS_UTILS_SIMPLE_THREADPOOL_H_
#define BUBBLEFS_UTILS_SIMPLE_THREADPOOL_H_

#include <queue>
#include <string>
#include <vector>
#include "platform/mutexlock.h"

namespace bubblefs {
namespace mysimple {

class Thread;

class ThreadPool {
 public:
  explicit ThreadPool(int size);
  ~ThreadPool();

  void Start();
  void Stop();

  int size() const { return size_; }

  void Put(void (*function)(void*), void* arg);
  size_t QueueSize() const;

 private:
  struct RunItem {
    void (*function)(void*);
    void* arg;
    RunItem() : function(NULL), arg(NULL) {}
    RunItem(void (*f)(void*), void* a) : function(f), arg(a) {}
  };

  static void ThreadFunc(void* obj);
  RunItem Take();

  mutable port::Mutex mutex_;
  port::CondVar cond_;
  bool started_;
  int size_;
  std::vector<Thread*> threads_;

  typedef std::queue<RunItem> RunQueue;
  RunQueue queue_;

  // No copying allowed
  ThreadPool(const ThreadPool&);
  void operator=(const ThreadPool&);
};

}  // namespace mysimple
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SIMPLE_THREADPOOL_H_