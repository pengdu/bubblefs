// Copyright (c) 2015 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// thread/mythread/include/mythread/thread.h

#ifndef BUBBLEFS_UTILS_SIMPLE_THREAD_H_
#define BUBBLEFS_UTILS_SIMPLE_THREAD_H_

#include <pthread.h>

namespace bubblefs {
namespace mysimple {

class Thread {
 public:
  Thread(void (*function)(void*), void* arg);
  ~Thread();

  void Start();
  void Join();

  bool Started() const { return started_; }
  pthread_t gettid() const { return thread_; }

 private:
  void PthreadCall(const char* label, int result);

  bool started_;
  bool joined_;
  pthread_t thread_;
  void (*function_)(void*);
  void* arg_;

  // No copying allowed
  Thread(const Thread&);
  void operator=(const Thread&);
};

}  // namespace mysimple
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SIMPLE_THREAD_H_