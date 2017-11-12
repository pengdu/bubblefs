// Copyright (c) 2017 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// saber/saber/util/thread.h

#ifndef BUBBLEFS_UTILS_SABER_THREAD_H_
#define BUBBLEFS_UTILS_SABER_THREAD_H_

#include <pthread.h>
#include <stdint.h>

namespace bubblefs {
namespace mysaber {

namespace CurrentThread {

extern __thread uint64_t cached_tid;

extern uint64_t Tid();

}  // namespace CurrentThread

class Thread {
 public:
  Thread();
  ~Thread();

  void Start(void* (*function)(void*), void* arg);
  void Join();

  bool Started() const { return started_; }
  pthread_t gettid() const { return thread_; }

 private:
  void PthreadCall(const char* label, int result);

  bool started_;
  bool joined_;
  pthread_t thread_;

  // No copying allowed
  Thread(const Thread&);
  void operator=(const Thread&);
};

}  // namespace mysaber
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SABER_THREAD_H_