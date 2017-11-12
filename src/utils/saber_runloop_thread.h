// Copyright (c) 2017 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// saber/saber/util/runloop_thread.h

#ifndef BUBBLEFS_UTILS_SABER_RUNLOOP_THREAD_H_
#define BUBBLEFS_UTILS_SABER_RUNLOOP_THREAD_H_

#include "platform/mutex.h"
#include "utils/saber_runloop.h"
#include "utils/saber_thread.h"

namespace bubblefs {
namespace mysaber {

class RunLoopThread {
 public:
  RunLoopThread();
  ~RunLoopThread();

  RunLoop* Loop();

 private:
  static void* StartRunLoop(void* data);

  void ThreadFunc();

  RunLoop* loop_;
  port::Mutex mu_;
  port::CondVar cond_;
  Thread thread_;

  // No copying allowed
  RunLoopThread(const RunLoopThread&);
  void operator=(const RunLoopThread&);
};

}  // namespace mysaber
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SABER_RUNLOOP_THREAD_H_