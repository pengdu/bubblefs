// Copyright (c) 2017 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// saber/saber/util/runloop_thread.cc

#include "utils/saber_runloop_thread.h"
#include <assert.h>
#include "platform/mutexlock.h"

namespace bubblefs {
namespace mysaber {

void* RunLoopThread::StartRunLoop(void* data) {
  RunLoopThread* thread = reinterpret_cast<RunLoopThread*>(data);
  thread->ThreadFunc();
  return nullptr;
}

RunLoopThread::RunLoopThread() : loop_(nullptr), mu_(), cond_(&mu_) {}

RunLoopThread::~RunLoopThread() {
  if (loop_ != nullptr) {
    loop_->Exit();
    thread_.Join();
  }
}

RunLoop* RunLoopThread::Loop() {
  assert(!thread_.Started());
  if (!thread_.Started()) {
    thread_.Start(&RunLoopThread::StartRunLoop, this);
    MutexLock lock(&mu_);
    while (loop_ == nullptr) {
      cond_.Wait();
    }
  }
  return loop_;
}

void RunLoopThread::ThreadFunc() {
  RunLoop loop;
  {
    MutexLock lock(&mu_);
    loop_ = &loop;
    cond_.Signal();
  }
  loop_->Loop();
  loop_ = nullptr;
}

}  // namespace mysaber
}  // namespace bubblefs