// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/include/bg_thread.h

#ifndef BUBBLEFS_UTILS_PINK_BG_THREAD_H_
#define BUBBLEFS_UTILS_PINK_BG_THREAD_H_

#include <atomic>
#include <queue>
#include "platform/mutexlock.h"
#include "utils/pink_thread.h"

namespace bubblefs {
namespace mypink {

struct TimerItem {
  uint64_t exec_time;
  void (*function)(void *);
  void* arg;
  TimerItem(uint64_t _exec_time, void (*_function)(void*), void* _arg) :
    exec_time(_exec_time),
    function(_function),
    arg(_arg) {}
  bool operator < (const TimerItem& item) const {
    return exec_time > item.exec_time;
  }
};

class BGThread : public Thread {
 public:
  explicit BGThread(int full = 100000) :
    Thread::Thread(),
    full_(full),
    mu_(),
    rsignal_(&mu_),
    wsignal_(&mu_) {
    }

  virtual ~BGThread() {
    StopThread();
  }

  virtual int StopThread() override {
    should_stop_ = true;
    rsignal_.Signal();
    wsignal_.Signal();
    return Thread::StopThread();
  }

  void Schedule(void (*function)(void*), void* arg);

  /*
   * timeout is in millionsecond
   */
  void DelaySchedule(uint64_t timeout, void (*function)(void *), void* arg);

  void QueueSize(int* pri_size, int* qu_size);
  void QueueClear();

 private:

  struct BGItem {
    void (*function)(void*);
    void* arg;
    BGItem(void (*_function)(void*), void* _arg)
      : function(_function), arg(_arg) {}
  };


  std::queue<BGItem> queue_;
  std::priority_queue<TimerItem> timer_queue_;

  size_t full_;
  port::Mutex mu_;
  port::CondVar rsignal_;
  port::CondVar wsignal_;
  virtual void *ThreadMain() override;
};

}  // namespace mypink
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PINK_BG_THREAD_H_