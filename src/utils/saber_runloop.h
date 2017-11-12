// Copyright (c) 2017 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef BUBBLEFS_UTILS_SABER_RUNLOOP_H_
#define BUBBLEFS_UTILS_SABER_RUNLOOP_H_

#include <stdint.h>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include "platform/mutex.h"

namespace bubblefs {
namespace mysaber {

class Timer;
class TimerList;

typedef std::pair<uint64_t, Timer*> TimerId;
typedef std::function<void()> TimerProcCallback;

class RunLoop {
 public:
  typedef std::function<void()> Func;

  RunLoop();
  ~RunLoop();

  void Loop();
  void Exit();

  bool IsInMyLoop() const;
  void AssertInMyLoop();

  void RunInLoop(const Func& func);
  void RunInLoop(Func&& func);
  void QueueInLoop(const Func& func);
  void QueueInLoop(Func&& func);

  TimerId RunAt(uint64_t micros_value, const TimerProcCallback& cb);
  TimerId RunAfter(uint64_t micros_delay, const TimerProcCallback& cb);
  TimerId RunEvery(uint64_t micros_interval, const TimerProcCallback& cb);

  TimerId RunAt(uint64_t micros_value, TimerProcCallback&& cb);
  TimerId RunAfter(uint64_t micros_delay, TimerProcCallback&& cb);
  TimerId RunEvery(uint64_t micros_interval, TimerProcCallback&& cb);

  void Remove(TimerId t);

 private:
  bool exit_;
  const uint64_t tid_;

  port::Mutex mutex_;
  port::CondVar cond_;
  std::vector<Func> funcs_;
  std::unique_ptr<TimerList> timers_;

  // No copying allowed
  RunLoop(const RunLoop&);
  void operator=(const RunLoop&);
};

}  // namespace mysaber
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SABER_RUNLOOP_H_