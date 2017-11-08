// Copyright (c) 2017 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// saber/saber/util/timerlist.h

#ifndef BUBBLEFS_UTILS_SABER_TIMERLIST_H_
#define BUBBLEFS_UTILS_SABER_TIMERLIST_H_

#include <stdint.h>
#include <functional>
#include <set>
#include <utility>
#include "utils/saber_runloop.h"

namespace bubblefs {
namespace saber {

class TimerList {
 public:
  explicit TimerList(RunLoop* loop);
  ~TimerList();

  TimerId RunAt(uint64_t micros_value, const TimerProcCallback& cb);
  TimerId RunAt(uint64_t micros_value, TimerProcCallback&& cb);

  TimerId RunAfter(uint64_t micros_delay, const TimerProcCallback& cb);
  TimerId RunAfter(uint64_t micros_delay, TimerProcCallback&& cb);

  TimerId RunEvery(uint64_t micros_interval, const TimerProcCallback& cb);
  TimerId RunEvery(uint64_t micros_interval, TimerProcCallback&& cb);

  void Remove(TimerId timer);

  uint64_t TimeoutMicros() const;
  void RunTimerProcs();

 private:
  void InsertInLoop(TimerId timer);

  uint64_t last_time_out_;

  RunLoop* loop_;
  std::set<Timer*> timer_ptrs_;
  std::set<TimerId> timers_;

  // No copying allowed
  TimerList(const TimerList&);
  void operator=(const TimerList&);
};

}  // namespace saber
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SABER_TIMERLIST_H_