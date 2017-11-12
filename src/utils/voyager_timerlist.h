// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/timerlist.h

#ifndef BUBBLEFS_UTILS_VOYAGER_TIMERLIST_H_
#define BUBBLEFS_UTILS_VOYAGER_TIMERLIST_H_

#include <set>
#include <utility>
#include <vector>
#include "utils/voyager_callback.h"
#include "utils/voyager_eventloop.h"

namespace bubblefs {
namespace myvoyager {

class TimerList {
 public:
  explicit TimerList(EventLoop* ev);
  ~TimerList();

  TimerId Insert(uint64_t micros_value, uint64_t micros_interval,
                 const TimerProcCallback& cb);
  TimerId Insert(uint64_t micros_value, uint64_t micros_interval,
                 TimerProcCallback&& cb);
  void Erase(TimerId timer);

  uint64_t TimeoutMicros() const;
  void RunTimerProcs();

 private:
  void InsertInLoop(TimerId timer);
  void EraseInLoop(TimerId timer);

  uint64_t last_time_out_;

  EventLoop* eventloop_;

  std::set<Timer*> timer_ptrs_;
  std::set<TimerId> timers_;

  // No copying allowed
  TimerList(const TimerList&);
  void operator=(const TimerList&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_TIMERLIST_H_