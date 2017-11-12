// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/newtimer.h

#ifndef BUBBLEFS_PLATFORM_VOYAGER_NEWTIMER_H_
#define BUBBLEFS_PLATFORM_VOYAGER_NEWTIMER_H_

#include <stdint.h>
#include "utils/voyager_callback.h"
#include "utils/voyager_dispatch.h"

namespace bubblefs {
namespace myvoyager {

class NewTimer {
 public:
  NewTimer(EventLoop* ev, const TimerProcCallback& cb);
  NewTimer(EventLoop* ev, TimerProcCallback&& cb);
  ~NewTimer();

  EventLoop* OwnerEventLoop() const { return eventloop_; }
  bool IsRepeat() const { return repeat_; }

  // (1) nanos_value 表示定时在距离当前的时间多久。
  // (2) 当 nanos_interval 不为 0 时，表示为周期定时
  // (3) 当 nanos_value和nanos_interval 都为 0 时，表示不再定时
  void SetTime(uint64_t nanos_value, uint64_t nanos_interval);

 private:
  void SetTimeInLoop(uint64_t nanos_value, uint64_t nanos_interval);
  void HandleRead();

  const int timerfd_;
  bool repeat_;
  EventLoop* eventloop_;
  Dispatch dispatch_;

  TimerProcCallback timerproc_cb_;

  // No copying allowed
  NewTimer(const NewTimer&);
  void operator=(const NewTimer&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_VOYAGER_NEWTIMER_H_