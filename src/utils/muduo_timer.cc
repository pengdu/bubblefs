// Copyright 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Use of this source code is governed by a BSD-style license
// that can be found in the License file.

// Author: Shuo Chen (chenshuo at chenshuo dot com)

// muduo/muduo/net/Timer.cc

#include "utils/muduo_timer.h"

namespace bubblefs {
namespace mymuduo {
namespace net {
  
AtomicInt64 Timer::s_numCreated_;

void Timer::restart(Timestamp now)
{
  if (repeat_)
  {
    expiration_ = addTime(now, interval_);
  }
  else
  {
    expiration_ = Timestamp::invalid();
  }
}
 
} // namespace net
} // namespace mymuduo 
} // namespace bubblefs
