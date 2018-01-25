// Copyright 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Use of this source code is governed by a BSD-style license
// that can be found in the License file.

// Author: Shuo Chen (chenshuo at chenshuo dot com)
//
// This is a public header file, it must only include public header files.

// muduo/muduo/net/TimerId.h

#ifndef BUBBLEFS_UTILS_MUDUO_NET_TIMERID_H_
#define BUBBLEFS_UTILS_MUDUO_NET_TIMERID_H_

namespace bubblefs {
namespace mymuduo {
namespace net {

class Timer;

///
/// An opaque identifier, for canceling Timer.
///
class TimerId
{
 public:
  TimerId()
    : timer_(NULL),
      sequence_(0)
  {
  }

  TimerId(Timer* timer, int64_t seq)
    : timer_(timer),
      sequence_(seq)
  {
  }

  // default copy-ctor, dtor and assignment are okay

  friend class TimerQueue;

 private:
  Timer* timer_;
  int64_t sequence_;
};

} // namespace net
} // namespace mymuduo
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_MUDUO_NET_TIMERID_H_