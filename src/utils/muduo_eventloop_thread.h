// Copyright 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Use of this source code is governed by a BSD-style license
// that can be found in the License file.

// Author: Shuo Chen (chenshuo at chenshuo dot com)
//
// This is a public header file, it must only include public header files.

// muduo/muduo/net/EventLoopThread.h

#ifndef BUBBLEFS_UTILS_MUDUO_NET_EVENTLOOP_THREAD_H_
#define BUBBLEFS_UTILS_MUDUO_NET_EVENTLOOP_THREAD_H_

#include <functional>
#include "platform/muduo_mutex.h"
#include "utils/muduo_thread.h"

namespace bubblefs {
namespace mymuduo {
namespace net {

class EventLoop;

class EventLoopThread
{
 public:
  typedef std::function<void(EventLoop*)> ThreadInitCallback;

  EventLoopThread(const ThreadInitCallback& cb = ThreadInitCallback(),
                  const string& name = string());
  ~EventLoopThread();
  EventLoop* startLoop();

 private:
  void threadFunc();

  EventLoop* loop_;
  bool exiting_;
  Thread thread_;
  MutexLock mutex_;
  Condition cond_;
  ThreadInitCallback callback_;
};

} // namespace net
} // namespace mymuduo
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_MUDUO_NET_EVENTLOOP_THREAD_H_