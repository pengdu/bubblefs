// Copyright 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Use of this source code is governed by a BSD-style license
// that can be found in the License file.

// Author: Shuo Chen (chenshuo at chenshuo dot com)
//
// This is an internal header file, you should not include this.

// muduo/muduo/net/EventLoopThreadPool.h

#ifndef BUBBLEFS_UTILS_MUDUO_NET_EVENTLOOP_THREADPOOL_H_
#define BUBBLEFS_UTILS_MUDUO_NET_EVENTLOOP_THREADPOOL_H_

#include "platform/muduo_types.h"

#include <functional>
#include <vector>
#include <boost/ptr_container/ptr_vector.hpp>

namespace bubblefs {
namespace mymuduo {
namespace net {

class EventLoop;
class EventLoopThread;

class EventLoopThreadPool
{
 public:
  typedef std::function<void(EventLoop*)> ThreadInitCallback;

  EventLoopThreadPool(EventLoop* baseLoop, const string& nameArg);
  ~EventLoopThreadPool();
  void setThreadNum(int numThreads) { numThreads_ = numThreads; }
  void start(const ThreadInitCallback& cb = ThreadInitCallback());

  // valid after calling start()
  /// round-robin
  EventLoop* getNextLoop();

  /// with the same hash code, it will always return the same EventLoop
  EventLoop* getLoopForHash(size_t hashCode);

  std::vector<EventLoop*> getAllLoops();

  bool started() const
  { return started_; }

  const string& name() const
  { return name_; }

 private:

  EventLoop* baseLoop_;
  string name_;
  bool started_;
  int numThreads_;
  int next_;
  boost::ptr_vector<EventLoopThread> threads_;
  std::vector<EventLoop*> loops_;
};

} // namespace net
} // namespace mymuduo
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_MUDUO_NET_EVENTLOOP_THREADPOOL_H_