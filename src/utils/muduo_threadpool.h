// Use of this source code is governed by a BSD-style license
// that can be found in the License file.
//
// Author: Shuo Chen (chenshuo at chenshuo dot com)

// muduo/muduo/base/ThreadPool.h

#ifndef BUBBLEFS_UTILS_MUDUO_THREADPOOL_H_
#define BUBBLEFS_UTILS_MUDUO_THREADPOOL_H_

#include <deque>
#include <functional>
#include "platform/macros.h"
#include "platform/muduo_mutex.h"
#include "platform/muduo_types.h"
#include "utils/muduo_thread.h"

#include "boost/ptr_container/ptr_vector.hpp"

namespace bubblefs {
namespace mymuduo {

class ThreadPool
{
 public:
  typedef std::function<void ()> Task;

  explicit ThreadPool(const string& nameArg = string("ThreadPool"));
  ~ThreadPool();

  // Must be called before start().
  void setMaxQueueSize(int maxSize) { maxQueueSize_ = maxSize; }
  void setThreadInitCallback(const Task& cb)
  { threadInitCallback_ = cb; }

  void start(int numThreads);
  void stop();

  const string& name() const
  { return name_; }

  size_t queueSize() const;

  // Could block if maxQueueSize > 0
  void run(const Task& f);
#ifdef __GXX_EXPERIMENTAL_CXX0X__
  void run(Task&& f);
#endif

 private:
  bool isFull() const;
  void runInThread();
  Task take();

  mutable MutexLock mutex_;
  Condition notEmpty_;
  Condition notFull_;
  string name_;
  Task threadInitCallback_;
  boost::ptr_vector<mymuduo::Thread> threads_;
  std::deque<Task> queue_;
  size_t maxQueueSize_;
  bool running_;
};

} // namespace mymuduo
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MUDUO_THREADPOOL_H_