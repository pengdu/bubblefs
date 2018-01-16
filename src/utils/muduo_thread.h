// Use of this source code is governed by a BSD-style license
// that can be found in the License file.
//
// Author: Shuo Chen (chenshuo at chenshuo dot com)

// muduo/muduo/base/Thread.h

#ifndef MUDUO_BASE_THREAD_H
#define MUDUO_BASE_THREAD_H

#include <pthread.h>
#include <functional>
#include <memory>
#include <boost/iterator/iterator_concepts.hpp>
#include "platform/macros.h"
#include "platform/muduo_atomic.h"
#include "platform/muduo_countdown_latch.h"
#include "platform/muduo_types.h"

namespace bubblefs {
namespace mymuduo {

class Thread
{
 public:
  typedef std::function<void ()> ThreadFunc;

  explicit Thread(const ThreadFunc&, const string& name = string());
#ifdef __GXX_EXPERIMENTAL_CXX0X__
  explicit Thread(ThreadFunc&&, const string& name = string());
#endif
  ~Thread();

  void start();
  int join(); // return pthread_join()

  bool started() const { return started_; }
  // pthread_t pthreadId() const { return pthreadId_; }
  pid_t tid() const { return tid_; }
  const string& name() const { return name_; }

  static int numCreated() { return numCreated_.get(); }

 private:
  void setDefaultName();

  bool       started_;
  bool       joined_;
  pthread_t  pthreadId_;
  pid_t      tid_;
  ThreadFunc func_;
  string     name_;
  CountDownLatch latch_;

  static AtomicInt32 numCreated_;
  
  DISALLOW_COPY_AND_ASSIGN(Thread);
};

} // namespace mymuduo
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MUDUO_THREAD_H_