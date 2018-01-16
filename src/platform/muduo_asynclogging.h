
// // muduo/muduo/base/AsyncLogging.h

#ifndef BUBBLEFS_PLATFORM_MUDUO_ASYNCLOGGING_H_
#define BUBBLEFS_PLATFORM_MUDUO_ASYNCLOGGING_H_

#include <functional>
#include "platform/macros.h"
#include "platform/muduo_countdown_latch.h"
#include "platform/muduo_mutex.h"
#include "platform/muduo_logstream.h"
#include "utils/muduo_blocking_queue.h"
#include "utils/muduo_bounded_blocking_queue.h"
#include "utils/muduo_thread.h"

#include "boost/ptr_container/ptr_vector.hpp"

namespace bubblefs {
namespace mymuduo {

class AsyncLogging
{
 public:

  AsyncLogging(const string& basename,
               off_t rollSize,
               int flushInterval = 3);

  ~AsyncLogging()
  {
    if (running_)
    {
      stop();
    }
  }

  void append(const char* logline, int len);

  void start()
  {
    running_ = true;
    thread_.start();
    latch_.wait();
  }

  void stop()
  {
    running_ = false;
    cond_.notify();
    thread_.join();
  }

 private:

  void threadFunc();

  typedef mymuduo::detail::FixedBuffer<mymuduo::detail::kLargeBuffer> Buffer;
  typedef boost::ptr_vector<Buffer> BufferVector;
  typedef BufferVector::auto_type BufferPtr;

  const int flushInterval_;
  bool running_;
  string basename_;
  off_t rollSize_;
  mymuduo::Thread thread_;
  mymuduo::CountDownLatch latch_;
  mymuduo::MutexLock mutex_;
  mymuduo::Condition cond_;
  BufferPtr currentBuffer_;
  BufferPtr nextBuffer_;
  BufferVector buffers_;
  
  DISALLOW_COPY_AND_ASSIGN(AsyncLogging);
};

} // namespace mymuduo
} // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_MUDUO_ASYNCLOGGING_H_