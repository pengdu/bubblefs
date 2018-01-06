// Use of this source code is governed by a BSD-style license
// that can be found in the License file.
//
// Author: Shuo Chen (chenshuo at chenshuo dot com)

// muduo/muduo/base/CountDownLatch.h

#ifndef BUBBLEFS_PLATFORM_MUDUO_COUNTDOWN_LATCH_H_
#define BUBBLEFS_PLATFORM_MUDUO_COUNTDOWN_LATCH_H_

#include "platform/muduo_mutex.h"

namespace bubblefs {
namespace mymuduo {

class CountDownLatch
{
 public:

  explicit CountDownLatch(int count);

  void wait();

  void countDown();

  int getCount() const;

 private:
  mutable MutexLock mutex_;
  Condition condition_;
  int count_;
  
  DISALLOW_COPY_AND_ASSIGN(CountDownLatch);
};

} // namespace mymuduo
} // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_MUDUO_COUNTDOWN_LATCH_H_