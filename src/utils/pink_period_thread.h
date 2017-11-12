// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/include/period_thread.h

#ifndef BUBBLEFS_UTILS_PINK_PERIOD_THREAD_H_
#define BUBBLEFS_UTILS_PINK_PERIOD_THREAD_H_

#include <sys/time.h>
#include "utils/pink_thread.h"

namespace bubblefs {
namespace mypink {

class PeriodThread : public Thread {
 public:
  explicit PeriodThread(timeval period = timeval{1, 0});
  virtual void *ThreadMain();
  virtual void PeriodMain() = 0;

 private:
  struct timeval period_;
};

}  // namespace mypink
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PINK_PERIOD_THREAD_H_