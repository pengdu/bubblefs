// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/src/period_thread.cc

#include "utils/pink_period_thread.h"
#include <unistd.h>

namespace bubblefs {
namespace mypink {

PeriodThread::PeriodThread(timeval period) :
  period_(period) {
}

void *PeriodThread::ThreadMain() {
  PeriodMain();
  select(0, NULL, NULL, NULL, &period_);
  return NULL;
}

}  // namespace mypink
}  // namespace bubblefs