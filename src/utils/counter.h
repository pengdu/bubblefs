// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// baidu/common/include/counter.h

#ifndef  BUBBLEFS_UTILS_COUNTER_H_
#define  BUBBLEFS_UTILS_COUNTER_H_

#include "platform/atomicops.h"

namespace bubblefs {
namespace bdcommon {
  
class Counter {
    volatile int64_t val_;
public:
    Counter(int64_t val = 0) : val_(val) {}
    int64_t Add(int64_t v) {
        return atomic_add64(&val_, v) + v;
    }
    int64_t Sub(int64_t v) {
        return atomic_add64(&val_, -v) - v;
    }
    int64_t Inc() {
        return atomic_add64(&val_, 1) + 1;
    }
    int64_t Dec() {
        return atomic_add64(&val_,-1) - 1;
    }
    int64_t Get() const {
        return val_;
    }
    int64_t Set(int64_t v) {
        return atomic_swap64(&val_, v);
    }
    int64_t Clear() {
        return atomic_swap64(&val_, 0);
    }
};

} // namespace bdcommon
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_COUNTER_H_