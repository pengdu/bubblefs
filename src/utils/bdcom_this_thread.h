// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// tera/src/common/this_thread.h

#ifndef  BUBBLEFS_UTILS_BDCOM_THIS_THREAD_H_
#define  BUBBLEFS_UTILS_BDCOM_THIS_THREAD_H_

#include <pthread.h>
#include <stdint.h>
#include <syscall.h>
#include <time.h>
#include <unistd.h>

namespace bubblefs {
namespace mybdcom {

class ThisThread {
public:
    /// Sleep in ms
    static void Sleep(int64_t time_ms) {
        if (time_ms > 0) {
            timespec ts = {time_ms / 1000, (time_ms % 1000) * 1000000 };
            nanosleep(&ts, &ts);
        }
    }
    /// Get thread id
    static int GetId() {
        return syscall(__NR_gettid);
    }
    /// Yield cpu
    static void Yield() {
        sched_yield();
    }
};

} // namespace mybdcom
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_BDCOM_THIS_THREAD_H_