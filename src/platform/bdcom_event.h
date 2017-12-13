// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// tera/src/common/event.h

#ifndef BUBBLEFS_PLATFORM_BDCOM_EVENT_H_
#define BUBBLEFS_PLATFORM_BDCOM_EVENT_H_

#include "platform/mutexlock.h"

namespace bubblefs {
namespace mybdcom {

class AutoResetEvent {
public:
    AutoResetEvent()
      : cv_(&mutex_), signaled_(false) {
    }
    /// Wait for signal
    void Wait() {
        MutexLock lock(&mutex_);
        while (!signaled_) {
            cv_.Wait();
        }
        signaled_ = false;
    }
    bool TimedWait(int64_t timeout) {
        MutexLock lock(&mutex_);
        if (!signaled_) {
            cv_.TimedWait(timeout);
        }
        bool ret = signaled_;
        signaled_ = false;
        return ret;
    }
    /// Signal one
    void Set() {
        MutexLock lock(&mutex_);
        signaled_ = true;
        cv_.Signal();
    }

private:
    port::Mutex mutex_;
    port::CondVar cv_;
    bool signaled_;
};

} // namespace mybdcom
} // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_BDCOM_EVENT_H_