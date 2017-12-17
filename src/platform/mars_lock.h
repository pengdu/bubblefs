// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

// mars/mars/comm/unix/thread/lock.h

#ifndef BUBBLEFS_PLATFORM_MARS_LOCK_H_
#define BUBBLEFS_PLATFORM_MARS_LOCK_H_

#include <unistd.h>

#include "platform/macros.h"
#include "platform/mars_mutex.h"
#include "platform/mars_spinlock.h"
#include "platform/mars_time_utils.h"

namespace bubblefs {
namespace mymars {

template <typename MutexType>
class BaseScopedLock {
  public:
    explicit BaseScopedLock(MutexType& mutex, bool initiallyLocked = true)
        : mutex_(mutex) , islocked_(false) {
        if (!initiallyLocked) return;

        lock();
    }

    explicit BaseScopedLock(MutexType& mutex, long _millisecond)
        : mutex_(mutex) , islocked_(false) {
        timedlock(_millisecond);
    }

    ~BaseScopedLock() {
        if (islocked_) unlock();
    }

    bool islocked() const {
        return islocked_;
    }

    void lock() {
        ASSERT(!islocked_);

        if (!islocked_ && mutex_.lock()) {
            islocked_ = true;
        }

        ASSERT(islocked_);
    }

    void unlock() {
        ASSERT(islocked_);

        if (islocked_) {
            mutex_.unlock();
            islocked_ = false;
        }
    }

    bool trylock() {
        if (islocked_) return false;

        islocked_ = mutex_.trylock();
        return islocked_;
    }

    //#ifdef __linux__
    bool timedlock(long _millisecond) {
        ASSERT(!islocked_);

        if (islocked_) return true;

        islocked_ = mutex_.timedlock(_millisecond);
        return islocked_;
    }

    MutexType& internal() {
        return mutex_;
    }

  private:
    MutexType& mutex_;
    bool islocked_;
};

typedef BaseScopedLock<Mutex> ScopedLock;
typedef BaseScopedLock<SpinLock> ScopedSpinLock;

} // namespace mymars
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_MARS_LOCK_H_