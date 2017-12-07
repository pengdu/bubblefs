// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/threading/spinlock.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_SPINLOCK_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_SPINLOCK_H_

#include <errno.h>
#include <stdlib.h>
#include <pthread.h>

#include "platform/toft_system_threading_scoped_locker.h"
#include "utils/toft_base_uncopyable.h"

namespace bubblefs {
namespace mytoft {

// SpinLock is faster than mutex at some condition, but
// some time may be slower, be careful!
class SpinLock
{
    DECLARE_UNCOPYABLE(SpinLock);
public:
    typedef ScopedLocker<SpinLock> Locker;
public:
    SpinLock();
    ~SpinLock();
    void Lock();
    bool TryLock();
    void Unlock();
private:
    pthread_spinlock_t m_lock;
    pid_t m_owner;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_SPINLOCK_H_