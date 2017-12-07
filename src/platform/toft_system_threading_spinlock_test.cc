// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/threading/spinlock_test.cpp

#include "platform/toft_system_threading_mutex.h"
#include "platform/toft_system_threading_spinlock.h"
#include "platform/toft_system_threading_thread_group.h"

#include "gtest/gtest.h"

namespace bubblefs {
namespace mytoft {

const int kLoopCount = 10000000;

TEST(MutexTest, Mutex)
{
    Mutex lock;
    for (int i = 0; i < kLoopCount; ++i)
    {
        Mutex::Locker locker(&lock);
    }
}

void TestThread(int* p, Mutex* mutex)
{
    for (;;)
    {
        Mutex::Locker locker(mutex);
        if (++(*p) >= kLoopCount)
            return;
    }
}

TEST(MutexTest, ThreadMutex)
{
    int n = 0;
    Mutex lock;
    ThreadGroup thread_group(std::bind(TestThread, &n, &lock), 4);
    thread_group.Join();
}

TEST(SpinLockTest, SpinLock)
{
    SpinLock lock;
    for (int i = 0; i < kLoopCount; ++i)
    {
        SpinLock::Locker locker(&lock);
    }
}

} // namespace mytoft
} // namespace bubblefs