// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/threading/spinlock.cpp

#include "platform/toft_system_threading_spinlock.h"
#include "platform/toft_system_check_error.h"
#include "platform/toft_system_threading_this_thread.h"

namespace bubblefs {
namespace mytoft {

SpinLock::SpinLock()
{
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_spin_init(&m_lock, 0));
    m_owner = 0;
}

SpinLock::~SpinLock()
{
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_spin_destroy(&m_lock));
    m_owner = -1;
}

void SpinLock::Lock()
{
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_spin_lock(&m_lock));
    m_owner = ThisThread::GetId();
}

bool SpinLock::TryLock()
{
    if (MYTOFT_CHECK_PTHREAD_TRYLOCK_ERROR(pthread_spin_trylock(&m_lock)))
    {
        m_owner = ThisThread::GetId();
        return true;
    }
    return false;
}

void SpinLock::Unlock()
{
    m_owner = 0;
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_spin_unlock(&m_lock));
}

} // namespace mytoft
} // namespace bubblefs