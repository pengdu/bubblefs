// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/threading/rwlock.cpp

#include "platform/toft_system_threading_rwlock.h"

#include <string.h>

namespace bubblefs {
namespace mytoft {

RwLock::RwLock()
{
    // Note: default rwlock is prefer reader
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_rwlock_init(&m_lock, NULL));
}

RwLock::RwLock(Kind kind)
{
    pthread_rwlockattr_t attr;
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_rwlockattr_init(&attr));
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_rwlockattr_setkind_np(&attr, kind));
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_rwlock_init(&m_lock, &attr));
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_rwlockattr_destroy(&attr));
}

RwLock::~RwLock()
{
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_rwlock_destroy(&m_lock));
    memset(&m_lock, 0xFF, sizeof(m_lock));
}

} // namespace mytoft
} // namespace bubblefs