// Copyright (c) 2011, The Toft Authors. All rights reserved.
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/threading/mutex.cpp

#include "platform/toft_system_threading_mutex.h"

namespace bubblefs {
namespace mytoft {
namespace internal {

static inline int DebugEnabled(int type)
{
#ifdef NDEBUG
    return type;
#else
    if (type == PTHREAD_MUTEX_RECURSIVE)
        return type;
    // Alwasy enable errcheck in debug mode
    return PTHREAD_MUTEX_ERRORCHECK_NP;
#endif
}

MutexBase::MutexBase(int type)
{
    type = DebugEnabled(type);
    pthread_mutexattr_t attr;
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_mutexattr_init(&attr));
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_mutexattr_settype(&attr, type));
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_mutex_init(&m_mutex, &attr));
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_mutexattr_destroy(&attr));
}

MutexBase::~MutexBase()
{
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_mutex_destroy(&m_mutex));
    // Since pthread_mutex_destroy will set __data.__kind to -1 and check
    // it in pthread_mutex_lock/pthread_mutex_unlock, nothing is necessary
    // to do for destructed access check.
}

} // namespace internal
} // namespace mytoft
} // namespace bubblefs