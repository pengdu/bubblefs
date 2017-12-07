// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/threading/condition_variable.cpp

#include "platform/toft_system_threading_condition_variable.h"

#include <assert.h>
#include <sys/time.h>

#include <stdexcept>
#include <string>

#include "platform/toft_system_check_error.h"
#include "platform/toft_system_threading_mutex.h"
#include "platform/toft_system_time_posix_time.h"

namespace bubblefs {
namespace mytoft {

ConditionVariable::ConditionVariable(internal::MutexBase* mutex)
{
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_cond_init(&m_cond, NULL));
    m_mutex = mutex;
}

ConditionVariable::~ConditionVariable()
{
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_cond_destroy(&m_cond));
    m_mutex = NULL;
}

void ConditionVariable::CheckValid() const
{
    // __total_seq will be set to -1 by pthread_cond_destroy
    assert(m_mutex != NULL && "this cond has been destructed");
}

void ConditionVariable::Signal()
{
    CheckValid();
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_cond_signal(&m_cond));
}

void ConditionVariable::Broadcast()
{
    CheckValid();
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_cond_broadcast(&m_cond));
}

void ConditionVariable::Wait()
{
    CheckValid();
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_cond_wait(&m_cond, &m_mutex->m_mutex));
}

bool ConditionVariable::TimedWait(int64_t timeout_in_ms)
{
    timespec ts;
    RelativeMilliSecondsToAbsolute(timeout_in_ms, &ts);
    return MYTOFT_CHECK_PTHREAD_TIMED_ERROR(
        pthread_cond_timedwait(&m_cond, &m_mutex->m_mutex, &ts));
}

} // namespace mytoft
} // namespace bubblefs