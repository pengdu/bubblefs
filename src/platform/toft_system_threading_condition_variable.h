// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/threading/condition_variable.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_CONDITION_VARIABLE_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_CONDITION_VARIABLE_H_

#include <assert.h>
#include <pthread.h>
#include <stdint.h>

namespace bubblefs {
namespace mytoft {
  
namespace internal {
class MutexBase;
} // namespace internal

class ConditionVariable
{
public:
    explicit ConditionVariable(internal::MutexBase* mutex);
    ~ConditionVariable();
    void Signal();
    void Broadcast();
    void Wait();
    bool TimedWait(int64_t timeout_in_ms);
private:
    void CheckValid() const;
private:
    pthread_cond_t m_cond;
    internal::MutexBase* m_mutex;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_CONDITION_VARIABLE_H_