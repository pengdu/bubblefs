// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/process/process_local_const.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_PROCESS_PROCESS_LOCAL_CONST_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_PROCESS_PROCESS_LOCAL_CONST_H_

//#pragma once

#include <unistd.h>

#include "platform/toft_system_memory_barrier.h"
#include "utils/toft_base_linking_assert.h"

namespace bubblefs {
namespace mytoft {

MYTOFT_DECLARE_LINKING_ASSERT(ProcessLocalConst);

// Fork aware process local storage.
// Exampls:
//
// ThisProcess::StartTime() {
//     // Yes, should be static
//     static ProcessLocalConst<time_t> s_start_time(GetStartTime);
//     return s_start_time.Value();
// }
//
// Cache the result of the first call of GetStartTime().
// But if the process forked, update it again.
//
template <typename T>
class ProcessLocalConst : private MYTOFT_LINKING_ASSERT_BASE(ProcessLocalConst) {
    typedef T (*GetFunction)();
public:
    explicit ProcessLocalConst(GetFunction get) :
#ifdef __unix__
        m_get(get), m_pid(getpid()),
#endif
        m_value(get()) {}

    const T& Value() const {
#ifdef __unix__
        int pid = getpid();
        if (pid != m_pid) {
            m_value = m_get();
            MemoryWriteBarrier();
            m_pid = pid;
        }
        MemoryReadBarrier();
#endif
        return m_value;
    }
private:
#ifdef __unix__
    mutable GetFunction m_get;
    mutable pid_t m_pid;
#endif
    mutable T m_value;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_PROCESS_PROCESS_LOCAL_CONST_H_