// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/threading/thread_group.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THREAD_GROUP_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THREAD_GROUP_H_

//#pragma once

#include <functional>
#include <vector>

#include "platform/toft_system_threading_thread.h"
#include "utils/toft_base_uncopyable.h"

namespace bubblefs {
namespace mytoft {

class ThreadGroup
{
    DECLARE_UNCOPYABLE(ThreadGroup);
public:
    ThreadGroup();
    ThreadGroup(const std::function<void ()>& callback, size_t count);
    ~ThreadGroup();
    void Add(const std::function<void ()>& callback, size_t count = 1);
    void Start();
    void Join();
    size_t Size() const;
private:
    std::vector<Thread*> m_threads;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THREAD_GROUP_H_