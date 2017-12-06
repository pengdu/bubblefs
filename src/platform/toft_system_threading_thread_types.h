// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>
// Created: 05/31/11

// toft/system/threading/thread_types.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THREAD_TYPES_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THREAD_TYPES_H_

//#pragma once

#include <pthread.h>
#include <string>

namespace bubblefs {
namespace mytoft {

typedef pthread_t ThreadHandleType;

class BaseThread;

/// ThreadAttribute represent thread attribute.
/// Usage:
/// ThreadAttribute()
///     .SetName("ThreadPoolThread")
///     .SetStackSize(64 * 1024)
class ThreadAttributes {
    friend class BaseThread;
public:
    ThreadAttributes();
    ~ThreadAttributes();
    ThreadAttributes& SetName(const std::string& name);
    ThreadAttributes& SetStackSize(size_t size);
    ThreadAttributes& SetDetached(bool detached);
    ThreadAttributes& SetPriority(int priority);
    bool IsDetached() const;
private:
    std::string m_name;
    pthread_attr_t m_attr;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THREAD_TYPES_H_