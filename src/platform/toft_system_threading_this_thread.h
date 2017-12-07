// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>
// Created: 05/31/11
// Description: current thread scoped attributes and operations

// toft/system/threading/this_thread.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THIS_THREAD_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THIS_THREAD_H_

//#pragma once

#include <stdint.h>
#include "platform/toft_system_threading_thread_types.h"

namespace bubblefs {
namespace mytoft {

/// thread scoped attribute and operations of current thread
class ThisThread
{
    ThisThread();
    ~ThisThread();
public:
    static void Exit();
    static void Yield();
    static void Sleep(int64_t time_in_ms);
    static int GetLastErrorCode();
    static ThreadHandleType GetHandle();
    static int GetId();
    static bool IsMain();
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THIS_THREAD_H_