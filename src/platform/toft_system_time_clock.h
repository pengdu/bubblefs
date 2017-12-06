// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/time/clock.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_TIME_CLOCK_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_TIME_CLOCK_H_

//#pragma once

#include <stdint.h>

namespace bubblefs {
namespace mytoft {

class Clock {
public:
    virtual ~Clock() {}
    // 19700101000000L
    virtual int64_t StartTime() = 0;

    virtual int64_t MicroSeconds() = 0;
    virtual int64_t MilliSeconds() = 0;
    virtual int64_t Seconds() = 0;

    virtual bool Set(int64_t microseconds) = 0;
};

extern Clock& RealtimeClock;

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_TIME_CLOCK_H_