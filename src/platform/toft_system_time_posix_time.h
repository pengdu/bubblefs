// Copyright (c) 2010, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/time/posix_time.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_TIME_POSIX_TIME_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_TIME_POSIX_TIME_H_

#include <stdint.h>

struct timespec;

namespace bubblefs {
namespace mytoft {

// for any timed* functions using absolute timespec
void RelativeMilliSecondsToAbsolute(
    int64_t relative_time_in_ms,
    timespec* ts
);

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_TIME_POSIX_TIME_H_
