// Copyright (c) 2010, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/time/posix_time.cpp

#include "platform/toft_system_time_posix_time.h"
#include <time.h>
#include <sys/time.h>

namespace bubblefs {
namespace mytoft {

void RelativeMilliSecondsToAbsolute(
    int64_t relative_time_in_ms,
    timespec* ts
    )
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64_t usec = tv.tv_usec + relative_time_in_ms * 1000LL;
    ts->tv_sec = tv.tv_sec + usec / 1000000;
    ts->tv_nsec = (usec % 1000000) * 1000;
}

} // namespace mytoft
} // namespace bubblefs