// Copyright 2013, Baidu Inc.
//
// Author: Qianqiong Zhang <zhangqianqiong02@baidu.com>

// toft/system/time/timestamp.cpp

#include "platform/toft_system_time_timestamp.h"
#include <sys/time.h>
#include <unistd.h>

namespace bubblefs {
namespace mytoft {

int64_t GetTimestampInUs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64_t result = tv.tv_sec;
    result *= 1000000;
    result += tv.tv_usec;
    return result;
}

int64_t GetTimestampInMs() {
    int64_t timestamp = GetTimestampInUs();
    return timestamp / 1000;
}

} // namespace mytoft
} // namespace bubblefs