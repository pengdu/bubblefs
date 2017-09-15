// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com
/*
 * Copyright (C) 2005 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// baidu/common/timer.h  

#ifndef BUBBLEFS_PLATFORM_TIMER_H
#define BUBBLEFS_PLATFORM_TIMER_H

#include <sys/time.h>
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <string>

namespace bubblefs {
namespace timing {

typedef int64_t nsecs_t;       // nano-seconds

static constexpr inline nsecs_t seconds_to_nanoseconds(nsecs_t secs)
{
    return secs*1000000000;
}

static constexpr inline nsecs_t milliseconds_to_nanoseconds(nsecs_t secs)
{
    return secs*1000000;
}

static constexpr inline nsecs_t microseconds_to_nanoseconds(nsecs_t secs)
{
    return secs*1000;
}

static constexpr inline nsecs_t nanoseconds_to_seconds(nsecs_t secs)
{
    return secs/1000000000;
}

static constexpr inline nsecs_t nanoseconds_to_milliseconds(nsecs_t secs)
{
    return secs/1000000;
}

static constexpr inline nsecs_t nanoseconds_to_microseconds(nsecs_t secs)
{
    return secs/1000;
}

static constexpr inline nsecs_t s2ns(nsecs_t v)  {return seconds_to_nanoseconds(v);}
static constexpr inline nsecs_t ms2ns(nsecs_t v) {return milliseconds_to_nanoseconds(v);}
static constexpr inline nsecs_t us2ns(nsecs_t v) {return microseconds_to_nanoseconds(v);}
static constexpr inline nsecs_t ns2s(nsecs_t v)  {return nanoseconds_to_seconds(v);}
static constexpr inline nsecs_t ns2ms(nsecs_t v) {return nanoseconds_to_milliseconds(v);}
static constexpr inline nsecs_t ns2us(nsecs_t v) {return nanoseconds_to_microseconds(v);}

static constexpr inline nsecs_t seconds(nsecs_t v)      { return s2ns(v); }
static constexpr inline nsecs_t milliseconds(nsecs_t v) { return ms2ns(v); }
static constexpr inline nsecs_t microseconds(nsecs_t v) { return us2ns(v); }

// return the system-time according to the specified clock
#ifdef __cplusplus
nsecs_t systemTime(int clock = CLOCK_MONOTONIC);
#else
nsecs_t systemTime(int clock);
#endif // def __cplusplus

/**
 * Returns the number of milliseconds to wait between the reference time and the timeout time.
 * If the timeout is in the past relative to the reference time, returns 0.
 * If the timeout is more than INT_MAX milliseconds in the future relative to the reference time,
 * such as when timeoutTime == LLONG_MAX, returns -1 to indicate an infinite timeout delay.
 * Otherwise, returns the difference between the reference time and timeout time
 * rounded up to the next millisecond.
 */
int toMillisecondTimeoutDelay(nsecs_t referenceTime, nsecs_t timeoutTime);  
  
enum Precision {
    kDay,
    kMin,
    kUsec,
};

static inline long get_micros() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<long>(tv.tv_sec) * 1000000 + tv.tv_usec; // us
}

static inline long get_millis() {
   return static_cast<int32_t>(get_micros() / 1000); // ms
}

static inline int32_t now_time() {
    return static_cast<int32_t>(get_micros() / 1000000); // s
}

static inline int32_t now_time_str(char* buf, int32_t len, Precision p = kUsec) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    const time_t seconds = tv.tv_sec;
    struct tm t;
    localtime_r(&seconds, &t);
    int32_t ret = 0;
    if (p == kDay) {
        ret = snprintf(buf, len, "%02d/%02d",
                t.tm_mon + 1,
                t.tm_mday);
    } else if (p == kMin) {
        ret = snprintf(buf, len, "%02d/%02d %02d:%02d",
                t.tm_mon + 1,
                t.tm_mday,
                t.tm_hour,
                t.tm_min);
    } else {
        ret = snprintf(buf, len, "%02d/%02d %02d:%02d:%02d.%06d",
            t.tm_mon + 1,
            t.tm_mday,
            t.tm_hour,
            t.tm_min,
            t.tm_sec,
            static_cast<int>(tv.tv_usec));
    }
    return ret;
}

static inline gettimestamp(const std::string &time) {
    tm tm_;
    char buf[128] = { 0 };
    strncpy(buf, time.c_str(), sizeof(buf)-1);
    buf[sizeof(buf) - 1] = 0;
    strptime(buf, "%Y-%m-%d %H:%M:%S", &tm_);
    tm_.tm_isdst = -1;
    return mktime(&tm_);
}

static inline gettimediff(const std::string &t1, const std::string &t2) {
    time_t time1 = gettimestamp(t1);
    time_t time2 = gettimestamp(t2);
    time_t time = time1 - time2;
    return time;
}

static inline void make_timeout(struct timespec* pts, long millisecond) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    pts->tv_sec = millisecond / 1000 + tv.tv_sec;
    pts->tv_nsec = (millisecond % 1000) * 1000 * 1000 + tv.tv_usec * 1000;

    pts->tv_sec += pts->tv_nsec / (1000 * 1000 * 1000);
    pts->tv_nsec = pts->tv_nsec % (1000 * 1000 * 1000);
}

class AutoTimer {
public:
    AutoTimer(double timeout_ms = -1, const char* msg1 = nullptr, const char* msg2 = nullptr)
      : timeout_(timeout_ms),
        msg1_(msg1),
        msg2_(msg2) {
        start_ = get_micros();
    }
    int64_t TimeUsed() const {
        return get_micros() - start_;
    }
    ~AutoTimer() {
        if (timeout_ == -1) return;
        long end = get_micros();
        if (end - start_ > timeout_ * 1000) {
            double t = (end - start_) / 1000.0;
            if (!msg2_) {
                fprintf(stderr, "[AutoTimer] %s use %.3f ms\n",
                    msg1_, t);
            } else {
                fprintf(stderr, "[AutoTimer] %s %s use %.3f ms\n",
                    msg1_, msg2_, t);
            }
        }
    }
private:
    long start_;
    double timeout_;
    const char* msg1_;
    const char* msg2_;
};

class TimeChecker {
public:
    TimeChecker() {
        start_ = get_micros();
    }
    void Check(int64_t timeout, const std::string& msg) {
        int64_t now = get_micros();
        int64_t interval = now - start_;
        if (timeout == -1 || interval > timeout) {
            char buf[30];
            now_time_str(buf, 30);
            fprintf(stderr, "[TimeChecker] %s %s use %ld us\n", buf, msg.c_str(), interval);
        }
        start_ = get_micros();
    }
    void Reset() {
        start_ = get_micros();
    }
private:
    int64_t start_;
};

} // namespace timing
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TIMER_H