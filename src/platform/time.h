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
// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// protobuf/src/google/protobuf/stubs/time.h
// baidu/common/timer.h
// brpc/src/butil/time.h

#ifndef BUBBLEFS_PLATFORM_TIME_H_
#define BUBBLEFS_PLATFORM_TIME_H_

#include <sys/time.h>
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <string>
#include "platform/types.h"

namespace bubblefs {
namespace timeutil {

// The min/max Timestamp/Duration values we support.
//
// For "0001-01-01T00:00:00Z".
static const int64 kTimestampMinSeconds = -62135596800LL;
// For "9999-12-31T23:59:59.999999999Z".
static const int64 kTimestampMaxSeconds = 253402300799LL;
static const int64 kDurationMinSeconds = -315576000000LL;
static const int64 kDurationMaxSeconds = 315576000000LL;  
  
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
nsecs_t systemTime(int clock = CLOCK_MONOTONIC);

/**
 * Returns the number of milliseconds to wait between the reference time and the timeout time.
 * If the timeout is in the past relative to the reference time, returns 0.
 * If the timeout is more than INT_MAX milliseconds in the future relative to the reference time,
 * such as when timeoutTime == LLONG_MAX, returns -1 to indicate an infinite timeout delay.
 * Otherwise, returns the difference between the reference time and timeout time
 * rounded up to the next millisecond.
 */
int toMillisecondTimeoutDelay(nsecs_t referenceTime, nsecs_t timeoutTime);  
  
struct DateTime {
  int year;
  int month;
  int day;
  int hour;
  int minute;
  int second;
};

// Converts a timestamp (seconds elapsed since 1970-01-01T00:00:00, could be
// negative to represent time before 1970-01-01) to DateTime. Returns false
// if the timestamp is not in the range between 0001-01-01T00:00:00 and
// 9999-12-31T23:59:59.
bool SecondsToDateTime(int64 seconds, DateTime* time);
// Converts DateTime to a timestamp (seconds since 1970-01-01T00:00:00).
// Returns false if the DateTime is not valid or is not in the valid range.
bool DateTimeToSeconds(const DateTime& time, int64* seconds);

void GetCurrentTime(int64* seconds, int32* nanos);

// Formats a time string in RFC3339 fromat.
//
// For example, "2015-05-20T13:29:35.120Z". For nanos, 0, 3, 6 or 9 fractional
// digits will be used depending on how many are required to represent the exact
// value.
//
// Note that "nanos" must in the range of [0, 999999999].
string FormatTime(int64 seconds, int32 nanos);
// Parses a time string. This method accepts RFC3339 date/time string with UTC
// offset. For example, "2015-05-20T13:29:35.120-08:00".
bool ParseTime(const string& value, int64* seconds, int32* nanos);

class TimeUtil {
public:
    // 得到当前的毫秒
    static int64_t GetCurrentMS();

    // 得到当前的微妙
    static int64_t GetCurrentUS();

    // 得到字符串形式的时间 格式：2015-04-10 10:11:12
    static std::string GetStringTime();

    // 得到字符串形式的详细时间 格式: 2015-04-10 10:11:12.967151
    static const char* GetStringTimeDetail();

    // 将字符串格式(2015-04-10 10:11:12)的时间，转为time_t(时间戳)
    static time_t GetTimeStamp(const std::string &time);

    // 取得两个时间戳字符串t1-t2的时间差，精确到秒,时间格式为2015-04-10 10:11:12
    static time_t GetTimeDiff(const std::string &t1, const std::string &t2);
};

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

static inline time_t gettimestamp(const std::string &time) {
    tm tm_;
    char buf[128] = { 0 };
    strncpy(buf, time.c_str(), sizeof(buf)-1);
    buf[sizeof(buf) - 1] = 0;
    strptime(buf, "%Y-%m-%d %H:%M:%S", &tm_);
    tm_.tm_isdst = -1;
    return mktime(&tm_);
}

static inline time_t gettimediff(const std::string &t1, const std::string &t2) {
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

// ----------------------
// timespec manipulations
// ----------------------

// Let tm->tv_nsec be in [0, 1,000,000,000) if it's not.
inline void timespec_normalize(timespec* tm) {
    if (tm->tv_nsec >= 1000000000L) {
        const int64_t added_sec = tm->tv_nsec / 1000000000L;
        tm->tv_sec += added_sec;
        tm->tv_nsec -= added_sec * 1000000000L;
    } else if (tm->tv_nsec < 0) {
        const int64_t sub_sec = (tm->tv_nsec - 999999999L) / 1000000000L;
        tm->tv_sec += sub_sec;
        tm->tv_nsec -= sub_sec * 1000000000L;
    }
}

// Add timespec |span| into timespec |*tm|.
inline void timespec_add(timespec *tm, const timespec& span) {
    tm->tv_sec += span.tv_sec;
    tm->tv_nsec += span.tv_nsec;
    timespec_normalize(tm);
}

// Minus timespec |span| from timespec |*tm|. 
// tm->tv_nsec will be inside [0, 1,000,000,000)
inline void timespec_minus(timespec *tm, const timespec& span) {
    tm->tv_sec -= span.tv_sec;
    tm->tv_nsec -= span.tv_nsec;
    timespec_normalize(tm);
}

// ------------------------------------------------------------------
// Get the timespec after specified duration from |start_time|
// ------------------------------------------------------------------
inline timespec nanoseconds_from(timespec start_time, int64_t nanoseconds) {
    start_time.tv_nsec += nanoseconds;
    timespec_normalize(&start_time);
    return start_time;
}

inline timespec microseconds_from(timespec start_time, int64_t microseconds) {
    return nanoseconds_from(start_time, microseconds * 1000L);
}

inline timespec milliseconds_from(timespec start_time, int64_t milliseconds) {
    return nanoseconds_from(start_time, milliseconds * 1000000L);
}

inline timespec seconds_from(timespec start_time, int64_t seconds) {
    return nanoseconds_from(start_time, seconds * 1000000000L);
}

// --------------------------------------------------------------------
// Get the timespec after specified duration from now (CLOCK_REALTIME)
// --------------------------------------------------------------------
inline timespec nanoseconds_from_now(int64_t nanoseconds) {
    timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return nanoseconds_from(time, nanoseconds);
}

inline timespec microseconds_from_now(int64_t microseconds) {
    return nanoseconds_from_now(microseconds * 1000L);
}

inline timespec milliseconds_from_now(int64_t milliseconds) {
    return nanoseconds_from_now(milliseconds * 1000000L);
}

inline timespec seconds_from_now(int64_t seconds) {
    return nanoseconds_from_now(seconds * 1000000000L);
}

inline timespec timespec_from_now(const timespec& span) {
    timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    timespec_add(&time, span);
    return time;
}

// ---------------------------------------------------------------------
// Convert timespec to and from a single integer.
// For conversions between timespec and timeval, use TIMEVAL_TO_TIMESPEC
// and TIMESPEC_TO_TIMEVAL defined in <sys/time.h>
// ---------------------------------------------------------------------1
inline int64_t timespec_to_nanoseconds(const timespec& ts) {
    return ts.tv_sec * 1000000000L + ts.tv_nsec;
}

inline int64_t timespec_to_microseconds(const timespec& ts) {
    return timespec_to_nanoseconds(ts) / 1000L;
}

inline int64_t timespec_to_milliseconds(const timespec& ts) {
    return timespec_to_nanoseconds(ts) / 1000000L;
}

inline int64_t timespec_to_seconds(const timespec& ts) {
    return timespec_to_nanoseconds(ts) / 1000000000L;
}

inline timespec nanoseconds_to_timespec(int64_t ns) {
    timespec ts;
    ts.tv_sec = ns / 1000000000L;
    ts.tv_nsec = ns - ts.tv_sec * 1000000000L;
    return ts;
}

inline timespec microseconds_to_timespec(int64_t us) {
    return nanoseconds_to_timespec(us * 1000L);
}

inline timespec milliseconds_to_timespec(int64_t ms) {
    return nanoseconds_to_timespec(ms * 1000000L);
}

inline timespec seconds_to_timespec(int64_t s) {
    return nanoseconds_to_timespec(s * 1000000000L);
}

// ---------------------------------------------------------------------
// Convert timeval to and from a single integer.                                             
// For conversions between timespec and timeval, use TIMEVAL_TO_TIMESPEC
// and TIMESPEC_TO_TIMEVAL defined in <sys/time.h>
// ---------------------------------------------------------------------
inline int64_t timeval_to_microseconds(const timeval& tv) {
    return tv.tv_sec * 1000000L + tv.tv_usec;
}

inline int64_t timeval_to_milliseconds(const timeval& tv) {
    return timeval_to_microseconds(tv) / 1000L;
}

inline int64_t timeval_to_seconds(const timeval& tv) {
    return timeval_to_microseconds(tv) / 1000000L;
}

inline timeval microseconds_to_timeval(int64_t us) {
    timeval tv;
    tv.tv_sec = us / 1000000L;
    tv.tv_usec = us - tv.tv_sec * 1000000L;
    return tv;
}

inline timeval milliseconds_to_timeval(int64_t ms) {
    return microseconds_to_timeval(ms * 1000L);
}

inline timeval seconds_to_timeval(int64_t s) {
    return microseconds_to_timeval(s * 1000000L);
}

// ---------------------------------------------------------------
// Get system-wide monotonic time.
// Cost ~85ns on 2.6.32_1-12-0-0, Intel(R) Xeon(R) CPU E5620 @ 2.40GHz
// ---------------------------------------------------------------
extern int64_t monotonic_time_ns();

inline int64_t monotonic_time_us() { 
    return monotonic_time_ns() / 1000L; 
}

inline int64_t monotonic_time_ms() {
    return monotonic_time_ns() / 1000000L; 
}

inline int64_t monotonic_time_s() {
    return monotonic_time_ns() / 1000000000L;
}

namespace detail {
inline uint64_t clock_cycles() {
    unsigned int lo = 0;
    unsigned int hi = 0;
    // We cannot use "=A", since this would use %rax on x86_64
    __asm__ __volatile__ (
        "rdtsc"
        : "=a" (lo), "=d" (hi)
        );
    return ((uint64_t)hi << 32) | lo;
}
}  // namespace detail

// ---------------------------------------------------------------
// Get cpu-wide (wall-) time.
// Cost ~9ns on Intel(R) Xeon(R) CPU E5620 @ 2.40GHz
// ---------------------------------------------------------------
inline int64_t cpuwide_time_ns() {
    extern const uint64_t invariant_cpu_freq;  // will be non-zero iff:
    // 1 Intel x86_64 CPU (multiple cores) supporting constant_tsc and
    // nonstop_tsc(check flags in /proc/cpuinfo)
    
    if (invariant_cpu_freq) {
        const uint64_t tsc = detail::clock_cycles();
        const uint64_t sec = tsc / invariant_cpu_freq;
        // TODO: should be OK until CPU's frequency exceeds 16GHz.
        return (tsc - sec * invariant_cpu_freq) * 1000000000L /
            invariant_cpu_freq + sec * 1000000000L;
    }
    // Lack of necessary features, return system-wide monotonic time instead.
    return monotonic_time_ns();
}

inline int64_t cpuwide_time_us() {
    return cpuwide_time_ns() / 1000L;
}

inline int64_t cpuwide_time_ms() { 
    return cpuwide_time_ns() / 1000000L;
}

inline int64_t cpuwide_time_s() {
    return cpuwide_time_ns() / 1000000000L;
}

// --------------------------------------------------------------------
// Get elapse since the Epoch.                                          
// No gettimeofday_ns() because resolution of timeval is microseconds.  
// Cost ~40ns on 2.6.32_1-12-0-0, Intel(R) Xeon(R) CPU E5620 @ 2.40GHz
// --------------------------------------------------------------------
inline int64_t gettimeofday_us() {
    timeval now;
    gettimeofday(&now, NULL);
    return now.tv_sec * 1000000L + now.tv_usec;
}

inline int64_t gettimeofday_ms() {
    return gettimeofday_us() / 1000L;
}

inline int64_t gettimeofday_s() {
    return gettimeofday_us() / 1000000L;
}

// ----------------------------------------
// Control frequency of operations.
// ----------------------------------------
// Example:
//   EveryManyUS every_1s(1000000L);
//   while (1) {
//       ...
//       if (every_1s) {
//           // be here at most once per second
//       }
//   }
class EveryManyUS {
public:
    explicit EveryManyUS(int64_t interval_us)
        : _last_time_us(cpuwide_time_us())
        , _interval_us(interval_us) {}
    
    operator bool() {
        const int64_t now_us = cpuwide_time_us();
        if (now_us < _last_time_us + _interval_us) {
            return false;
        }
        _last_time_us = now_us;
        return true;
    }

private:
    int64_t _last_time_us;
    const int64_t _interval_us;
};

// ---------------
//  Count elapses
// ---------------
class Timer {
public:

    enum TimerType {
        STARTED,
    };

    Timer() : _stop(0), _start(0) {}
    explicit Timer(const TimerType) {
        start();
    }

    // Start this timer
    void start() {
        _start = cpuwide_time_ns();
        _stop = _start;
    }
    
    // Stop this timer
    void stop() {
        _stop = cpuwide_time_ns();
    }

    // Get the elapse from start() to stop(), in various units.
    int64_t n_elapsed() const { return _stop - _start; }
    int64_t u_elapsed() const { return n_elapsed() / 1000L; }
    int64_t m_elapsed() const { return u_elapsed() / 1000L; }
    int64_t s_elapsed() const { return m_elapsed() / 1000L; }

    double n_elapsed(double) const { return (double)(_stop - _start); }
    double u_elapsed(double) const { return (double)n_elapsed() / 1000.0; }
    double m_elapsed(double) const { return (double)u_elapsed() / 1000.0; }
    double s_elapsed(double) const { return (double)m_elapsed() / 1000.0; }
    
private:
    int64_t _stop;
    int64_t _start;
};

// NOTE: Don't call fast_realtime*! they're still experimental.
inline int64_t fast_realtime_ns() {
    extern const uint64_t invariant_cpu_freq;
    extern __thread int64_t tls_cpuwidetime_ns;
    extern __thread int64_t tls_realtime_ns;
    
    if (invariant_cpu_freq) {
        // 1 Intel x86_64 CPU (multiple cores) supporting constant_tsc and
        // nonstop_tsc(check flags in /proc/cpuinfo)
    
        const uint64_t tsc = detail::clock_cycles();
        const uint64_t sec = tsc / invariant_cpu_freq;
        // TODO: should be OK until CPU's frequency exceeds 16GHz.
        const int64_t diff = (tsc - sec * invariant_cpu_freq) * 1000000000L /
            invariant_cpu_freq + sec * 1000000000L - tls_cpuwidetime_ns;
        if (__builtin_expect(diff < 10000000, 1)) {
            return diff + tls_realtime_ns;
        }
        timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        tls_cpuwidetime_ns += diff;
        tls_realtime_ns = timespec_to_nanoseconds(ts);
        return tls_realtime_ns;
    }
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return timespec_to_nanoseconds(ts);
}

inline int fast_realtime(timespec* ts) {
    extern const uint64_t invariant_cpu_freq;
    extern __thread int64_t tls_cpuwidetime_ns;
    extern __thread int64_t tls_realtime_ns;
    
    if (invariant_cpu_freq) {    
        const uint64_t tsc = detail::clock_cycles();
        const uint64_t sec = tsc / invariant_cpu_freq;
        // TODO: should be OK until CPU's frequency exceeds 16GHz.
        const int64_t diff = (tsc - sec * invariant_cpu_freq) * 1000000000L /
            invariant_cpu_freq + sec * 1000000000L - tls_cpuwidetime_ns;
        if (__builtin_expect(diff < 10000000, 1)) {
            const int64_t now = diff + tls_realtime_ns;
            ts->tv_sec = now / 1000000000L;
            ts->tv_nsec = now - ts->tv_sec * 1000000000L;
            return 0;
        }
        const int rc = clock_gettime(CLOCK_REALTIME, ts);
        tls_cpuwidetime_ns += diff;
        tls_realtime_ns = timespec_to_nanoseconds(*ts);
        return rc;
    }
    return clock_gettime(CLOCK_REALTIME, ts);
}

} // namespace timeutil
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TIME_H_