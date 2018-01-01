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
// caffe2/caffe2/core/timer.h

#ifndef BUBBLEFS_PLATFORM_TIME_H_
#define BUBBLEFS_PLATFORM_TIME_H_

#include <sys/time.h>
#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <chrono>
#include <string>
#include "platform/types.h"

namespace bubblefs {
namespace timeutil {

// The range of timestamp values we support.
constexpr int64 kMinTime = -62135596800LL;  // 0001-01-01T00:00:00
constexpr int64 kMaxTime = 253402300799LL;  // 9999-12-31T23:59:59

// The min/max Timestamp/Duration values we support.
//
// For "0001-01-01T00:00:00Z".
constexpr int64 kTimestampMinSeconds = -62135596800LL;
// For "9999-12-31T23:59:59.999999999Z".
constexpr int64 kTimestampMaxSeconds = 253402300799LL;
constexpr int64 kDurationMinSeconds = -315576000000LL;
constexpr int64 kDurationMaxSeconds = 315576000000LL;

constexpr char kTimestampFormat[] = "%E4Y-%m-%dT%H:%M:%S";
  
constexpr int64 kSecondsPerMinute = 60;
constexpr int64 kSecondsPerHour = 3600;
constexpr int64 kSecondsPerDay = kSecondsPerHour * 24;
constexpr int64 kSecondsPer400Years =
    kSecondsPerDay * (400 * 365 + 400 / 4 - 3);
// Seconds from 0001-01-01T00:00:00 to 1970-01-01T:00:00:00
constexpr int64 kSecondsFromEraToEpoch = 62135596800LL;

constexpr int64_t kMillisPerSecond = 1000;
constexpr int64_t kMicrosPerMillisecond = 1000;
constexpr int64_t kMicrosPerSecond = 1000000;
constexpr int64_t kMicrosPerMinute = kMicrosPerSecond * 60;
constexpr int64_t kMicrosPerHour = kMicrosPerMinute * 60;
constexpr int64_t kMicrosPerDay = kMicrosPerHour * 24;
constexpr int64_t kMicrosPerWeek = kMicrosPerDay * 7;
constexpr int64_t kNanosPerMicrosecond = 1000;
constexpr int64_t kNanosPerMillisecond = 1000000;
constexpr int64_t kNanosPerSecond = 1000000000;

using ChronoTime = decltype(std::chrono::high_resolution_clock::now());

ChronoTime chrono_now();

double chrono_time_diff(ChronoTime t1, ChronoTime t2);
  
typedef int64_t nsecs_t;       // nano-seconds

constexpr inline nsecs_t seconds_to_nanoseconds(nsecs_t secs)
{
    return secs*1000000000;
}

constexpr inline nsecs_t milliseconds_to_nanoseconds(nsecs_t secs)
{
    return secs*1000000;
}

constexpr inline nsecs_t microseconds_to_nanoseconds(nsecs_t secs)
{
    return secs*1000;
}

constexpr inline nsecs_t nanoseconds_to_seconds(nsecs_t secs)
{
    return secs/1000000000;
}

constexpr inline nsecs_t nanoseconds_to_milliseconds(nsecs_t secs)
{
    return secs/1000000;
}

constexpr inline nsecs_t nanoseconds_to_microseconds(nsecs_t secs)
{
    return secs/1000;
}

constexpr inline nsecs_t s2ns(nsecs_t v)  {return seconds_to_nanoseconds(v);}
constexpr inline nsecs_t ms2ns(nsecs_t v) {return milliseconds_to_nanoseconds(v);}
constexpr inline nsecs_t us2ns(nsecs_t v) {return microseconds_to_nanoseconds(v);}
constexpr inline nsecs_t ns2s(nsecs_t v)  {return nanoseconds_to_seconds(v);}
constexpr inline nsecs_t ns2ms(nsecs_t v) {return nanoseconds_to_milliseconds(v);}
constexpr inline nsecs_t ns2us(nsecs_t v) {return nanoseconds_to_microseconds(v);}

constexpr inline nsecs_t seconds(nsecs_t v)      { return s2ns(v); }
constexpr inline nsecs_t milliseconds(nsecs_t v) { return ms2ns(v); }
constexpr inline nsecs_t microseconds(nsecs_t v) { return us2ns(v); }

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

inline time_t gettimestamp(const std::string &time) {
    tm tm_;
    char buf[128] = { 0 };
    strncpy(buf, time.c_str(), sizeof(buf)-1);
    buf[sizeof(buf) - 1] = 0;
    strptime(buf, "%Y-%m-%d %H:%M:%S", &tm_);
    tm_.tm_isdst = -1;
    return mktime(&tm_);
}

inline time_t gettimediff(const std::string &t1, const std::string &t2) {
    time_t time1 = gettimestamp(t1);
    time_t time2 = gettimestamp(t2);
    time_t time = time1 - time2;
    return time;
}

inline void make_timeout(struct timespec* pts, long millisecond) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    pts->tv_sec = millisecond / 1000 + tv.tv_sec;
    pts->tv_nsec = (millisecond % 1000) * 1000000 + tv.tv_usec * 1000;

    pts->tv_sec += pts->tv_nsec / 1000000000;
    pts->tv_nsec = pts->tv_nsec % 1000000000;
}

inline std::string get_curtime_str() {
    struct tm tt;
    char buf[20];
    time_t t = time(nullptr);
    strftime(buf, 20, "%Y%m%d-%H:%M:%S", localtime_r(&t, &tt));
    return std::string(buf, 17);
}

inline std::string get_curtime_str_plain() {
    struct tm tt;
    char buf[20];
    time_t t = time(NULL);
    strftime(buf, 20, "%Y%m%d%H%M%S", localtime_r(&t, &tt));
    return std::string(buf);
}

inline long get_micros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<long>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

inline int64_t get_millis() {
    return static_cast<int64_t>(get_micros() / 1000); // ms
}

inline int64_t get_seconds() {
    return static_cast<int64_t>(get_micros() / 1000000); // s
}

inline int64_t get_unique_micros(int64_t ref) {
    int64_t now;
    do {
        now = get_micros();
    } while (now == ref);
    return now;
}

inline int64_t GetTimeStampInUs() {
    return get_micros();
}

inline int64_t GetTimeStampInMs() {
    return get_millis();
}

int64_t clock_now_ns(); // ns

// ----------------------
// timespec manipulations
// ----------------------

inline timespec make_timespec(int64_t sec, int64_t nsec) {
  timespec tm;
  tm.tv_sec = sec;
  tm.tv_nsec = nsec;
  return tm;
}

inline timespec max_timespec() {
  return make_timespec(-1, 0);
}

inline bool is_max_timespec(timespec &tm) {
  return (-1 == tm.tv_sec);
}

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

inline timeval make_timeval(int64_t sec, int64_t usec) {
  timeval tv;
  tv.tv_sec = sec;
  tv.tv_usec = usec;
  return tv;
}

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
    gettimeofday(&now, nullptr);
    return now.tv_sec * 1000000L + now.tv_usec;
}

inline int64_t gettimeofday_ms() {
    return gettimeofday_us() / 1000L;
}

inline int64_t gettimeofday_s() {
    return gettimeofday_us() / 1000000L;
}

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

/**
 * Return the current time in milliseconds since the Unix epoch.
 *
 * @return The number of milliseconds since the Unix epoch.
 */
// dmlc-core/include/dmlc/timer.h
inline double current_time_ms(void) {
  return std::chrono::duration<double>(
      std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// saber/saber/util/timeops.cc

extern uint64_t NowMillis();

extern uint64_t NowMicros();

extern void SleepForMicroseconds(int micros);

} // namespace timeutil
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TIME_H_