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

// android/platform_system_core/libutils/Timers.cpp
// protobuf/src/google/protobuf/stubs/time.cc

#include "platform/time.h"
#include <sys/time.h>
#include <sys/types.h>                       // open
#include <sys/stat.h>                        
#include <assert.h>
#include <fcntl.h>                           
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <syscall.h>                         // SYS_clock_gettime
#include <time.h>
#include <unistd.h>                          // syscall
#include "platform/macros.h"

namespace bubblefs {
namespace timeutil {  
  
// linux
nsecs_t systemTime(int clock)
{
    static const clockid_t clocks[] = {
            CLOCK_REALTIME,
            CLOCK_MONOTONIC,
            CLOCK_PROCESS_CPUTIME_ID,
            CLOCK_THREAD_CPUTIME_ID,
            CLOCK_BOOTTIME
    };
    struct timespec t;
    t.tv_sec = t.tv_nsec = 0;
    clock_gettime(clocks[clock], &t);
    return nsecs_t(t.tv_sec)*1000000000LL + t.tv_nsec;
}

ChronoTime chrono_now() {
  return std::chrono::high_resolution_clock::now();
}

double chrono_time_diff(ChronoTime t1, ChronoTime t2) {
  typedef std::chrono::microseconds ms;
  auto diff = t2 - t1;
  ms counter = std::chrono::duration_cast<ms>(diff);
  return counter.count() / 1000.0;
}

int toMillisecondTimeoutDelay(nsecs_t referenceTime, nsecs_t timeoutTime)
{
    nsecs_t timeoutDelayMillis;
    if (timeoutTime > referenceTime) {
        uint64_t timeoutDelay = uint64_t(timeoutTime - referenceTime);
        if (timeoutDelay > uint64_t((INT_MAX - 1) * 1000000LL)) {
            timeoutDelayMillis = -1;
        } else {
            timeoutDelayMillis = (timeoutDelay + 999999LL) / 1000000LL;
        }
    } else {
        timeoutDelayMillis = 0;
    }
    return (int)timeoutDelayMillis;
}

namespace {

// Count the seconds from the given year (start at Jan 1, 00:00) to 100 years
// after.
int64 SecondsPer100Years(int year) {
  if (year % 400 == 0 || year % 400 > 300) {
    return kSecondsPerDay * (100 * 365 + 100 / 4);
  } else {
    return kSecondsPerDay * (100 * 365 + 100 / 4 - 1);
  }
}

// Count the seconds from the given year (start at Jan 1, 00:00) to 4 years
// after.
int64 SecondsPer4Years(int year) {
  if ((year % 100 == 0 || year % 100 > 96) &&
      !(year % 400 == 0 || year % 400 > 396)) {
    // No leap years.
    return kSecondsPerDay * (4 * 365);
  } else {
    // One leap years.
    return kSecondsPerDay * (4 * 365 + 1);
  }
}

bool IsLeapYear(int year) {
  return year % 400 == 0 || (year % 4 == 0 && year % 100 != 0);
}

int64 SecondsPerYear(int year) {
  return kSecondsPerDay * (IsLeapYear(year) ? 366 : 365);
}

static const int kDaysInMonth[13] = {
  0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

int64 SecondsPerMonth(int month, bool leap) {
  if (month == 2 && leap) {
    return kSecondsPerDay * (kDaysInMonth[month] + 1);
  }
  return kSecondsPerDay * kDaysInMonth[month];
}

static const int kDaysSinceJan[13] = {
  0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334,
};

bool ValidateDateTime(const DateTime& time) {
  if (time.year < 1 || time.year > 9999 ||
      time.month < 1 || time.month > 12 ||
      time.day < 1 || time.day > 31 ||
      time.hour < 0 || time.hour > 23 ||
      time.minute < 0 || time.minute > 59 ||
      time.second < 0 || time.second > 59) {
    return false;
  }
  if (time.month == 2 && IsLeapYear(time.year)) {
    return time.month <= kDaysInMonth[time.month] + 1;
  } else {
    return time.month <= kDaysInMonth[time.month];
  }
}

// Count the number of seconds elapsed from 0001-01-01T00:00:00 to the given
// time.
int64 SecondsSinceCommonEra(const DateTime& time) {
  int64 result = 0;
  // Years should be between 1 and 9999.
  assert(time.year >= 1 && time.year <= 9999);
  int year = 1;
  if ((time.year - year) >= 400) {
    int count_400years = (time.year - year) / 400;
    result += kSecondsPer400Years * count_400years;
    year += count_400years * 400;
  }
  while ((time.year - year) >= 100) {
    result += SecondsPer100Years(year);
    year += 100;
  }
  while ((time.year - year) >= 4) {
    result += SecondsPer4Years(year);
    year += 4;
  }
  while (time.year > year) {
    result += SecondsPerYear(year);
    ++year;
  }
  // Months should be between 1 and 12.
  assert(time.month >= 1 && time.month <= 12);
  int month = time.month;
  result += kSecondsPerDay * kDaysSinceJan[month];
  if (month > 2 && IsLeapYear(year)) {
    result += kSecondsPerDay;
  }
  assert(time.day >= 1 &&
         time.day <= (month == 2 && IsLeapYear(year)
                          ? kDaysInMonth[month] + 1
                          : kDaysInMonth[month]));
  result += kSecondsPerDay * (time.day - 1);
  result += kSecondsPerHour * time.hour +
      kSecondsPerMinute * time.minute +
      time.second;
  return result;
}

}  // namespace

bool SecondsToDateTime(int64 seconds, DateTime* time) {
  if (seconds < kMinTime || seconds > kMaxTime) {
    return false;
  }
  // It's easier to calcuate the DateTime starting from 0001-01-01T00:00:00
  seconds = seconds + kSecondsFromEraToEpoch;
  int year = 1;
  if (seconds >= kSecondsPer400Years) {
    int count_400years = seconds / kSecondsPer400Years;
    year += 400 * count_400years;
    seconds %= kSecondsPer400Years;
  }
  while (seconds >= SecondsPer100Years(year)) {
    seconds -= SecondsPer100Years(year);
    year += 100;
  }
  while (seconds >= SecondsPer4Years(year)) {
    seconds -= SecondsPer4Years(year);
    year += 4;
  }
  while (seconds >= SecondsPerYear(year)) {
    seconds -= SecondsPerYear(year);
    year += 1;
  }
  bool leap = IsLeapYear(year);
  int month = 1;
  while (seconds >= SecondsPerMonth(month, leap)) {
    seconds -= SecondsPerMonth(month, leap);
    ++month;
  }
  int day = 1 + seconds / kSecondsPerDay;
  seconds %= kSecondsPerDay;
  int hour = seconds / kSecondsPerHour;
  seconds %= kSecondsPerHour;
  int minute = seconds / kSecondsPerMinute;
  seconds %= kSecondsPerMinute;
  time->year = year;
  time->month = month;
  time->day = day;
  time->hour = hour;
  time->minute = minute;
  time->second = static_cast<int>(seconds);
  return true;
}

bool DateTimeToSeconds(const DateTime& time, int64* seconds) {
  if (!ValidateDateTime(time)) {
    return false;
  }
  *seconds = SecondsSinceCommonEra(time) - kSecondsFromEraToEpoch;
  return true;
}

void GetCurrentTime(int64* seconds, int32* nanos) {
  // TODO(xiaofeng): Improve the accuracy of this implementation (or just
  // remove this method from protobuf).
  *seconds = time(nullptr);
  *nanos = 0;
}

int64_t clock_now_ns() {
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return static_cast<int64_t>(tp.tv_sec)*1000000000LL + tp.tv_nsec;
}

clockid_t get_monotonic_clockid() {
    // http://lxr.free-electrons.com/source/include/uapi/linux/time.h#L44
    const clockid_t MY_CLOCK_MONOTONIC_RAW = 4;
    
    timespec ts;
    if (0 == syscall(SYS_clock_gettime, MY_CLOCK_MONOTONIC_RAW, &ts)) {
        return MY_CLOCK_MONOTONIC_RAW;
    }
    return CLOCK_MONOTONIC;
}

extern const clockid_t monotonic_clockid = get_monotonic_clockid();

int64_t monotonic_time_ns() {
    timespec now;
    syscall(SYS_clock_gettime, monotonic_clockid, &now);
    return now.tv_sec * 1000000000L + now.tv_nsec;
}

/*
   read_cpu_frequency() is modified from source code of glibc.
   
   Copyright (C) 2002 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.
   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.
   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, write to the Free
   Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307 USA.  */
uint64_t read_cpu_frequency(bool* invariant_tsc) {
    /* We read the information from the /proc filesystem.  It contains at
       least one line like
       cpu MHz         : 497.840237
       or also
       cpu MHz         : 497.841
       We search for this line and convert the number in an integer.  */

    const int fd = open("/proc/cpuinfo", O_RDONLY);
    if (fd < 0) {
        return 0;
    }

    uint64_t result = 0;
    char buf[4096];  // should be enough
    const ssize_t n = read(fd, buf, sizeof(buf));
    if (n > 0) {
        char *mhz = static_cast<char*>(memmem(buf, n, "cpu MHz", 7));

        if (mhz != NULL) {
            char *endp = buf + n;
            int seen_decpoint = 0;
            int ndigits = 0;

            /* Search for the beginning of the string.  */
            while (mhz < endp && (*mhz < '0' || *mhz > '9') && *mhz != '\n') {
                ++mhz;
            }
            while (mhz < endp && *mhz != '\n') {
                if (*mhz >= '0' && *mhz <= '9') {
                    result *= 10;
                    result += *mhz - '0';
                    if (seen_decpoint)
                        ++ndigits;
                } else if (*mhz == '.') {
                    seen_decpoint = 1;
                }
                ++mhz;
            }

            /* Compensate for missing digits at the end.  */
            while (ndigits++ < 6) {
                result *= 10;
            }
        }

        if (invariant_tsc) {
            char* flags_pos = static_cast<char*>(memmem(buf, n, "flags", 5));
            *invariant_tsc = 
                (flags_pos &&
                 memmem(flags_pos, buf + n - flags_pos, "constant_tsc", 12) &&
                 memmem(flags_pos, buf + n - flags_pos, "nonstop_tsc", 11));
        }
    }
    close (fd);
    return result;
}

uint64_t read_invariant_cpu_frequency() {
    bool invariant_tsc = false;
    const uint64_t freq = read_cpu_frequency(&invariant_tsc);
    return (invariant_tsc ? freq : 0);
}

extern const uint64_t invariant_cpu_freq = read_invariant_cpu_frequency();

__thread int64_t tls_realtime_ns = 0;
__thread int64_t tls_cpuwidetime_ns = 0;

uint64_t NowMillis() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<uint64_t>(tv.tv_sec) * 1000 +
         static_cast<uint64_t>(tv.tv_usec) / 1000;
}

uint64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

void SleepForMicroseconds(int micros) { usleep(micros); }

}  // namespace timeutil
}  // namespace bubblefs