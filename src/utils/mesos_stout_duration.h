// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// mesos/3rdparty/stout/include/stout/duration.hpp
// mesos/3rdparty/stout/include/stout/stopwatch.hpp

#ifndef BUBBLEFS_UTILS_MESOS_STOUT_DURATION_H_
#define BUBBLEFS_UTILS_MESOS_STOUT_DURATION_H_

#include <ctype.h> // For 'isdigit'.
#include <stdint.h>
#include <time.h>

// For 'timeval'.
#ifndef __WINDOWS__
#include <sys/time.h>
#endif // __WINDOWS__

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "utils/mesos_stout_error.h"
#include "utils/mesos_stout_numify.h"
#include "utils/mesos_stout_try.h"

namespace bubblefs {
namespace mymesos {

class Duration
{
public:
  static Try<Duration> parse(const std::string& s)
  {
    // TODO(benh): Support negative durations (i.e., starts with '-').
    size_t index = 0;
    while (index < s.size()) {
      if (isdigit(s[index]) || s[index] == '.') {
        index++;
        continue;
      }

      Try<double> value = numify<double>(s.substr(0, index));

      if (value.isError()) {
        return Error(value.error());
      }

      const std::string unit = s.substr(index);

      int64_t factor;
      if (unit == "ns") {
        factor = NANOSECONDS;
      } else if (unit == "us") {
        factor = MICROSECONDS;
      } else if (unit == "ms") {
        factor = MILLISECONDS;
      } else if (unit == "secs") {
        factor = SECONDS;
      } else if (unit == "mins") {
        factor = MINUTES;
      } else if (unit == "hrs") {
        factor = HOURS;
      } else if (unit == "days") {
        factor = DAYS;
      } else if (unit == "weeks") {
        factor = WEEKS;
      } else {
        return Error(
            "Unknown duration unit '" + unit + "'; supported units are"
            " 'ns', 'us', 'ms', 'secs', 'mins', 'hrs', 'days', and 'weeks'");
      }

      double nanos = value.get() * factor;
      if (nanos > max().nanos || nanos < min().nanos) {
        return Error(
            "Argument out of the range that a Duration can represent due"
            " to int64_t's size limit");
      }

      return Duration(value.get(), factor);
    }

    return Error("Invalid duration '" + s + "'");
  }

  static Try<Duration> create(double seconds);

  constexpr Duration() : nanos(0) {}

  explicit Duration(const timeval& t)
  {
    nanos = t.tv_sec * SECONDS + t.tv_usec * MICROSECONDS;
  }

  int64_t ns() const   { return nanos; }
  double us() const    { return static_cast<double>(nanos) / MICROSECONDS; }
  double ms() const    { return static_cast<double>(nanos) / MILLISECONDS; }
  double secs() const  { return static_cast<double>(nanos) / SECONDS; }
  double mins() const  { return static_cast<double>(nanos) / MINUTES; }
  double hrs() const   { return static_cast<double>(nanos) / HOURS; }
  double days() const  { return static_cast<double>(nanos) / DAYS; }
  double weeks() const { return static_cast<double>(nanos) / WEEKS; }

  struct timeval timeval() const
  {
    struct timeval t;

    // Explicitly compute `tv_sec` and `tv_usec` instead of using `us` and
    // `secs` to avoid converting `int64_t` -> `double` -> `long`.
    t.tv_sec = static_cast<decltype(t.tv_sec)>(ns() / SECONDS);
    t.tv_usec = static_cast<decltype(t.tv_usec)>(
        (ns() / MICROSECONDS) - (t.tv_sec * SECONDS / MICROSECONDS));
    return t;
  }

  bool operator<(const Duration& d) const { return nanos < d.nanos; }
  bool operator<=(const Duration& d) const { return nanos <= d.nanos; }
  bool operator>(const Duration& d) const { return nanos > d.nanos; }
  bool operator>=(const Duration& d) const { return nanos >= d.nanos; }
  bool operator==(const Duration& d) const { return nanos == d.nanos; }
  bool operator!=(const Duration& d) const { return nanos != d.nanos; }

  Duration& operator+=(const Duration& that)
  {
    nanos += that.nanos;
    return *this;
  }

  Duration& operator-=(const Duration& that)
  {
    nanos -= that.nanos;
    return *this;
  }

  Duration& operator*=(double multiplier)
  {
    nanos = static_cast<int64_t>(nanos * multiplier);
    return *this;
  }

  Duration& operator/=(double divisor)
  {
    nanos = static_cast<int64_t>(nanos / divisor);
    return *this;
  }

  Duration operator+(const Duration& that) const
  {
    Duration sum = *this;
    sum += that;
    return sum;
  }

  Duration operator-(const Duration& that) const
  {
    Duration diff = *this;
    diff -= that;
    return diff;
  }

  Duration operator*(double multiplier) const
  {
    Duration product = *this;
    product *= multiplier;
    return product;
  }

  Duration operator/(double divisor) const
  {
    Duration quotient = *this;
    quotient /= divisor;
    return quotient;
  }

  // A constant holding the maximum value a Duration can have.
  static constexpr Duration max();
  // A constant holding the minimum (negative) value a Duration can
  // have.
  static constexpr Duration min();
  // A constant holding a Duration of a "zero" value.
  static constexpr Duration zero() { return Duration(); }

protected:
  static constexpr int64_t NANOSECONDS  = 1;
  static constexpr int64_t MICROSECONDS = 1000 * NANOSECONDS;
  static constexpr int64_t MILLISECONDS = 1000 * MICROSECONDS;
  static constexpr int64_t SECONDS      = 1000 * MILLISECONDS;
  static constexpr int64_t MINUTES      = 60 * SECONDS;
  static constexpr int64_t HOURS        = 60 * MINUTES;
  static constexpr int64_t DAYS         = 24 * HOURS;
  static constexpr int64_t WEEKS        = 7 * DAYS;

  // Construct from a (value, unit) pair.
  constexpr Duration(int64_t value, int64_t unit)
    : nanos(value * unit) {}

private:
  // Used only by "parse".
  constexpr Duration(double value, int64_t unit)
    : nanos(static_cast<int64_t>(value * unit)) {}

  int64_t nanos;

  friend std::ostream& operator<<(
    std::ostream& stream,
    const Duration& duration);
};


class Nanoseconds : public Duration
{
public:
  explicit constexpr Nanoseconds(int64_t nanoseconds)
    : Duration(nanoseconds, NANOSECONDS) {}

  constexpr Nanoseconds(const Duration& d) : Duration(d) {}

  double value() const { return static_cast<double>(this->ns()); }

  static std::string units() { return "ns"; }
};


class Microseconds : public Duration
{
public:
  explicit constexpr Microseconds(int64_t microseconds)
    : Duration(microseconds, MICROSECONDS) {}

  constexpr Microseconds(const Duration& d) : Duration(d) {}

  double value() const { return this->us(); }

  static std::string units() { return "us"; }
};


class Milliseconds : public Duration
{
public:
  explicit constexpr Milliseconds(int64_t milliseconds)
    : Duration(milliseconds, MILLISECONDS) {}

  constexpr Milliseconds(const Duration& d) : Duration(d) {}

  double value() const { return this->ms(); }

  static std::string units() { return "ms"; }
};


class Seconds : public Duration
{
public:
  explicit constexpr Seconds(int64_t seconds)
    : Duration(seconds, SECONDS) {}

  constexpr Seconds(const Duration& d) : Duration(d) {}

  double value() const { return this->secs(); }

  static std::string units() { return "secs"; }
};


class Minutes : public Duration
{
public:
  explicit constexpr Minutes(int64_t minutes)
    : Duration(minutes, MINUTES) {}

  constexpr Minutes(const Duration& d) : Duration(d) {}

  double value() const { return this->mins(); }

  static std::string units() { return "mins"; }
};


class Hours : public Duration
{
public:
  explicit constexpr Hours(int64_t hours)
    : Duration(hours, HOURS) {}

  constexpr Hours(const Duration& d) : Duration(d) {}

  double value() const { return this->hrs(); }

  static std::string units() { return "hrs"; }
};


class Days : public Duration
{
public:
  explicit constexpr Days(int64_t days)
    : Duration(days, DAYS) {}

  constexpr Days(const Duration& d) : Duration(d) {}

  double value() const { return this->days(); }

  static std::string units() { return "days"; }
};


class Weeks : public Duration
{
public:
  explicit constexpr Weeks(int64_t value) : Duration(value, WEEKS) {}

  constexpr Weeks(const Duration& d) : Duration(d) {}

  double value() const { return this->weeks(); }

  static std::string units() { return "weeks"; }
};


inline std::ostream& operator<<(std::ostream& stream, const Duration& duration_)
{
  // Output the duration in full double precision and save the old precision.
  std::streamsize precision =
    stream.precision(std::numeric_limits<double>::digits10);

  // Parse the duration as the sign and the absolute value.
  Duration duration = duration_;
  if (duration_ < Duration::zero()) {
    stream << "-";

    // Duration::min() may not be representable as a positive Duration.
    if (duration_ == Duration::min()) {
      duration = Duration::max();
    } else {
      duration = duration_ * -1;
    }
  }

  // First determine which bucket of time unit the duration falls into
  // then check whether the duration can be represented as a whole
  // number with this time unit or a smaller one.
  // e.g. 1.42857142857143weeks falls into the 'Weeks' bucket but
  // reads better with a smaller unit: '10days'. So we use 'days'
  // instead of 'weeks' to output the duration.
  int64_t nanoseconds = duration.ns();
  if (duration < Microseconds(1)) {
    stream << duration.ns() << Nanoseconds::units();
  } else if (duration < Milliseconds(1)) {
    if (nanoseconds % Duration::MICROSECONDS != 0) {
      // We can't get a whole number using this unit but we can at
      // one level down.
      stream << duration.ns() << Nanoseconds::units();
    } else {
      stream << duration.us() << Microseconds::units();
    }
  } else if (duration < Seconds(1)) {
    if (nanoseconds % Duration::MILLISECONDS != 0 &&
        nanoseconds % Duration::MICROSECONDS == 0) {
      stream << duration.us() << Microseconds::units();
    } else {
      stream << duration.ms() << Milliseconds::units();
    }
  } else if (duration < Minutes(1)) {
    if (nanoseconds % Duration::SECONDS != 0 &&
        nanoseconds % Duration::MILLISECONDS == 0) {
      stream << duration.ms() << Milliseconds::units();
    } else {
      stream << duration.secs() << Seconds::units();
    }
  } else if (duration < Hours(1)) {
    if (nanoseconds % Duration::MINUTES != 0 &&
        nanoseconds % Duration::SECONDS == 0) {
      stream << duration.secs() << Seconds::units();
    } else {
      stream << duration.mins() << Minutes::units();
    }
  } else if (duration < Days(1)) {
    if (nanoseconds % Duration::HOURS != 0 &&
        nanoseconds % Duration::MINUTES == 0) {
      stream << duration.mins() << Minutes::units();
    } else {
      stream << duration.hrs() << Hours::units();
    }
  } else if (duration < Weeks(1)) {
    if (nanoseconds % Duration::DAYS != 0 &&
        nanoseconds % Duration::HOURS == 0) {
      stream << duration.hrs() << Hours::units();
    } else {
      stream << duration.days() << Days::units();
    }
  } else {
    if (nanoseconds % Duration::WEEKS != 0 &&
        nanoseconds % Duration::DAYS == 0) {
      stream << duration.days() << Days::units();
    } else {
      stream << duration.weeks() << Weeks::units();
    }
  }

  // Return the stream to original formatting state.
  stream.precision(precision);

  return stream;
}


inline Try<Duration> Duration::create(double seconds)
{
  if (seconds * SECONDS > max().nanos || seconds * SECONDS < min().nanos) {
    return Error("Argument out of the range that a Duration can represent due "
                 "to int64_t's size limit");
  }

  return Nanoseconds(static_cast<int64_t>(seconds * SECONDS));
}


inline constexpr Duration Duration::max()
{
  return Nanoseconds(std::numeric_limits<int64_t>::max());
}


inline constexpr Duration Duration::min()
{
  return Nanoseconds(std::numeric_limits<int64_t>::min());
}

class Stopwatch
{
public:
  Stopwatch()
    : running(false)
  {
    started.tv_sec = 0;
    started.tv_nsec = 0;
    stopped.tv_sec = 0;
    stopped.tv_nsec = 0;
  }

  void start()
  {
    started = now();
    running = true;
  }

  void stop()
  {
    stopped = now();
    running = false;
  }

  Nanoseconds elapsed() const
  {
    if (!running) {
      return Nanoseconds(diff(stopped, started));
    }

    return Nanoseconds(diff(now(), started));
  }

private:
  static timespec now()
  {
    timespec ts;
#ifdef __MACH__
    // OS X does not have clock_gettime, use clock_get_time.
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    ts.tv_sec = mts.tv_sec;
    ts.tv_nsec = mts.tv_nsec;
#elif __WINDOWS__
    const static __int64 ticks_in_second = 10000000i64;
    const static __int64 ns_in_tick = 100;

    // Interpret the filetime as a 64-bit unsigned integer to simplify the math
    // of converting to `timespec`.
    static_assert(
        sizeof(FILETIME) == sizeof(__int64),
        "stopwatch: We currently require `FILETIME` to be 64 bits in size");
    __int64 filetime_in_ticks;

    GetSystemTimeAsFileTime(reinterpret_cast<FILETIME*>(&filetime_in_ticks));

    // Conversions. `tv_sec` is elapsed seconds, and `tv_nsec` is the remainder
    // of the elapsed time, in nanoseconds.
    ts.tv_sec  = filetime_in_ticks / ticks_in_second;
    ts.tv_nsec = filetime_in_ticks % ticks_in_second * ns_in_tick;
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif // __MACH__
    return ts;
  }

  static uint64_t diff(const timespec& from, const timespec& to)
  {
    return ((from.tv_sec - to.tv_sec) * 1000000000LL)
      + (from.tv_nsec - to.tv_nsec);
  }

  bool running;
  timespec started, stopped;
};

} // namespace mymesos
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MESOS_STOUT_DURATION_H_