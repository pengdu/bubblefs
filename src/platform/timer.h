// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
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
//------------------------------------------------------------------------------
// Copyright (c) 2016 by contributors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

// brpc/src/butil/time.h
// caffe2/caffe2/core/timer.h
// xlearn/src/base/timer.h

#ifndef BUBBLEFS_PLATFORM_TIMER_H_
#define BUBBLEFS_PLATFORM_TIMER_H_

#include <chrono>
#include <functional>
#include <list>
#include <memory>
#include <ratio>
#include <sstream>
#include <string>
#include "platform/base_error.h"
#include "platform/time.h"
#include "utils/peloton_printable.h"

namespace bubblefs {
namespace timeutil {

// ---------------
//  Count elapses
// ---------------
class SimpleTimer {
public:

    enum TimerType {
        STARTED,
    };

    SimpleTimer() : _stop(0), _start(0) {}
    explicit SimpleTimer(const TimerType) {
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

/**
 * @brief A simple timer object for measuring time.
 *
 * This is a minimal class around a std::chrono::high_resolution_clock that
 * serves as a utility class for testing code.
 */
template<typename ResolutionRatio = std::ratio<1> >
class ChronoTimer : public mypeloton::Printable {
 public:
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::nanoseconds ns;
  typedef std::chrono::time_point<clock> time_point;
  ChronoTimer() : elapsed_(0), invocations_(0) { Start(); }
  /**
   * @brief Starts a timer.
   */
  inline void Start() { begin_ = clock::now(); }
  inline double NanoSeconds() {
    return std::chrono::duration_cast<ns>(clock::now() - begin_).count();
  }
  /**
   * @brief Returns the elapsed time in milliseconds.
   */
  inline double MilliSeconds() { return NanoSeconds() / 1000000.f; }
  /**
   * @brief Returns the elapsed time in microseconds.
   */
  inline double MicroSeconds() { return NanoSeconds() / 1000.f; }
  /**
   * @brief Returns the elapsed time in seconds.
   */
  inline double Seconds() { return NanoSeconds() / 1000000000.f; }
  
  inline void Stop() {
    end_ = clock::now();

    double duration =
        std::chrono::duration_cast<
            std::chrono::duration<double, ResolutionRatio> >(end_ - begin_)
            .count();

    elapsed_ += duration;
    ++invocations_;
  }
  
  inline void Reset() { elapsed_ = 0; }
  
  // Get Elapsed duration
  inline double GetDuration() const { return elapsed_; }

  // Get Number of invocations
  inline int GetInvocations() const { return invocations_; }

  // Get a string representation for debugging
  inline const std::string GetInfo() const {
    std::ostringstream os;
    os << "ChronoTimer["
       << "elapsed=" << elapsed_ << ", "
       << "invocations=" << invocations_ << "]";
    return (os.str());
  }

 protected:
  time_point begin_;
  time_point end_;
  // Elapsed time (with desired resolution)
  double elapsed_;
  // Number of invocations
  int invocations_;
  
  DISALLOW_COPY_AND_ASSIGN(ChronoTimer);
};

//------------------------------------------------------------------------------
// We can use the Timer class like this:
//
//   Timer timer();
//   timer.tic();
//
//     .... /* code we want to evaluate */
//
//   float time = timer.toc();  // (sec)
//
// This class can be used to evaluate multi-thread code.
//------------------------------------------------------------------------------
class MyxlearnTimer {
 public:
    MyxlearnTimer();
    // Reset start time
    void reset();
    // Code start
    void tic();
    // Code end
    float toc();
    // Get the time duration
    float get();

 protected:
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::milliseconds duration;

 private:
  DISALLOW_COPY_AND_ASSIGN(MyxlearnTimer);
};
  
} // namespace timeutil
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TIMER_H_