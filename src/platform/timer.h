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

// Pebble/src/common/time_utility.h
// baidu/common/timer.h
// brpc/src/butil/time.h
// caffe2/caffe2/core/timer.h

#ifndef BUBBLEFS_PLATFORM_TIMER_H_
#define BUBBLEFS_PLATFORM_TIMER_H_

#include <functional>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include "platform/base_error.h"
#include "platform/time.h"

namespace bubblefs {
namespace timeutil {
  
class TimeUtility {
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

/**
 * @brief A simple timer object for measuring time.
 *
 * This is a minimal class around a std::chrono::high_resolution_clock that
 * serves as a utility class for testing code.
 */
class ChronoTimer {
 public:
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::nanoseconds ns;
  ChronoTimer() { Start(); }
  /**
   * @brief Starts a timer.
   */
  inline void Start() { start_time_ = clock::now(); }
  inline float NanoSeconds() {
    return std::chrono::duration_cast<ns>(clock::now() - start_time_).count();
  }
  /**
   * @brief Returns the elapsed time in milliseconds.
   */
  inline float MilliSeconds() { return NanoSeconds() / 1000000.f; }
  /**
   * @brief Returns the elapsed time in microseconds.
   */
  inline float MicroSeconds() { return NanoSeconds() / 1000.f; }
  /**
   * @brief Returns the elapsed time in seconds.
   */
  inline float Seconds() { return NanoSeconds() / 1000000000.f; }

 protected:
  std::chrono::time_point<clock> start_time_;
  DISALLOW_COPY_AND_ASSIGN(ChronoTimer);
};

/// @brief Timer模块错误码定义
typedef enum {
    kTIMER_ERROR_BASE       = TIMER_ERROR_CODE_BASE,
    kTIMER_INVALID_PARAM    = kTIMER_ERROR_BASE - 1, // 参数错误
    kTIMER_NUM_OUT_OF_RANGE = kTIMER_ERROR_BASE - 2, // 定时器数量超出限制范围
    kTIMER_UNEXISTED        = kTIMER_ERROR_BASE - 3, // 定时器不存在
    kSYSTEM_ERROR           = kTIMER_ERROR_BASE - 4, // 系统错误
} TimerErrorCode;

/// @brief Timer超时回调函数返回码定义
typedef enum {
    kTIMER_BE_REMOVED   = -1,   // 超时后timer被停止并remove
    kTIMER_BE_CONTINUED = 0,    // 超时后timer仍然保持，重新计时
    kTIMER_BE_RESETED           // 超时后timer仍然保持，使用返回值作为超时时间(ms)重新计时
} OnTimerCallbackReturnCode;

/// @brief 定时器超时回调函数定义
/// @return kTIMER_BE_REMOVED 停止并删除定时器
/// @return kTIMER_BE_CONTINUED 重启定时器
/// @return >0 使用返回值作为新的超时时间(ms)重启定时器
/// @note 超时回调中不能有阻塞操作
/// @see OnTimerCallbackReturnCode
typedef std::function<int32_t()> TimeoutCallback;

/// @brief 定时器接口
class ITimer {
public:
    virtual ~ITimer() {}

    /// @brief 启动定时器
    /// @param timeout_ms 超时时间(>0)，单位为毫秒
    /// @param cb 定时器超时回调函数
    /// @return >=0 定时器ID
    /// @return <0 创建失败 @see TimerErrorCode
    virtual int64_t StartTimer(uint32_t timeout_ms, const TimeoutCallback& cb) = 0;

    /// @brief 停止定时器
    /// @param timer_id StartTimer时返回的ID
    /// @return 0 成功
    /// @return <0 失败 @see TimerErrorCode
    virtual int32_t StopTimer(int64_t timer_id) = 0;

    /// @brief 定时器驱动
    /// @return 超时定时器数，为0时表示本轮无定时器超时
    virtual int32_t Update() = 0;

    /// @brief 返回最后一次的错误信息描述
    virtual const char* GetLastError() const { return NULL; }

    /// @brief 获取定时器数目
    virtual int64_t GetTimerNum() { return 0; }
};

/// @brief 顺序定时器，按超时时间组织，每个超时时间维护一个列表，先加入先超时
///     适合一组离散的单次超时处理，如RPC的请求、协程的超时等
///     复杂度:start O(1gn)，timeout O(1)，stop O(1gn)
class SequenceTimer : public ITimer {
public:
    SequenceTimer();
    virtual ~SequenceTimer();

    /// @see Timer::StartTimer
    virtual int64_t StartTimer(uint32_t timeout_ms, const TimeoutCallback& cb);

    /// @see Timer::StopTimer
    virtual int32_t StopTimer(int64_t timer_id);

    /// @see Timer::Update
    virtual int32_t Update();

    /// @see Timer::LastErrorStr
    virtual const char* GetLastError() const {
        return m_last_error;
    }

    /// @see Timer::GetTimerNum
    virtual int64_t GetTimerNum() {
        return m_id_2_timer.size();
    }

private:
    struct TimerItem {
        TimerItem() {
            stoped  = false;
            id      = -1;
            timeout = 0;
        }

        bool    stoped;
        int64_t id;
        int64_t timeout;
        TimeoutCallback cb;
    };

private:
    int64_t m_timer_seqid;
    // map<timeout_ms, list<TimerItem> >
    std::unordered_map<uint32_t, std::list<std::shared_ptr<TimerItem> > > m_timers;
    // map<timer_seqid, TimerItem>
    std::unordered_map<int64_t, std::shared_ptr<TimerItem> > m_id_2_timer;
    char m_last_error[256];
};
  
} // namespace timeutil
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TIMER_H_