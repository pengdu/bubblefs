// Modifications copyright (C) 2017, Baidu.com, Inc.
// Copyright 2017 The Apache Software Foundation

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// palo/be/src/util/stopwatch.hpp

#ifndef BUBBLEFS_UTILS_STOPWATCH_H_
#define BUBBLEFS_UTILS_STOPWATCH_H_

#include <time.h>
#include <stdint.h>

namespace bubblefs {
namespace timer {
// Utility class to measure time.  This is measured using the cpu tick counter which
// is very low overhead but can be inaccurate if the thread is switched away.  This
// is useful for measuring cpu time at the row batch level (too much overhead at the
// row granularity).
class StopWatch {
public:
    StopWatch() {
        _total_time = 0;
        _running = false;
    }

    void start() {
        if (!_running) {
            _start = rdtsc();
            _running = true;
        }
    }

    void stop() {
        if (_running) {
            _total_time += rdtsc() - _start;
            _running = false;
        }
    }

    // Returns time in cpu ticks.
    uint64_t elapsed_time() const {
        return _running ? rdtsc() - _start : _total_time;
    }

    static uint64_t rdtsc() {
        uint32_t lo, hi;
        __asm__ __volatile__(
            "xorl %%eax,%%eax \n        cpuid"
            ::: "%rax", "%rbx", "%rcx", "%rdx");
        __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
        return (uint64_t)hi << 32 | lo;
    }

private:
    uint64_t _start, _total_time;
    bool _running;
};

} // namespace timer
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_STOPWATCH_H_
