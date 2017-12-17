// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

// mars/mars/comm/time_utils.c
// mars/mars/comm/tickcount.cc

#include "platform/mars_time_utils.h"
#include <sys/time.h>
#include <stdint.h>
#include <time.h>

namespace bubblefs {
namespace mymars {

uint64_t gettickcount() {//todoyy
    struct timespec ts;
    if (0==clock_gettime(CLOCK_MONOTONIC, &ts)){
        return (ts.tv_sec * 1000ULL + ts.tv_nsec / 1000000);
    }
    return 0;
}

int64_t gettickspan(uint64_t _old_tick) {
    uint64_t cur_tick = gettickcount();
    if (_old_tick > cur_tick) return 0;

    return cur_tick - _old_tick;
}

uint64_t timeMs() {
   struct timeval tv;
   gettimeofday(&tv,NULL);
   return (uint64_t)tv.tv_sec * 1000 + (uint64_t)tv.tv_usec / 1000;
}

static uint64_t sg_tick_start = gettickcount();
static const uint64_t sg_tick_init = 2000000000;

tickcount_t::tickcount_t(bool _now)
  :tickcount_(0)
{
    if (_now) gettickcount();
}

tickcount_t& tickcount_t::gettickcount()
{
    tickcount_ = sg_tick_init + mymars::gettickcount() - sg_tick_start;
    return *this;
}

} // namespace mymars
} // namespace bubblefs