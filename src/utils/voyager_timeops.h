// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/util/timeops.h

#ifndef BUBBLEFS_UTILS_VOYAGER_TIMEOPS_H_
#define BUBBLEFS_UTILS_VOYAGER_TIMEOPS_H_

#include <stdint.h>
#include <string>

namespace bubblefs {
namespace myvoyager {
namespace timeops {

static const uint64_t kSecondsPerMinute = 60;
static const uint64_t kSecondsPerHour = 3600;
static const uint64_t kSecondsPerDay = kSecondsPerHour * 24;
static const uint64_t kMilliSecondsPerSecond = 1000;
static const uint64_t kMicroSecondsPerSecond = 1000 * 1000;
static const uint64_t kNonasSecondsPerSecond = 1000 * 1000 * 1000;

extern uint64_t NowMicros();

extern std::string FormatTimestamp(uint64_t micros);

}  // namespace timeops
}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_TIMEOPS_H_