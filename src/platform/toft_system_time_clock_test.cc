// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/time/clock_test.cpp

#include "platform/toft_system_time_clock.h"

#include "gtest/gtest.h"

namespace bubblefs {
namespace mytoft {

TEST(RealtimeClock, Test)
{
    int64_t us = RealtimeClock.MicroSeconds();
    ASSERT_GT(us, 0);
}

} // namespace mytoft
} // namespace bubblefs