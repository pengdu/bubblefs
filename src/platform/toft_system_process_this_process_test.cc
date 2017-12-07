// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Chen Feng <chen3feng@gmail.com>

// toft/system/process/this_process_test.cpp

#include "platform/toft_system_process_this_process.h"
#include "platform/toft_system_threading_this_thread.h"
#include "utils/toft_base_string_algorithm.h"

#include "gtest/gtest.h"

namespace bubblefs {
namespace mytoft {

TEST(ThisProcess, BinaryPath)
{
    EXPECT_TRUE(StringEndsWith(ThisProcess::BinaryPath(), "/this_process_test"));
}

TEST(ThisProcess, BinaryName)
{
    EXPECT_EQ("this_process_test", ThisProcess::BinaryName());
}

TEST(ThisProcess, BinaryDirectory)
{
    EXPECT_TRUE(StringEndsWith(ThisProcess::BinaryDirectory(), "/process"));
}

TEST(ThisProcess, StartTime)
{
    time_t t = ThisProcess::StartTime();
    EXPECT_GT(t, 0);
    EXPECT_LE(t, time(NULL));
    ThisThread::Sleep(2000);
    ASSERT_EQ(t, ThisProcess::StartTime());
}

TEST(ThisProcess, ElapsedTime)
{
    time_t t = ThisProcess::ElapsedTime();
    EXPECT_GE(t, 0);
    EXPECT_LT(t, 100);
    ThisThread::Sleep(2000);
    EXPECT_GT(ThisProcess::ElapsedTime() - t, 1);
}

} // namespace mytoft
} // namespace bubblefs