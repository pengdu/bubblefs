// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>
// Created: 05/11/11

// toft/base/string/byte_set_test.cpp

#include "utils/toft_base_string_byte_set.h"

#include "gtest/gtest.h"

namespace bubblefs {
namespace mytoft {

TEST(ByteSet, Empty)
{
    ByteSet bs;
    EXPECT_FALSE(bs.Find('A'));
    bs.Insert('A');
    EXPECT_TRUE(bs.Find('A'));
}

} // namespace mytoft
} // namespace bubblefs