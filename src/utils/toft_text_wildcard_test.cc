// Copyright (c) 2010, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/text/wildcard_test.cpp

#include "utils/toft_text_wildcard.h"

#include "gtest/gtest.h"

namespace bubblefs {
namespace mytoft {

TEST(Wildcard, Default)
{
    EXPECT_TRUE(Wildcard::Match("*.c", "a.c"));
    EXPECT_FALSE(Wildcard::Match("*.c", "a.cpp"));
    EXPECT_TRUE(Wildcard::Match("*.cpp", "a.cpp"));
    EXPECT_TRUE(Wildcard::Match("*_test.cpp", "_test.cpp"));
    EXPECT_TRUE(Wildcard::Match("lib*.a", "libc.a"));
    EXPECT_TRUE(Wildcard::Match("lib?.a", "libm.a"));
    EXPECT_FALSE(Wildcard::Match("lib?.a", "librt.a"));
}

TEST(Wildcard, FileNameOnly)
{
    EXPECT_TRUE(Wildcard::Match("*.c", "a.c", Wildcard::MATCH_FILE_NAME_ONLY));
    EXPECT_FALSE(Wildcard::Match("*.c", "dir/a.c", Wildcard::MATCH_FILE_NAME_ONLY));
    EXPECT_TRUE(Wildcard::Match("*.c", "dir/a.c"));
}

TEST(Wildcard, IgnoreCase)
{
    EXPECT_TRUE(Wildcard::Match("*.c", "a.c", Wildcard::IGNORE_CASE));
    EXPECT_TRUE(Wildcard::Match("*.c", "a.C", Wildcard::IGNORE_CASE));
    EXPECT_FALSE(Wildcard::Match("*.c", "a.d", Wildcard::IGNORE_CASE));
}

} // namespace mytoft
} // namespace bubblefs