// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>
// Created: 12/14/11
// Description: test for preprocess.h

// toft/base/preprocess_test.cpp

#include "utils/toft_base_preprocess.h"

#include "gtest/gtest.h"

TEST(Preprocess, Stringize)
{
    EXPECT_STREQ("ABC", MYTOFT_PP_STRINGIZE(ABC));
}

TEST(Preprocess, Join)
{
    EXPECT_EQ(12, MYTOFT_PP_JOIN(1, 2));
}

TEST(Preprocess, DisallowInHeader)
{
    MYTOFT_PP_DISALLOW_IN_HEADER_FILE();
}

TEST(Preprocess, VaNargs)
{
    EXPECT_EQ(0, MYTOFT_PP_N_ARGS());
    EXPECT_EQ(1, MYTOFT_PP_N_ARGS(a));
    EXPECT_EQ(2, MYTOFT_PP_N_ARGS(a, b));
    EXPECT_EQ(3, MYTOFT_PP_N_ARGS(a, b, c));
    EXPECT_EQ(4, MYTOFT_PP_N_ARGS(a, b, c, d));
    EXPECT_EQ(5, MYTOFT_PP_N_ARGS(a, b, c, d, e));
    EXPECT_EQ(6, MYTOFT_PP_N_ARGS(a, b, c, d, e, f));
    EXPECT_EQ(7, MYTOFT_PP_N_ARGS(a, b, c, d, e, f, g));
    EXPECT_EQ(8, MYTOFT_PP_N_ARGS(a, b, c, d, e, f, g, h));
    EXPECT_EQ(9, MYTOFT_PP_N_ARGS(a, b, c, d, e, f, g, h, i));
    EXPECT_EQ(10, MYTOFT_PP_N_ARGS(a, b, c, d, e, f, g, h, i, j));
    EXPECT_EQ(11, MYTOFT_PP_N_ARGS(a, b, c, d, e, f, g, h, i, j, k));
    EXPECT_EQ(12, MYTOFT_PP_N_ARGS(a, b, c, d, e, f, g, h, i, j, k, l));
    EXPECT_EQ(13, MYTOFT_PP_N_ARGS(a, b, c, d, e, f, g, h, i, j, k, l, m));
    EXPECT_EQ(14, MYTOFT_PP_N_ARGS(a, b, c, d, e, f, g, h, i, j, k, l, m, n));
    EXPECT_EQ(15, MYTOFT_PP_N_ARGS(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o));
}

TEST(Preprocess, Varargs)
{
    EXPECT_EQ("a", MYTOFT_PP_FOR_EACH_ARGS(MYTOFT_PP_STRINGIZE, a));
    EXPECT_EQ("ab", MYTOFT_PP_FOR_EACH_ARGS(MYTOFT_PP_STRINGIZE, a, b));
    EXPECT_EQ("abc", MYTOFT_PP_FOR_EACH_ARGS(MYTOFT_PP_STRINGIZE, a, b, c));
    EXPECT_EQ("abcd", MYTOFT_PP_FOR_EACH_ARGS(MYTOFT_PP_STRINGIZE, a, b, c, d));
    EXPECT_EQ("abcde",
              MYTOFT_PP_FOR_EACH_ARGS(MYTOFT_PP_STRINGIZE, a, b, c, d, e));
    EXPECT_EQ("abcdef",
              MYTOFT_PP_FOR_EACH_ARGS(MYTOFT_PP_STRINGIZE, a, b, c, d, e, f));
    EXPECT_EQ("abcdefg",
              MYTOFT_PP_FOR_EACH_ARGS(MYTOFT_PP_STRINGIZE, a, b, c, d, e, f, g));
    EXPECT_EQ("abcdefgh",
              MYTOFT_PP_FOR_EACH_ARGS(
                  MYTOFT_PP_STRINGIZE, a, b, c, d, e, f, g, h));
    EXPECT_EQ("abcdefghi",
              MYTOFT_PP_FOR_EACH_ARGS(
                  MYTOFT_PP_STRINGIZE, a, b, c, d, e, f, g, h, i));
    EXPECT_EQ("abcdefghij",
              MYTOFT_PP_FOR_EACH_ARGS(
                  MYTOFT_PP_STRINGIZE, a, b, c, d, e, f, g, h, i, j));
    EXPECT_EQ("abcdefghijk",
              MYTOFT_PP_FOR_EACH_ARGS(MYTOFT_PP_STRINGIZE,
                  a, b, c, d, e, f, g, h, i, j, k));
    EXPECT_EQ("abcdefghijkl",
              MYTOFT_PP_FOR_EACH_ARGS(
                  MYTOFT_PP_STRINGIZE, a, b, c, d, e, f, g, h, i, j, k, l));
    EXPECT_EQ("abcdefghijklm",
              MYTOFT_PP_FOR_EACH_ARGS(
                  MYTOFT_PP_STRINGIZE, a, b, c, d, e, f, g, h, i, j, k, l, m));
    EXPECT_EQ("abcdefghijklmn",
              MYTOFT_PP_FOR_EACH_ARGS(
                  MYTOFT_PP_STRINGIZE,
                  a, b, c, d, e, f, g, h, i, j, k, l, m, n));
    EXPECT_EQ("abcdefghijklmno",
              MYTOFT_PP_FOR_EACH_ARGS(
                  MYTOFT_PP_STRINGIZE,
                  a, b, c, d, e, f, g, h, i, j, k, l, m, n, o));
}

#define DEFINE_METHOD(cmd, name) (cmd, name)

#define EXPAND_METHOD_(cmd, name) int name() { return cmd; }
#define EXPAND_METHOD(x) EXPAND_METHOD_ x

#define DEFINE_SERVICE(name, ...) \
    class name { \
    public: \
        MYTOFT_PP_FOR_EACH_ARGS(EXPAND_METHOD, __VA_ARGS__) \
    };

DEFINE_SERVICE(TestService,
    DEFINE_METHOD(1, Echo),
    DEFINE_METHOD(2, Inc)
)

TEST(Preprocess, VaForEach)
{
    TestService service;
    EXPECT_EQ(1, service.Echo());
    EXPECT_EQ(2, service.Inc());
}