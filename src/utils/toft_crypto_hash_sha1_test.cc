// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/crypto/hash/sha1_test.cpp

#include "utils/toft_crypto_hash_sha1.h"

#include "gtest/gtest.h"

namespace bubblefs {
namespace mytoft {

TEST(SHA1Test, EmptyString) {
    std::string test_str = "";
    EXPECT_EQ(SHA1::HexDigest(test_str), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
};

TEST(SHA1Test, LittleChanged) {
    std::string test_str = "The quick brown fox jumps over the lazy dog";
    EXPECT_EQ(SHA1::HexDigest(test_str), "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12");

    std::string diff_str = "The quick brown fox jumps over the lazy cog";
    EXPECT_EQ(SHA1::HexDigest(diff_str), "de9f2c7fd25e1b3afad3e85a0bd17d9b100db4b3");
};

} // namespace toft
} // namespace bubblefs