// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/compress/block/block_compression_test.cpp

#include <string>

#include "utils/toft_compress_block_block_compression.h"
#include "platform/base_error.h"

#include "gtest/gtest.h"

namespace bubblefs {
namespace mytoft {

static const std::string& test_str = "asdgfsdglzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";
static const std::string& test_empty_str = "";

void TestCompression(const std::string& name, const std::string& test_str) {
    BlockCompression* compression = MYTOFT_CREATE_BLOCK_COMPRESSION(name);
    EXPECT_TRUE(compression != NULL);
    std::string compressed_data;
    bool ret = compression->Compress(test_str.data(),
                                     test_str.size(),
                                     &compressed_data);
    EXPECT_TRUE(ret);
    PRINTF_INFO("raw len: %d, compressed len: %d\n",
                (int)test_str.length(), (int)compressed_data.size());

    std::string uncompressed_data;
    ret = compression->Uncompress(compressed_data.c_str(),
                    compressed_data.size(), &uncompressed_data);
    EXPECT_TRUE(ret);
    EXPECT_EQ(test_str, uncompressed_data);
    delete compression;
}

TEST(CompressionTest, SnappyCompression) {
    TestCompression("snappy", test_str);
    TestCompression("snappy", test_empty_str);
}

TEST(CompressionTest, LzoCompression) {
    TestCompression("lzo", test_str);
    TestCompression("lzo", test_empty_str);
}
}  // namespace mytoft
}  // namespace bubblefs