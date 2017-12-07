// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/compress/block/snappy.cpp

#include "utils/toft_compress_block_snappy.h"

#include "snappy.h"

namespace bubblefs {
namespace mytoft {
  
SnappyCompression::SnappyCompression() {}

SnappyCompression::~SnappyCompression() {}

bool SnappyCompression::DoCompress(const char* str, size_t length, std::string* out) {
    snappy::Compress(str, length, out);
    return true;
}

bool SnappyCompression::DoUncompress(const char* str, size_t length, std::string* out) {
    return snappy::Uncompress(str, length, out);
}

MYTOFT_REGISTER_BLOCK_COMPRESSION(SnappyCompression, "snappy");

}  // namespace mytoft
}  // namespace bubblefs