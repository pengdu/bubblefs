// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/compress/block/snappy.h

#ifndef BUBBLEFS_UTILS_TOFT_COMPRESS_BLOCK_SNAPPY_H_
#define BUBBLEFS_UTILS_TOFT_COMPRESS_BLOCK_SNAPPY_H_

#include <string>

#include "utils/toft_compress_block_block_compression.h"

namespace bubblefs {
namespace mytoft {

class SnappyCompression : public BlockCompression {
    DECLARE_UNCOPYABLE(SnappyCompression);

public:
    SnappyCompression();
    virtual ~SnappyCompression();

    virtual std::string GetName() {
        return "snappy";
    }

private:
    virtual bool DoCompress(const char* str, size_t length, std::string* out);
    virtual bool DoUncompress(const char* str, size_t length, std::string* out);
};

}  // namespace mytoft
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_TOFT_COMPRESS_BLOCK_SNAPPY_H_