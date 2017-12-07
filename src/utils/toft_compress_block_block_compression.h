// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/compress/block/block_compression.h

#ifndef BUBBLEFS_UTILS_TOFT_COMPRESS_BLOCK_BLOCK_COMPRESSION_H_
#define BUBBLEFS_UTILS_TOFT_COMPRESS_BLOCK_BLOCK_COMPRESSION_H_

#include <string>

#include "utils/toft_base_class_registry_class_registry.h"
#include "utils/toft_base_string_string_piece.h"
#include "utils/toft_base_uncopyable.h"

namespace bubblefs {
namespace mytoft {

class BlockCompression {
    DECLARE_UNCOPYABLE(BlockCompression);

public:
    BlockCompression();
    virtual ~BlockCompression();

    bool Compress(const char* str, size_t length, std::string* out);
    bool Compress(StringPiece sp, std::string* out);
    bool Uncompress(const char* str, size_t length, std::string* out);
    bool Uncompress(StringPiece sp, std::string* out);
    virtual std::string GetName() = 0;
    void SetMaxUnCompressedSize(size_t s) {
        max_unCompressed_size_ = s;
    }

protected:
    size_t max_unCompressed_size_;

private:
    virtual bool DoCompress(const char* str, size_t length, std::string* out) = 0;
    virtual bool DoUncompress(const char* str, size_t length, std::string* out) = 0;
};

MYTOFT_CLASS_REGISTRY_DEFINE(block_compression_registry, BlockCompression);

#define MYTOFT_REGISTER_BLOCK_COMPRESSION(class_name, algorithm_name) \
    MYTOFT_CLASS_REGISTRY_REGISTER_CLASS( \
        ::bubblefs::mytoft::block_compression_registry, \
        ::bubblefs::mytoft::BlockCompression, \
        algorithm_name, \
        class_name)

#define MYTOFT_CREATE_BLOCK_COMPRESSION(name) \
    MYTOFT_CLASS_REGISTRY_CREATE_OBJECT(block_compression_registry, name)

}  // namespace mytoft
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_TOFT_COMPRESS_BLOCK_BLOCK_COMPRESSION_H_