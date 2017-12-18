// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/storage/sstable/hfile/block.h

#ifndef BUBBLEFS_UTILS_TOFT_STORAGE_HFILE_BLOCK_H_
#define BUBBLEFS_UTILS_TOFT_STORAGE_HFILE_BLOCK_H_

#include <string>

#include "utils/toft_base_uncopyable.h"

namespace bubblefs {
namespace mytoft {
class File;

namespace hfile {
class Block {
    DECLARE_UNCOPYABLE(Block);

public:
    Block() {
    }
    virtual ~Block() = 0;

    // Append the block into the file.
    bool WriteToFile(File *fb);

    virtual const std::string EncodeToString() const {
        return std::string();
    }
    virtual bool DecodeFromString(const std::string &str) = 0;
};

}  // namespace hfile
}  // namespace mytoft
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_TOFT_STORAGE_HFILE_BLOCK_H_