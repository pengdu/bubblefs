// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/storage/sstable/hfile/block.cpp

#include "utils/toft_storage_hfile_block.h"

#include "utils/toft_storage_file_file.h"

#include "platform/base_error.h"

namespace bubblefs {
namespace mytoft {
namespace hfile {

Block::~Block() {}

bool Block::WriteToFile(File *fb) {
    const std::string &str = EncodeToString();
    if (str.empty()) {
        return true;
    }
    int64_t s = fb->Write(str.c_str(), str.length());
    if (s != static_cast<int64_t>(str.length()))
        PRINTF_WARN("fail to write buffer\n");
    return true;
}

}  // namespace hfile
}  // namespace toft
}  // namespace bubblefs