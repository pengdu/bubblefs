// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/storage/sstable/hfile/data_block.cpp

#include "utils/toft_storage_hfile_data_block.h"

#include "utils/toft_base_string_algorithm.h"
#include "utils/toft_compress_block_block_compression.h"
#include "utils/toft_storage_hfile_coding.h"

//  GLOBAL_NOLINT(runtime/sizeof)

namespace {
static const std::string kDataBlockMagic = "DATABLK\42";
}

namespace bubblefs {
namespace mytoft {
namespace hfile {

DataBlock::~DataBlock() {
}

DataBlock::DataBlock(CompressType codec)
                : compression_(NULL),
                  compressed_size_(0) {
    switch (codec) {
    case CompressType_kSnappy:
        compression_ = MYTOFT_CREATE_BLOCK_COMPRESSION("snappy");
        break;
    case CompressType_kUnCompress:
        break;
    default:
        PANIC("not supported yet!\n");
    }
}

const std::string DataBlock::EncodeToString() const {
    if (compression_ != NULL) {
        std::string compressed;
        if (!compression_->Compress(buffer_.c_str(), buffer_.size(), &compressed)) {
            PRINTF_ERROR("compress failed!\n");
            return "";
        }
        // save the compressed info
        compressed_size_ = compressed.size();
        return compressed;
    }
    compressed_size_ = buffer_.size();
    return buffer_;
}

bool DataBlock::DecodeFromString(const std::string &str) {
    if (compression_ != NULL) {
        std::string uncompressed;
        if (!compression_->Uncompress(str.c_str(), str.size(), &uncompressed)) {
            PRINTF_ERROR("uncompress failed!\n");
            return false;
        }
        return DecodeInternal(uncompressed);
    }
    return DecodeInternal(str);
}

bool DataBlock::DecodeInternal(const std::string &str) {
    if (!StringStartsWith(str, kDataBlockMagic)) {
        //LOG(INFO)<< "invalid data block header.";
        return false;
    }
    data_items_.clear();
    const char *begin = str.c_str() + kDataBlockMagic.size();
    const char *end = str.c_str() + str.length();
    while (begin < end) {
        int key_length = ReadInt32(&begin);
        int value_length = ReadInt32(&begin);
        std::string key = std::string(begin, key_length);
        begin += key_length;
        std::string value = std::string(begin, value_length);
        begin += value_length;
        data_items_.push_back(make_pair(key, value));
    }
    if (begin > end) {
        PRINTF_ERROR("not a complete data block\n");
        return false;
    }
    return true;
}

void DataBlock::AddItem(const std::string &key, const std::string &value) {
    // ignore totally empty item
    if (key.empty() && value.empty())
        return;

    if (buffer_.empty())
        buffer_ = kDataBlockMagic;

    PutFixed32(&buffer_, key.size());
    PutFixed32(&buffer_, value.size());
    buffer_ += key;
    buffer_ += value;
}

}  // namespace hfile
}  // namespace mytoft
}  // namespace bubblefs