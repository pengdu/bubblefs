// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// // tera/src/common/file/recordio/record_io.cc

#include "utils/bdcom_recordio.h"

#include "platform/base_error.h"

namespace bubblefs {
namespace mybdcom {

RecordWriter::RecordWriter(): file_(NULL) {}

RecordWriter::~RecordWriter() {}

bool RecordWriter::Reset(FileStream *file) {
    PANIC_ENFORCE(file != NULL, "file is NULL");
    file_ = file;
    return true;
}

bool RecordWriter::WriteMessage(const ::google::protobuf::Message& message) {
    std::string output;
    if (!message.IsInitialized()) {
        PRINTF_WARN("Missing required fields. %s\n",
                    message.InitializationErrorString().c_str());
        return false;
    }
    if (!message.AppendToString(&output)) {
        return false;
    }
    if (!WriteRecord(output.data(), output.size())) {
        return false;
    }
    return true;
}

bool RecordWriter::WriteRecord(const char *data, uint32_t size) {
    if (!Write(reinterpret_cast<char*>(&size), sizeof(size))) {
        return false;
    }
    if (!Write(data, size)) {
        return false;
    }
    return true;
}

bool RecordWriter::WriteRecord(const std::string& data) {
    return WriteRecord(data.data(), data.size());
}

bool RecordWriter::Write(const char *data, uint32_t size) {
    uint32_t write_size = 0;
    while (write_size < size) {
        int32_t ret = file_->Write(data + write_size, size - write_size);
        if (ret == -1) {
            PRINTF_ERROR("RecordWriter error.\n");
            return false;
        }
        write_size += ret;
    }
    file_->Flush();

    return true;
}


RecordReader::RecordReader()
    : file_(NULL),
      file_size_(0),
      buffer_size_(1 * 1024 * 1024),
      data_size_(0) {
    buffer_.reset(new char[buffer_size_]);
}

RecordReader::~RecordReader() {}

bool RecordReader::Reset(FileStream *file) {
    PANIC_ENFORCE(file != NULL, "file is NULL");
    file_ = file;
    if (-1 == file_->Seek(0, SEEK_END)) {
        PRINTF_ERROR("RecordReader Reset error.\n");
        return false;
    }
    file_size_ = file_->Tell();
    if (-1 == file_->Seek(0, SEEK_SET)) {
        PRINTF_ERROR("RecordReader Reset error.\n");
        return false;
    }
    return true;
}

int RecordReader::Next() {
    // read size
    int64_t ret = file_->Tell();
    if (ret == -1) {
        PRINTF_ERROR("Tell error.\n");
        return -1;
    }

    if (ret == file_size_) {
        return 0;
    } else if (file_size_ - ret >= static_cast<int64_t>(sizeof(data_size_))) { // NO_LINT
        if (!Read(reinterpret_cast<char*>(&data_size_), sizeof(data_size_))) {
            PRINTF_ERROR("Read size error.\n");
            return -1;
        }
    }

    // read data
    ret = file_->Tell();
    if (ret == -1) {
        PRINTF_ERROR("Tell error.\n");
        return -1;
    }

    if (ret >= file_size_ && data_size_ != 0) {
        PRINTF_ERROR("read error.\n");
        return -1;
    } else if (file_size_ - ret >= data_size_) { // NO_LINT
        if (data_size_ > buffer_size_) {
            while (data_size_ > buffer_size_) {
                buffer_size_ *= 2;
            }
            buffer_.reset(new char[buffer_size_]);
        }

        if (!Read(buffer_.get(), data_size_)) {
            PRINTF_ERROR("Read data error.\n");
            return -1;
        }
    } else {
        PRINTF_ERROR("data_size_ of current record is invalid: %u bigger than %ld\n",
                     data_size_, (file_size_ - ret));
        return -1;
    }

    return 1;
}

bool RecordReader::ReadMessage(::google::protobuf::Message *message) {
    std::string str(buffer_.get(), data_size_);
    if (!message->ParseFromArray(buffer_.get(), data_size_)) {
        PRINTF_WARN("Missing required fields.\n");
        return false;
    }
    return true;
}

bool RecordReader::ReadNextMessage(::google::protobuf::Message *message) {
    while (Next() == 1) {
        std::string str(buffer_.get(), data_size_);
        if (message->ParseFromArray(buffer_.get(), data_size_)) {
            return true;
        }
    }
    return false;
}

bool RecordReader::ReadRecord(const char **data, uint32_t *size) {
    *data = buffer_.get();
    *size = data_size_;
    return true;
}

bool RecordReader::ReadRecord(std::string *data) {
    data->assign(buffer_.get());
    return true;
}

bool RecordReader::Read(char *data, uint32_t size) {
    // Read
    uint32_t read_size = 0;
    while (read_size < size) {
        int64_t ret = file_->Read(data + read_size, size - read_size);
        if (ret == -1) {
            PRINTF_ERROR("Read error.\n");
            return false;
        }
        read_size += ret;
    }

    return true;
}

} // namespace mybdcom
} // namespace bubblefs