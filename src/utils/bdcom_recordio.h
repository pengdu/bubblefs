// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/common/file/recordio/record_io.h

#ifndef BUBBLEFS_UTILS_BDCOM_RECORDIO_H_
#define BUBBLEFS_UTILS_BDCOM_RECORDIO_H_

#include <string>

#include "utils/bdcom_scoped_ptr.h"
#include "utils/bdcom_file_stream.h"

#include "google/protobuf/message.h"

namespace bubblefs {
namespace mybdcom {
  
class RecordWriter {
public:
    RecordWriter();
    ~RecordWriter();

    bool Reset(FileStream *file);
    bool WriteMessage(const ::google::protobuf::Message& message);
    bool WriteRecord(const char *data, uint32_t size);
    bool WriteRecord(const std::string& data);

private:
    bool Write(const char *data, uint32_t size);

private:
    FileStream* file_;
};

class RecordReader {
public:
    RecordReader();
    ~RecordReader();

    bool Reset(FileStream *file);

    // for ok, return 1;
    // for no more data, return 0;
    // for error, return -1;
    int Next();

    bool ReadMessage(::google::protobuf::Message *message);
    bool ReadNextMessage(::google::protobuf::Message *message);
    bool ReadRecord(const char **data, uint32_t *size);
    bool ReadRecord(std::string *data);

private:
    bool Read(char *data, uint32_t size);

private:
    FileStream* file_;
    scoped_array<char> buffer_;
    uint32_t file_size_;
    uint32_t buffer_size_;
    uint32_t data_size_;
};

} // namespace mybdcom
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOM_RECORDIO_H_