// Copyright (C) 2013, The Toft Authors.
// Author: An Qin <anqin.qin@gmail.com>
//
// Description:

#ifndef TOFT_STORAGE_RECORDIO_RECORDIO_H
#define TOFT_STORAGE_RECORDIO_RECORDIO_H

#include <string>

#include "utils/toft_base_scoped_array.h"
#include "utils/toft_base_string_string_piece.h"
#include "utils/toft_storage_file_file.h"

#include "google/protobuf/message.h"

namespace bubblefs {
namespace mytoft {

class RecordWriter {
public:
    explicit RecordWriter(File *file);
    ~RecordWriter();

    bool WriteMessage(const ::google::protobuf::Message& message);
    bool WriteRecord(const char *data, uint32_t size);
    bool WriteRecord(const std::string& data);
    bool WriteRecord(const StringPiece& data);

private:
    bool Write(const char *data, uint32_t size);

private:
    File* m_file;
};

class RecordReader {
public:
    explicit RecordReader(File *file);
    ~RecordReader();

    bool Reset();

    // for ok, return 1;
    // for no more data, return 0;
    // for error, return -1;
    int Next();

    bool ReadMessage(::google::protobuf::Message *message);
    bool ReadNextMessage(::google::protobuf::Message *message);
    bool ReadRecord(const char **data, uint32_t *size);
    bool ReadRecord(std::string *data);
    bool ReadRecord(StringPiece* data);

private:
    bool Read(char *data, uint32_t size);

private:
    File* m_file;
    scoped_array<char> m_buffer;
    uint32_t m_file_size;
    uint32_t m_buffer_size;
    uint32_t m_data_size;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_TOFT_STORAGE_RECORDIO_RECORDIO_H_