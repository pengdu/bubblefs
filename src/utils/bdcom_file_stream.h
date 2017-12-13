// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/common/file/file_stream.h

#ifndef BUBBLEFS_UTILS_BDCOM_FILE_STREAM_H_
#define BUBBLEFS_UTILS_BDCOM_FILE_STREAM_H_

#include <string>
#include <fstream>

#include "utils/bdcom_file_types.h"

namespace bubblefs {
namespace mybdcom {
  
class FileStream {
public:
    FileStream();
    ~FileStream() {}

    bool Open(const std::string& file_path,
              FileOpenMode flag,
              FileErrorCode* error_code = NULL);
    bool Close(FileErrorCode* error_code = NULL);

    int64_t Write(const void* buffer, int64_t buffer_size,
                  FileErrorCode* error_code = NULL);

    int64_t Read(void* buffer, int64_t buffer_size,
                 FileErrorCode* error_code = NULL);

    bool Flush();

    int64_t Seek(int64_t offset, int32_t origin,
              FileErrorCode* error_code = NULL);

    int64_t Tell(FileErrorCode* error_code = NULL);

    int64_t GetSize(const std::string& file_path,
                    FileErrorCode* error_code = NULL);

    int32_t ReadLine(void* buffer, int32_t max_size);
    int32_t ReadLine(std::string* result);

private:
    void SetErrorCode(FileErrorCode* error_code, FileErrorCode code);
    std::string FileOpenModeToString(uint32_t flag);

private:
    FILE* fp_;
};

} // namespace mybdcom
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOM_FILE_STREAM_H_