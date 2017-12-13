// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/common/file/file_types.h

#ifndef BUBBLEFS_UTILS_BDCOM_FILE_TYPES_H_
#define BUBBLEFS_UTILS_BDCOM_FILE_TYPES_H_

#include <stdint.h>

namespace bubblefs {
namespace mybdcom {
  
enum FileOpenMode {
    FILE_READ = 0x01,
    FILE_WRITE = 0x02,
    FILE_APPEND = 0x04
};

enum FileErrorCode {
    kFileSuccess,
    kFileErrParameter,
    kFileErrOpenFail,
    kFileErrNotOpen,
    kFileErrWrite,
    kFileErrRead,
    kFileErrClose,
    kFileErrNotExit
};

} // namespace mybdcom
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOM_FILE_TYPES_H_