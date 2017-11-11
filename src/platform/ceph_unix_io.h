/*
 * Ceph - scalable distributed file system
 *
 * Copyright (C) 2011 New Dream Network
 *
 * This is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1, as published by the Free Software
 * Foundation.  See file COPYING.
 *
 */
/*
 * Tencent is pleased to support the open source community by making Pebble available.
 * Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.
 * Licensed under the MIT License (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 * http://opensource.org/licenses/MIT
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

// ceph/src/common/safe_io.h
// Pebble/src/common/dir_util.h

#ifndef BUBBLEFS_PALTFORM_CEPH_UNIX_IO_
#define BUBBLEFS_PALTFORM_CEPH_UNIX_IO_

#include <sys/types.h>
#include <string>
#include "platform/macros.h"

namespace bubblefs {
namespace myceph {
/*
 * Safe functions wrapping the raw read() and write() libc functions.
 * These retry on EINTR, and on error return -errno instead of returning
 * -1 and setting errno).
 */
ssize_t safe_read(int fd, void *buf, size_t count)
    WARN_UNUSED_RESULT;
ssize_t safe_write(int fd, const void *buf, size_t count)
    WARN_UNUSED_RESULT;
ssize_t safe_pread(int fd, void *buf, size_t count, off_t offset)
    WARN_UNUSED_RESULT;
ssize_t safe_pwrite(int fd, const void *buf, size_t count, off_t offset)
    WARN_UNUSED_RESULT;
/*
 * Similar to the above (non-exact version) and below (exact version).
 * See splice(2) for parameter descriptions.
 */
ssize_t safe_splice(int fd_in, off_t *off_in, int fd_out, off_t *off_out,
                    size_t len, unsigned int flags)
  WARN_UNUSED_RESULT;
ssize_t safe_splice_exact(int fd_in, off_t *off_in, int fd_out,
                          off_t *off_out, size_t len, unsigned int flags)
  WARN_UNUSED_RESULT;

/*
 * Same as the above functions, but return -EDOM unless exactly the requested
 * number of bytes can be read.
 */
ssize_t safe_read_exact(int fd, void *buf, size_t count)
    WARN_UNUSED_RESULT;
ssize_t safe_pread_exact(int fd, void *buf, size_t count, off_t offset)
    WARN_UNUSED_RESULT;

/*
 * Safe functions to read and write an entire file.
 */
int safe_write_file(const char *base, const char *file,
                    const char *val, size_t vallen);
int safe_read_file(const char *base, const char *file,
                   char *val, size_t vallen);

class DirUtil {
public:
    /// @brief 创建目录，仅在目录不存在时创建(同mkdir)
    /// @param path 要创建的目录
    /// @return 0成功，非0失败
    static int MakeDir(const std::string& path);

    /// @brief 创建多级目录，连同父目录一起创建(同mkdir -p)
    /// @param path 要创建的目录
    /// @return 0成功，非0失败
    static int MakeDirP(const std::string& path);

    static const char* GetLastError() {
        return m_last_error;
    }

private:
    static char m_last_error[256];
};

} // namespace myceph
} // namespace bubblefs

#endif // BUBBLEFS_PALTFORM_CEPH_UNIX_IO_