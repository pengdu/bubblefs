// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/common/file/file_path.h

#ifndef BUBBLEFS_UTILS_BDCOM_FILE_PATH_H_
#define BUBBLEFS_UTILS_BDCOM_FILE_PATH_H_

#include <unistd.h>
#include <string>
#include <vector>

namespace bubblefs {
namespace mybdcom {
  
void SplitStringPath(const std::string& full_path,
                     std::string* dir_part,
                     std::string* file_part);

std::string ConcatStringPath(const std::vector<std::string>& sections,
                             const std::string& delim = ".");

std::string GetPathPrefix(const std::string& full_path,
                          const std::string& delim = "/");

bool CreateDirWithRetry(const std::string& dir_path);

std::string GidToName(gid_t gid);

std::string UidToName(uid_t uid);

bool ListCurrentDir(const std::string& dir_path,
                    std::vector<std::string>* file_list);

bool IsExist(const std::string& path);

bool IsDir(const std::string& path);

// return true when path is a dir and empty, or return false
bool IsEmpty(const std::string& path);

bool RemoveLocalFile(const std::string& path);

bool MoveLocalFile(const std::string& src_file,
                   const std::string& dst_file);

} // namespace mybdcom
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOM_FILE_PATH_H_