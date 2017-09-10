/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tensorflow/tensorflow/core/platform/posix/posix_file_system.h

#ifndef BUBBLEFS_PLATFORM_FILE_SYSTEM_POSIX_H_
#define BUBBLEFS_PLATFORM_FILE_SYSTEM_POSIX_H_

#include "platform/env.h"
#include "platform/error.h"
#include "utils/path.h"

namespace bubblefs {


class LocalPosixFileSystem : public PosixFileSystem {
 public:
  string TranslateName(const string& name) const override {
    StringPiece scheme, host, path;
    io::ParseURI(name, &scheme, &host, &path);
    return path.ToString();
  }
};

}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_FILE_SYSTEM_POSIX_H_
