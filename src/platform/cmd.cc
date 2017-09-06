/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Paddle/paddle/utils/PythonUtil.h

#include "platform/cmd.h"
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include "platform/logging.h"

namespace bubblefs {
namespace cmd {

constexpr int kExecuteCMDBufLength = 204800;  
  
int ExecuteCMD(const char* cmd, char* result) {
  char bufPs[kExecuteCMDBufLength];
  char ps[kExecuteCMDBufLength] = {0};
  FILE* ptr = nullptr;
  strncpy(ps, cmd, kExecuteCMDBufLength);
  if ((ptr = popen(ps, "r")) != nullptr) {
    size_t count = fread(bufPs, 1, kExecuteCMDBufLength, ptr);
    memcpy(result,
           bufPs,
           count - 1);  // why count-1: remove the '\n' at the end
    result[count] = 0;
    pclose(ptr);
    ptr = nullptr;
    return count - 1;
  } else {
    LOG(FATAL) << "ExecuteCMD popen failed";
    return -1;
  }
}
  
} // namespace cmd  
} // namespace bubblefs