/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/core/flags.cc

#include "platform/caffe2_flags.h"
#include <stdlib.h>
#include <sstream>

namespace bubblefs {
namespace mycaffe2 {
  
void SetUsageMessage(const std::string& str) {
  if (UsageMessage() != nullptr) {
    // Usage message has already been set, so we will simply return.
    return;
  }
  gflags::SetUsageMessage(str);
}

const char* UsageMessage() {
  return gflags::ProgramUsage();
}

bool ParseCaffeCommandLineFlags(int* pargc, char*** pargv) {
  if (*pargc == 0) return true;
  return gflags::ParseCommandLineFlags(pargc, pargv, true);
}

bool CommandLineFlagsHasBeenParsed() {
  // There is no way we query gflags right now, so we will simply return true.
  return true;
}

} // namespace mycaffe2
} // namespace bubblefs