/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

// tensorflow/tensorflow/core/platform/posix/error.h

#ifndef BUBBLEFS_PLATFORM_ERROR_H_
#define BUBBLEFS_PLATFORM_ERROR_H_

#include "utils/status.h"

namespace bubblefs {
  
Status IOError(const string& context, int err_number);

Status IOError(const string& context, const string& file_name, int err_number);

}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_ERROR_H_