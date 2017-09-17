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

// tensorflow/tensorflow/core/platform/snappy.h

#ifndef BUBBLEFS_PLATFORM_COMPRESSION_H_
#define BUBBLEFS_PLATFORM_COMPRESSION_H_

#include "platform/macros.h"
#include "platform/types.h"

namespace bubblefs {
namespace port {

inline bool Snappy_Supported() {
#ifdef TF_USE_SNAPPY
  return true;
#endif
  return false;
}
  
// Snappy compression/decompression support
bool Snappy_Compress(const char* input, size_t length, string* output);

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result);
bool Snappy_Uncompress(const char* input, size_t length, char* output);

}  // namespace port
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_COMPRESSION_H_