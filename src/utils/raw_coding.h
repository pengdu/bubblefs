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

// tensorflow/tensorflow/core/lib/core/raw_coding.h

#ifndef BUBBLEFS_UTILS_RAW_CODING_H_
#define BUBBLEFS_UTILS_RAW_CODING_H_

#include <string.h>
#include "platform/cpu_info.h"
#include "platform/types.h"

namespace bubblefs {
namespace core {

// Lower-level versions of Get... that read directly from a character buffer
// without any bounds checking.
  
inline void EncodeBigEndian(char* buf, uint64_t value) {
    buf[0] = (value >> 56) & 0xff;
    buf[1] = (value >> 48) & 0xff;
    buf[2] = (value >> 40) & 0xff;
    buf[3] = (value >> 32) & 0xff;
    buf[4] = (value >> 24) & 0xff;
    buf[5] = (value >> 16) & 0xff;
    buf[6] = (value >> 8) & 0xff;
    buf[7] = value & 0xff;
}

inline void EncodeBigEndian(char* buf, uint32_t value) {
    buf[0] = (value >> 24) & 0xff;
    buf[1] = (value >> 16) & 0xff;
    buf[2] = (value >> 8) & 0xff;
    buf[3] = value & 0xff;
}

inline uint64_t DecodeBigEndian64(const char* buf) {
    return ((static_cast<uint64_t>(static_cast<unsigned char>(buf[0]))) << 56
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[1])) << 48)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[2])) << 40)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[3])) << 32)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[4])) << 24)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[5])) << 16)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[6])) << 8)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[7]))));
}

inline uint32_t DecodeBigEndian32(const char* buf) {
    return ((static_cast<uint64_t>(static_cast<unsigned char>(buf[0])) << 24)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[1])) << 16)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[2])) << 8)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[3]))));
}

inline uint16 DecodeFixed16(const char* ptr) {
  if (port::kLittleEndian) {
    // Load the raw bytes
    uint16 result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    return ((static_cast<uint16>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint16>(static_cast<unsigned char>(ptr[1])) << 8));
  }
}

inline uint32 DecodeFixed32(const char* ptr) {
  if (port::kLittleEndian) {
    // Load the raw bytes
    uint32 result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    return ((static_cast<uint32>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[1])) << 8) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[2])) << 16) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[3])) << 24));
  }
}

inline uint64 DecodeFixed64(const char* ptr) {
  if (port::kLittleEndian) {
    // Load the raw bytes
    uint64 result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    uint64 lo = DecodeFixed32(ptr);
    uint64 hi = DecodeFixed32(ptr + 4);
    return (hi << 32) | lo;
  }
}

}  // namespace core
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_RAW_CODING_H_