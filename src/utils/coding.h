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

// tensorflow/tensorflow/core/lib/core/coding.h

// Endian-neutral encoding:
// * Fixed-length numbers are encoded with least-significant byte first
// * In addition we support variable length "varint" encoding
// * Strings are encoded prefixed by their length in varint format

#ifndef BUBBLEFS_UTILS_CODING_H_
#define BUBBLEFS_UTILS_CODING_H_

#include <stdint.h>
#include <strings.h>
#include "platform/types.h"
#include "utils/raw_coding.h"
#include "utils/stringpiece.h"

// Some processors does not allow unaligned access to memory
#if defined(__sparc)
  #define PLATFORM_UNALIGNED_ACCESS_NOT_ALLOWED
#endif

namespace bubblefs {
namespace core {
  
static inline void EncodeBigEndian(char* buf, uint64_t value) {
    buf[0] = (value >> 56) & 0xff;
    buf[1] = (value >> 48) & 0xff;
    buf[2] = (value >> 40) & 0xff;
    buf[3] = (value >> 32) & 0xff;
    buf[4] = (value >> 24) & 0xff;
    buf[5] = (value >> 16) & 0xff;
    buf[6] = (value >> 8) & 0xff;
    buf[7] = value & 0xff;
}

static inline uint64_t DecodeBigEndian64(const char* buf) {
    return ((static_cast<uint64_t>(static_cast<unsigned char>(buf[0]))) << 56
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[1])) << 48)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[2])) << 40)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[3])) << 32)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[4])) << 24)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[5])) << 16)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[6])) << 8)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[7]))));
}

static inline void EncodeBigEndian(char* buf, uint32_t value) {
    buf[0] = (value >> 24) & 0xff;
    buf[1] = (value >> 16) & 0xff;
    buf[2] = (value >> 8) & 0xff;
    buf[3] = value & 0xff;
}

static inline uint32_t DecodeBigEndian32(const char* buf) {
    return ((static_cast<uint64_t>(static_cast<unsigned char>(buf[0])) << 24)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[1])) << 16)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[2])) << 8)
        | (static_cast<uint64_t>(static_cast<unsigned char>(buf[3]))));
}

// Maximum number of bytes occupied by a varint32.
static const int kMaxVarint32Bytes = 5;
// The maximum length of a varint in bytes for 64-bit.
static const unsigned int kMaxVarint64Bytes = 10;

// Lower-level versions of Put... that write directly into a character buffer
// REQUIRES: dst has enough space for the value being written
extern void EncodeFixed16(char* dst, uint16 value);
extern void EncodeFixed32(char* dst, uint32 value);
extern void EncodeFixed64(char* dst, uint64 value);

extern void PutFixed16(string* dst, uint16 value);
extern void PutFixed32(string* dst, uint32 value);
extern void PutFixed64(string* dst, uint64 value);

extern void PutVarint32(string* dst, uint32 value);
extern void PutVarint64(string* dst, uint64 value);

extern void PutVarint32Varint32(string* dst, uint32_t value1,
                                uint32_t value2);
extern void PutVarint32Varint32Varint32(string* dst, uint32_t value1,
                                        uint32_t value2, uint32_t value3);
extern void PutVarint64Varint64(string* dst, uint64_t value1,
                                uint64_t value2);
extern void PutVarint32Varint64(string* dst, uint32_t value1,
                                uint64_t value2);
extern void PutVarint32Varint32Varint64(string* dst, uint32_t value1,
                                        uint32_t value2, uint64_t value3);

extern void PutLengthPrefixedSlice(string* dst, const StringPieceParts& value);
extern void PutLengthPrefixedSliceParts(string* dst,
                                        const StringPieceParts& slice_parts);

// Standard Get... routines parse a value from the beginning of a StringPiece
// and advance the slice past the parsed value.
extern bool GetFixed64(StringPiece* input, uint64_t* value);
extern bool GetFixed32(StringPiece* input, uint32_t* value);

extern bool GetVarint32(StringPiece* input, uint32* value);
extern bool GetVarint64(StringPiece* input, uint64* value);

// Provide an interface for platform independent endianness transformation
extern uint64_t EndianTransform(uint64_t input, size_t size);

extern bool GetLengthPrefixedSlice(StringPiece* input, StringPiece* result);
// This function assumes data is well-formed.
extern StringPiece GetLengthPrefixedSlice(const char* data);
extern StringPiece GetSliceUntil(StringPiece* slice, char delimiter);

// Pointer-based variants of GetVarint...  These either store a value
// in *v and return a pointer just past the parsed value, or return
// nullptr on error.  These routines only look at bytes in the range
// [p..limit-1]
// Internal routine for use by fallback path of GetVarint32Ptr
extern const char* GetVarint32PtrFallback(const char* p, const char* limit,
                                          uint32* value);
extern const char* GetVarint32Ptr(const char* p, const char* limit, uint32* value);
extern const char* GetVarint64Ptr(const char* p, const char* limit, uint64* v);

// Lower-level versions of Put... that write directly into a character buffer
// REQUIRES: dst has enough space for the value being written
extern void EncodeFixed32(char* dst, uint32_t value);
extern void EncodeFixed64(char* dst, uint64_t value);

// Lower-level versions of Put... that write directly into a character buffer
// and return a pointer just past the last byte written.
// REQUIRES: dst has enough space for the value being written
extern char* EncodeVarint32(char* dst, uint32 v);
extern char* EncodeVarint64(char* dst, uint64 v);

// Returns the length of the varint32 or varint64 encoding of "v"
extern int VarintLength(uint64_t v);

// Lower-level versions of Get... that read directly from a character buffer
// without any bounds checking.

inline uint32_t DecodeFixed32(const char* ptr) {
  if (port::kLittleEndian) {
    // Load the raw bytes
    uint32_t result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    return ((static_cast<uint32_t>(static_cast<unsigned char>(ptr[0])))
        | (static_cast<uint32_t>(static_cast<unsigned char>(ptr[1])) << 8)
        | (static_cast<uint32_t>(static_cast<unsigned char>(ptr[2])) << 16)
        | (static_cast<uint32_t>(static_cast<unsigned char>(ptr[3])) << 24));
  }
}

inline uint64_t DecodeFixed64(const char* ptr) {
  if (port::kLittleEndian) {
    // Load the raw bytes
    uint64_t result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    uint64_t lo = DecodeFixed32(ptr);
    uint64_t hi = DecodeFixed32(ptr + 4);
    return (hi << 32) | lo;
  }
}

template<class T>
#if defined(__clang__)
__attribute__((__no_sanitize__("alignment")))
#elif defined(__GNUC__)
__attribute__((__no_sanitize_undefined__))
#endif
inline void PutUnaligned(T *memory, const T &value) {
#if defined(PLATFORM_UNALIGNED_ACCESS_NOT_ALLOWED)
  char *nonAlignedMemory = reinterpret_cast<char*>(memory);
  memcpy(nonAlignedMemory, reinterpret_cast<const char*>(&value), sizeof(T));
#else
  *memory = value;
#endif
}

template<class T>
#if defined(__clang__)
__attribute__((__no_sanitize__("alignment")))
#elif defined(__GNUC__)
__attribute__((__no_sanitize_undefined__))
#endif
inline void GetUnaligned(const T *memory, T *value) {
#if defined(PLATFORM_UNALIGNED_ACCESS_NOT_ALLOWED)
  char *nonAlignedMemory = reinterpret_cast<char*>(value);
  memcpy(nonAlignedMemory, reinterpret_cast<const char*>(memory), sizeof(T));
#else
  *value = *memory;
#endif
}

}  // namespace core
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_CODING_H_