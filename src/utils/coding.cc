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

// tensorflow/tensorflow/core/lib/core/coding.cc

#include "platform/cpu_info.h"
#include "utils/coding.h"

namespace bubblefs {
namespace core {

void EncodeFixed16(char* buf, uint16 value) {
  if (port::kLittleEndian) {
    memcpy(buf, &value, sizeof(value));
  } else {
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
  }
}

void EncodeFixed32(char* buf, uint32 value) {
  if (port::kLittleEndian) {
    memcpy(buf, &value, sizeof(value));
  } else {
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
    buf[2] = (value >> 16) & 0xff;
    buf[3] = (value >> 24) & 0xff;
  }
}

void EncodeFixed64(char* buf, uint64 value) {
  if (port::kLittleEndian) {
    memcpy(buf, &value, sizeof(value));
  } else {
    buf[0] = value & 0xff;
    buf[1] = (value >> 8) & 0xff;
    buf[2] = (value >> 16) & 0xff;
    buf[3] = (value >> 24) & 0xff;
    buf[4] = (value >> 32) & 0xff;
    buf[5] = (value >> 40) & 0xff;
    buf[6] = (value >> 48) & 0xff;
    buf[7] = (value >> 56) & 0xff;
  }
}

void PutFixed16(string* dst, uint16 value) {
  char buf[sizeof(value)];
  EncodeFixed16(buf, value);
  dst->append(buf, sizeof(buf));
}

void PutFixed32(string* dst, uint32 value) {
  char buf[sizeof(value)];
  EncodeFixed32(buf, value);
  dst->append(buf, sizeof(buf));
}

void PutFixed64(string* dst, uint64 value) {
  char buf[sizeof(value)];
  EncodeFixed64(buf, value);
  dst->append(buf, sizeof(buf));
}

char* EncodeVarint32(char* dst, uint32 v) {
  // Operate on characters as unsigneds
  unsigned char* ptr = reinterpret_cast<unsigned char*>(dst);
  static const int B = 128;
  if (v < (1 << 7)) {
    *(ptr++) = v;
  } else if (v < (1 << 14)) {
    *(ptr++) = v | B;
    *(ptr++) = v >> 7;
  } else if (v < (1 << 21)) {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = v >> 14;
  } else if (v < (1 << 28)) {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = (v >> 14) | B;
    *(ptr++) = v >> 21;
  } else {
    *(ptr++) = v | B;
    *(ptr++) = (v >> 7) | B;
    *(ptr++) = (v >> 14) | B;
    *(ptr++) = (v >> 21) | B;
    *(ptr++) = v >> 28;
  }
  return reinterpret_cast<char*>(ptr);
}

void PutVarint32(string* dst, uint32 v) {
  char buf[5];
  char* ptr = EncodeVarint32(buf, v);
  dst->append(buf, ptr - buf);
}

char* EncodeVarint64(char* dst, uint64 v) {
  static const int B = 128;
  unsigned char* ptr = reinterpret_cast<unsigned char*>(dst);
  while (v >= B) {
    *(ptr++) = (v & (B - 1)) | B;
    v >>= 7;
  }
  *(ptr++) = static_cast<unsigned char>(v);
  return reinterpret_cast<char*>(ptr);
}

void PutVarint64(string* dst, uint64 v) {
  char buf[10];
  char* ptr = EncodeVarint64(buf, v);
  dst->append(buf, ptr - buf);
}

void PutVarint32Varint32(string* dst, uint32_t v1, uint32_t v2) {
  char buf[10];
  char* ptr = EncodeVarint32(buf, v1);
  ptr = EncodeVarint32(ptr, v2);
  dst->append(buf, static_cast<size_t>(ptr - buf));
}

void PutVarint32Varint32Varint32(string* dst, uint32_t v1,
                                 uint32_t v2, uint32_t v3) {
  char buf[15];
  char* ptr = EncodeVarint32(buf, v1);
  ptr = EncodeVarint32(ptr, v2);
  ptr = EncodeVarint32(ptr, v3);
  dst->append(buf, static_cast<size_t>(ptr - buf));
}

void PutVarint64Varint64(string* dst, uint64_t v1, uint64_t v2) {
  char buf[20];
  char* ptr = EncodeVarint64(buf, v1);
  ptr = EncodeVarint64(ptr, v2);
  dst->append(buf, static_cast<size_t>(ptr - buf));
}

void PutVarint32Varint64(string* dst, uint32_t v1, uint64_t v2) {
  char buf[15];
  char* ptr = EncodeVarint32(buf, v1);
  ptr = EncodeVarint64(ptr, v2);
  dst->append(buf, static_cast<size_t>(ptr - buf));
}

void PutVarint32Varint32Varint64(string* dst, uint32_t v1,
                                 uint32_t v2, uint64_t v3) {
  char buf[20];
  char* ptr = EncodeVarint32(buf, v1);
  ptr = EncodeVarint32(ptr, v2);
  ptr = EncodeVarint64(ptr, v3);
  dst->append(buf, static_cast<size_t>(ptr - buf));
}

void PutLengthPrefixedSliceParts(string* dst,
                                 const StringPieceParts& slice_parts) {
  size_t total_bytes = 0;
  for (int i = 0; i < slice_parts.num_parts; ++i) {
    total_bytes += slice_parts.parts[i].size();
  }
  PutVarint32(dst, static_cast<uint32_t>(total_bytes));
  for (int i = 0; i < slice_parts.num_parts; ++i) {
    dst->append(slice_parts.parts[i].data(), slice_parts.parts[i].size());
  }
}

bool GetFixed64(StringPiece* input, uint64_t* value) {
  if (input->size() < sizeof(uint64_t)) {
    return false;
  }
  *value = DecodeFixed64(input->data());
  input->remove_prefix(sizeof(uint64_t));
  return true;
}

bool GetFixed32(StringPiece* input, uint32_t* value) {
  if (input->size() < sizeof(uint32_t)) {
    return false;
  }
  *value = DecodeFixed32(input->data());
  input->remove_prefix(sizeof(uint32_t));
  return true;
}

bool GetVarint32(StringPiece* input, uint32_t* value) {
  const char* p = input->data();
  const char* limit = p + input->size();
  const char* q = GetVarint32Ptr(p, limit, value);
  if (q == nullptr) {
    return false;
  } else {
    *input = StringPiece(q, static_cast<size_t>(limit - q));
    return true;
  }
}

bool GetVarint64(StringPiece* input, uint64_t* value) {
  const char* p = input->data();
  const char* limit = p + input->size();
  const char* q = GetVarint64Ptr(p, limit, value);
  if (q == nullptr) {
    return false;
  } else {
    *input = StringPiece(q, static_cast<size_t>(limit - q));
    return true;
  }
}

uint64_t EndianTransform(uint64_t input, size_t size) {
  char* pos = reinterpret_cast<char*>(&input);
  uint64_t ret_val = 0;
  for (size_t i = 0; i < size; ++i) {
    ret_val |= (static_cast<uint64_t>(static_cast<unsigned char>(pos[i]))
                << ((size - i - 1) << 3));
  }
  return ret_val;
}

bool GetLengthPrefixedSlice(StringPiece* input, StringPiece* result) {
  uint32_t len = 0;
  if (GetVarint32(input, &len) && input->size() >= len) {
    *result = StringPiece(input->data(), len);
    input->remove_prefix(len);
    return true;
  } else {
    return false;
  }
}

StringPiece GetLengthPrefixedSlice(const char* data) {
  uint32_t len = 0;
  // +5: we assume "data" is not corrupted
  // unsigned char is 7 bits, uint32_t is 32 bits, need 5 unsigned char
  auto p = GetVarint32Ptr(data, data + 5 /* limit */, &len);
  return StringPiece(p, len);
}

StringPiece GetSliceUntil(StringPiece* slice, char delimiter) {
  uint32_t len = 0;
  for (len = 0; len < slice->size() && slice->data()[len] != delimiter; ++len) {
    // nothing
  }

  StringPiece ret(slice->data(), len);
  slice->remove_prefix(len + ((len < slice->size()) ? 1 : 0));
  return ret;
}

int VarintLength(uint64_t v) {
  int len = 1;
  while (v >= 128) {
    v >>= 7;
    len++;
  }
  return len;
}

const char* GetVarint32PtrFallback(const char* p, const char* limit,
                                   uint32* value) {
  uint32 result = 0;
  for (uint32 shift = 0; shift <= 28 && p < limit; shift += 7) {
    uint32 byte = *(reinterpret_cast<const unsigned char*>(p));
    p++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return reinterpret_cast<const char*>(p);
    }
  }
  return nullptr;
}

const char* GetVarint32Ptr(const char* p, const char* limit,
                           uint32* value) {
  if (p < limit) {
    uint32 result = *(reinterpret_cast<const unsigned char*>(p));
    if ((result & 128) == 0) {
      *value = result;
      return p + 1;
    }
  }
  return GetVarint32PtrFallback(p, limit, value);
}

const char* GetVarint64Ptr(const char* p, const char* limit, uint64* value) {
  uint64 result = 0;
  for (uint32 shift = 0; shift <= 63 && p < limit; shift += 7) {
    uint64 byte = *(reinterpret_cast<const unsigned char*>(p));
    p++;
    if (byte & 128) {
      // More bytes are present
      result |= ((byte & 127) << shift);
    } else {
      result |= (byte << shift);
      *value = result;
      return reinterpret_cast<const char*>(p);
    }
  }
  return nullptr;
}

}  // namespace core
}  // namespace bubblefs