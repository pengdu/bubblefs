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

// tensorflow/tensorflow/core/lib/core/stringpiece.cc

#include "utils/stringpiece.h"
#include <algorithm>
#include <iostream>
#include "utils/hash.h"

namespace bubblefs {
  
namespace {

// For each character in characters_wanted, sets the index corresponding
// to the ASCII code of that character to 1 in table.  This is used by
// the find_.*_of methods below to tell whether or not a character is in
// the lookup table in constant time.
// The argument `table' must be an array that is large enough to hold all
// the possible values of an unsigned char.  Thus it should be be declared
// as follows:
//   bool table[UCHAR_MAX + 1]
/*
inline void BuildLookupTable(const StringPiece& characters_wanted,
                             bool* table) {
  const size_t length = characters_wanted.length();
  const char* const data = characters_wanted.data();
  for (size_t i = 0; i < length; ++i) {
    table[static_cast<unsigned char>(data[i])] = true;
  }
}
*/

}  // namespace

// 2 small internal utility functions, for efficient hex conversions
// and no need for snprintf, toupper etc...
// Originally from wdt/util/EncryptionUtils.cpp - for ToString(true)/DecodeHex:
static char toHex(unsigned char v) {
  if (v <= 9) {
    return '0' + v;
  }
  return 'A' + v - 10;
}
// most of the code is for validation/error check
static int fromHex(char c) {
  // toupper:
  if (c >= 'a' && c <= 'f') {
    c -= ('a' - 'A');  // aka 0x20
  }
  // validation
  if (c < '0' || (c > '9' && (c < 'A' || c > 'F'))) {
    return -1;  // invalid not 0-9A-F hex char
  }
  if (c <= '9') {
    return c - '0';
  }
  return c - 'A' + 10;
}

StringPiece::StringPiece(const StringPieceParts& parts, string* buf) {
  size_t length = 0;
  for (int i = 0; i < parts.num_parts; ++i) {
    length += parts.parts[i].size();
  }
  buf->reserve(length);

  for (int i = 0; i < parts.num_parts; ++i) {
    buf->append(parts.parts[i].data(), parts.parts[i].size());
  }
  data_ = buf->data();
  size_ = buf->size();
}

// Return a string that contains the copy of the referenced data.
string StringPiece::ToString(bool hex) const {
  string result;  // RVO/NRVO/move
  if (hex) {
    result.reserve(2 * size_);
    for (size_t i = 0; i < size_; ++i) {
      unsigned char c = data_[i];
      result.push_back(toHex(c >> 4));
      result.push_back(toHex(c & 0xf));
    }
    return result;
  } else {
    result.assign(data_, size_);
    return result;
  }
}

bool StringPiece::DecodeHex(string* result) const {
  string::size_type len = size_;
  if (len % 2) {
    // Hex string must be even number of hex digits to get complete bytes back
    return false;
  }
  if (!result) {
    return false;
  }
  result->clear();
  result->reserve(len / 2);

  for (size_t i = 0; i < len;) {
    int h1 = fromHex(data_[i++]);
    if (h1 < 0) {
      return false;
    }
    int h2 = fromHex(data_[i++]);
    if (h2 < 0) {
      return false;
    }
    result->push_back((h1 << 4) | h2);
  }
  return true;
}

size_t StringPiece::Hasher::operator()(StringPiece s) const {
  return Hash64(s.data(), s.size());
}

std::ostream& operator<<(std::ostream& o, StringPiece piece) {
  o.write(piece.data(), piece.size());
  return o;
}

bool StringPiece::contains(StringPiece s) const {
  return std::search(begin(), end(), s.begin(), s.end()) != end();
}

size_t StringPiece::find(char c, size_t pos) const {
  if (pos >= size_) {
    return npos;
  }
  const char* result =
      reinterpret_cast<const char*>(memchr(data_ + pos, c, size_ - pos));
  return result != nullptr ? result - data_ : npos;
}

// Search range is [0..pos] inclusive.  If pos == npos, search everything.
size_t StringPiece::rfind(char c, size_t pos) const {
  if (size_ == 0) return npos;
  for (const char* p = data_ + std::min(pos, size_ - 1); p >= data_; p--) {
    if (*p == c) {
      return p - data_;
    }
  }
  return npos;
}

StringPiece StringPiece::substr(size_t pos, size_t n) const {
  if (pos > size_) pos = size_;
  if (n > size_ - pos) n = size_ - pos;
  return StringPiece(data_ + pos, n);
}

const StringPiece::size_type StringPiece::npos = size_type(-1);

}  // namespace bubblefs