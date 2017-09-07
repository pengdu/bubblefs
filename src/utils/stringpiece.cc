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
#include "utils/str_util.h"

namespace bubblefs {

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
      result.push_back(str_util::ToHex(c >> 4));
      result.push_back(str_util::ToHex(c & 0xf));
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
    int h1 = str_util::FromHex(data_[i++]);
    if (h1 < 0) {
      return false;
    }
    int h2 = str_util::FromHex(data_[i++]);
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