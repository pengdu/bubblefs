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
// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//

// tensorflow/tensorflow/core/lib/core/stringpiece.h
// rocksdb/include/rocksdb/slice.h

// StringPiece is a simple structure containing a pointer into some external
// storage and a size.  The user of a StringPiece must ensure that the slice
// is not used after the corresponding external storage has been
// deallocated.
//
// Multiple threads can invoke const methods on a StringPiece without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same StringPiece must use
// external synchronization.

#ifndef BUBBLEFS_UTILS_STRINGPIECE_H_
#define BUBBLEFS_UTILS_STRINGPIECE_H_

#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <iosfwd>
#include <functional>
#include <string>
#include "platform/types.h"
#include "utils/cleanable.h"

namespace bubblefs {

struct StringPieceParts;
  
// also known as Slice
class StringPiece {
 public:
  typedef size_t size_type;

  // Create an empty slice.
  StringPiece() : data_(""), size_(0) {}

  // Create a slice that refers to d[0,n-1].
  StringPiece(const char* d, size_t n) : data_(d), size_(n) {}

  // Create a slice that refers to the contents of "s"
  StringPiece(const string& s) : data_(s.data()), size_(s.size()) {}

  // Create a slice that refers to s[0,strlen(s)-1]
  StringPiece(const char* s) : data_(s), size_(strlen(s)) {}
  
  // Create a single slice from SliceParts using buf as storage.
  // buf must exist as long as the returned Slice exists.
  StringPiece(const struct StringPieceParts& parts, string* buf);

  void set(const void* data, size_t len) {
    data_ = reinterpret_cast<const char*>(data);
    size_ = len;
  }

  // Return a pointer to the beginning of the referenced data
  const char* data() const { return data_; }

  // Return the length (in bytes) of the referenced data
  size_t size() const { return size_; }
  size_type length() const { return size_; }

  // Return true iff the length of the referenced data is zero
  bool empty() const { return size_ == 0; }

  typedef const char* const_iterator;
  typedef const char* iterator;
  iterator begin() const { return data_; }
  iterator end() const { return data_ + size_; }

  static const size_t npos;

  // Return the ith byte in the referenced data.
  // REQUIRES: n < size()
  char operator[](size_t n) const {
    assert(n < size());
    return data_[n];
  }

  // Change this slice to refer to an empty array
  void clear() {
    data_ = "";
    size_ = 0;
  }

  // Drop the first "n" bytes from this slice.
  void remove_prefix(size_t n) {
    assert(n <= size());
    data_ += n;
    size_ -= n;
  }

  void remove_suffix(size_t n) {
    assert(size_ >= n);
    size_ -= n;
  }

  size_t find(char c, size_t pos = 0) const;
  size_t rfind(char c, size_t pos = npos) const;
  bool contains(StringPiece s) const;

  // Checks whether StringPiece starts with x and if so advances the beginning
  // of it to past the match.  It's basically a shortcut for starts_with
  // followed by remove_prefix.
  bool Consume(StringPiece x) {
    if (starts_with(x)) {
      remove_prefix(x.size_);
      return true;
    }
    return false;
  }

  StringPiece substr(size_t pos, size_t n = npos) const;
  
  // Return a string that contains the copy of a suffix of the referenced data.
  std::string substr_copy(size_t start) const {
    assert(start <= size());
    return std::string(data_ + start, size_ - start);
  }

  struct Hasher {
    size_t operator()(StringPiece arg) const;
  };

  // Return a string that contains the copy of the referenced data.
  // when hex is true, returns a string of twice the length hex encoded (0-9A-F)
  string ToString(bool hex = false) const;
  
  string as_string() const {
    // std::string doesn't like to take a NULL pointer even with a 0 size.
    return empty() ? string() : string(data(), size());
  }
  
  const char* c_str() const {
    assert(data_[size_] == 0);
    return data_;
  }
  
  // Decodes the current slice interpreted as an hexadecimal string into result,
  // if successful returns true, if this isn't a valid hex string
  // (e.g not coming from Slice::ToString(true)) DecodeHex returns false.
  // This slice is expected to have an even number of 0-9A-F characters
  // also accepts lowercase (a-f)
  bool DecodeHex(string* result) const;

  // Three-way comparison.  Returns value:
  //   <  0 iff "*this" <  "b",
  //   == 0 iff "*this" == "b",
  //   >  0 iff "*this" >  "b"
  int compare(StringPiece b) const;

  // Return true iff "x" is a prefix of "*this"
  bool starts_with(StringPiece x) const {
    return ((size_ >= x.size_) && (memcmp(data_, x.data_, x.size_) == 0));
  }
  // Return true iff "x" is a suffix of "*this"
  bool ends_with(StringPiece x) const {
    return ((size_ >= x.size_) &&
            (memcmp(data_ + (size_ - x.size_), x.data_, x.size_) == 0));
  }
  
  // Compare two slices and returns the first byte where they differ
  size_t difference_offset(const StringPiece b) const;

  // private: make these public for rocksdbjni access
  const char* data_;
  size_t size_;

  // Intentionally copyable
};

// A set of Slices that are virtually concatenated together.  'parts' points
// to an array of Slices.  The number of elements in the array is 'num_parts'.
struct StringPieceParts {
  StringPieceParts(const StringPiece* _parts, int _num_parts) :
      parts(_parts), num_parts(_num_parts) {}
  StringPieceParts() : parts(nullptr), num_parts(0) {}

  const StringPiece* parts;
  int num_parts;
};

inline bool operator==(StringPiece x, StringPiece y) {
  return ((x.size() == y.size()) &&
          (memcmp(x.data(), y.data(), x.size()) == 0));
}

inline bool operator!=(StringPiece x, StringPiece y) { return !(x == y); }

inline bool operator<(StringPiece x, StringPiece y) { return x.compare(y) < 0; }
inline bool operator>(StringPiece x, StringPiece y) { return x.compare(y) > 0; }
inline bool operator<=(StringPiece x, StringPiece y) {
  return x.compare(y) <= 0;
}
inline bool operator>=(StringPiece x, StringPiece y) {
  return x.compare(y) >= 0;
}

inline int StringPiece::compare(StringPiece b) const {
  const size_t min_len = (size_ < b.size_) ? size_ : b.size_;
  int r = memcmp(data_, b.data_, min_len);
  if (r == 0) {
    if (size_ < b.size_)
      r = -1;
    else if (size_ > b.size_)
      r = +1;
  }
  return r;
}

// Compare two slices and returns the first byte where they differ
inline size_t StringPiece::difference_offset(const StringPiece b) const {
  size_t off = 0;
  const size_t len = (size_ < b.size_) ? size_ : b.size_;
  for (; off < len; off++) {
    if (data_[off] != b.data_[off]) break;
  }
  return off;
}

inline void AppendSliceTo(std::string* str, const StringPiece& value) {
  str->append(value.data(), value.size());
}

// allow StringPiece to be logged
extern std::ostream& operator<<(std::ostream& o, StringPiece piece);

/**
 * A StringPiece that can be pinned with some cleanup tasks, which will be run upon
 * ::Reset() or object destruction, whichever is invoked first. This can be used
 * to avoid memcpy by having the PinnsableSlice object referring to the data
 * that is locked in the memory and release them after the data is consumed.
 */
class PinnableSlice : public StringPiece, public Cleanable {
 public:
  PinnableSlice() { buf_ = &self_space_; }
  explicit PinnableSlice(string* buf) { buf_ = buf; }

  inline void PinSlice(const StringPiece& s, CleanupFunction f, void* arg1,
                       void* arg2) {
    assert(!pinned_);
    pinned_ = true;
    data_ = s.data();
    size_ = s.size();
    RegisterCleanup(f, arg1, arg2);
    assert(pinned_);
  }

  inline void PinSlice(const StringPiece& s, Cleanable* cleanable) {
    assert(!pinned_);
    pinned_ = true;
    data_ = s.data();
    size_ = s.size();
    cleanable->DelegateCleanupsTo(this);
    assert(pinned_);
  }

  inline void PinSelf(const StringPiece& slice) {
    assert(!pinned_);
    buf_->assign(slice.data(), slice.size());
    data_ = buf_->data();
    size_ = buf_->size();
    assert(!pinned_);
  }

  inline void PinSelf() {
    assert(!pinned_);
    data_ = buf_->data();
    size_ = buf_->size();
    assert(!pinned_);
  }

  void remove_suffix(size_t n) {
    assert(n <= size());
    if (pinned_) {
      size_ -= n;
    } else {
      buf_->erase(size() - n, n);
      PinSelf();
    }
  }

  void remove_prefix(size_t n) {
    assert(0);  // Not implemented
  }

  void Reset() {
    Cleanable::Reset();
    pinned_ = false;
  }

  inline string* GetSelf() { return buf_; }

  inline bool IsPinned() { return pinned_; }

 private:
  friend class PinnableSlice4Test;
  string self_space_;
  string* buf_;
  bool pinned_ = false;
};

}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_STRINGPIECE_H_