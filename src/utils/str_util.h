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
// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com
//================================================================================
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
//=================================================================================
//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
/*
 * Tencent is pleased to support the open source community by making Pebble available.
 * Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.
 * Licensed under the MIT License (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 * http://opensource.org/licenses/MIT
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

////////////////////////////////////////////////////////////////////////////////

// protobuf/src/google/protobuf/stubs/strutil.h
// Paddle/paddle/utils/StringUtil.h
// Pebble/src/common/string_utility.h
// baidu/common/include/string_util.h
// mars/mars/comm/strutil.h
// rocksdb/util/string_util..h
// tensorflow/tensorflow/core/lib/strings/str_util.h

#ifndef BUBBLEFS_UTILS_STR_UTIL_H_
#define BUBBLEFS_UTILS_STR_UTIL_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
#include "platform/types.h"
#include "utils/array_slice.h"
#include "utils/strcat.h"
#include "utils/stringpiece.h"

// Basic string utility routines
namespace bubblefs {
namespace str_util {
  
extern const string kNullptrString;

// ----------------------------------------------------------------------
// ascii_isalnum()
//    Check if an ASCII character is alphanumeric.  We can't use ctype's
//    isalnum() because it is affected by locale.  This function is applied
//    to identifiers in the protocol buffer language, not to natural-language
//    strings, so locale should not be taken into account.
// ascii_isdigit()
//    Like above, but only accepts digits.
// ascii_isspace()
//    Check if the character is a space character.
// ----------------------------------------------------------------------

inline bool ascii_isalnum(char c) {
  return ('a' <= c && c <= 'z') ||
         ('A' <= c && c <= 'Z') ||
         ('0' <= c && c <= '9');
}

inline bool ascii_isdigit(char c) {
  return ('0' <= c && c <= '9');
}

inline bool ascii_isspace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' ||
      c == '\r';
}

inline bool ascii_isupper(char c) {
  return c >= 'A' && c <= 'Z';
}

inline bool ascii_islower(char c) {
  return c >= 'a' && c <= 'z';
}

inline char ascii_toupper(char c) {
  return ascii_islower(c) ? c - ('a' - 'A') : c;
}

inline char ascii_tolower(char c) {
  return ascii_isupper(c) ? c + ('a' - 'A') : c;
}

inline int hex_digit_to_int(char c) {
  /* Assume ASCII. */
  int x = static_cast<unsigned char>(c);
  if (x > '9') {
    x += 9;
  }
  return x & 0xf;
}

// stringfy
template <typename T>
inline string ToString(T value) {
#if !(defined OS_ANDROID) && !(defined CYGWIN) && !(defined OS_FREEBSD)
  return std::to_string(value);
#else
  // Andorid or cygwin doesn't support all of C++11, std::to_string() being
  // one of the not supported features.
  std::ostringstream os;
  os << value;
  return os.str();
#endif
}

// Reserves enough memory in |str| to accommodate |length_with_null| characters,
// sets the size of |str| to |length_with_null - 1| characters, and returns a
// pointer to the underlying contiguous array of characters.  This is typically
// used when calling a function that writes results into a character array, but
// the caller wants the data to be managed by a string-like object.  It is
// convenient in that is can be used inline in the call, and fast in that it
// avoids copying the results of the call from a char* into a string.
//
// |length_with_null| must be at least 2, since otherwise the underlying string
// would have size 0, and trying to access &((*str)[0]) in that case can result
// in a number of problems.
//
// Internally, this takes linear time because the resize() call 0-fills the
// underlying array for potentially all
// (|length_with_null - 1| * sizeof(string_type::value_type)) bytes.  Ideally we
// could avoid this aspect of the resize() call, as we expect the caller to
// immediately write over this memory, but there is no other way to set the size
// of the string, and not doing that will mean people who access |str| rather
// than str.c_str() will get back a string of whatever size |str| had on entry
// to this function (probably 0).
template <class string_type>
inline typename string_type::value_type* WriteInto(string_type* str,
                                                   size_t length_with_null) {
  DCHECK_GT(length_with_null, 1u);
  str->reserve(length_with_null);
  str->resize(length_with_null - 1);
  return &((*str)[0]);
}
  
inline bool IsVisible(char c) {
    return (c >= 0x20 && c <= 0x7E);
}

// 2 small internal utility functions, for efficient hex conversions
// and no need for snprintf, toupper etc...
// Originally from wdt/util/EncryptionUtils.cpp - for ToString(true)/DecodeHex:
inline char ToHex(uint8_t v) {
  if (v <= 9) {
    return '0' + v;
  }
  return 'A' + v - 10;
}

// most of the code is for validation/error check
inline int FromHex(char c) {
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

// ASCII-specific tolower.  The standard library's tolower is locale sensitive,
// so we don't want to use it here.
inline char ToLowerASCII(char c) {
  return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}
inline uint16_t ToLowerASCII(uint16_t c) {
  return (c >= 'A' && c <= 'Z') ? (c + ('a' - 'A')) : c;
}

// ASCII-specific toupper.  The standard library's toupper is locale sensitive,
// so we don't want to use it here.
inline char ToUpperASCII(char c) {
  return (c >= 'a' && c <= 'z') ? (c + ('A' - 'a')) : c;
}
inline uint16_t ToUpperASCII(uint16_t c) {
  return (c >= 'a' && c <= 'z') ? (c + ('A' - 'a')) : c;
}

// Function objects to aid in comparing/searching strings.
// butil::CaseInsensitiveCompare<typename STR::value_type>()
template<typename Char> struct CaseInsensitiveCompare {
 public:
  bool operator()(Char x, Char y) const {
    // TODO(darin): Do we really want to do locale sensitive comparisons here?
    // See http://crbug.com/24917
    return tolower(x) == tolower(y);
  }
};

template<typename Char> struct CaseInsensitiveCompareASCII {
 public:
  bool operator()(Char x, Char y) const {
    return ToLowerASCII(x) == ToLowerASCII(y);
  }
};

// Determines the type of ASCII character, independent of locale (the C
// library versions will change based on locale).
template <typename Char>
inline bool IsAsciiWhitespace(Char c) {
  return c == ' ' || c == '\r' || c == '\n' || c == '\t';
}
template <typename Char>
inline bool IsAsciiAlpha(Char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
template <typename Char>
inline bool IsAsciiUpper(Char c) {
  return c >= 'A' && c <= 'Z';
}
template <typename Char>
inline bool IsAsciiLower(Char c) {
  return c >= 'a' && c <= 'z';
}
template <typename Char>
inline bool IsAsciiDigit(Char c) {
  return c >= '0' && c <= '9';
}

template <typename Char>
inline bool IsHexDigit(Char c) {
  return (c >= '0' && c <= '9') ||
         (c >= 'A' && c <= 'F') ||
         (c >= 'a' && c <= 'f');
}

template <typename Char>
inline Char HexDigitToInt(Char c) {
  DCHECK(IsHexDigit(c));
  if (c >= '0' && c <= '9')
    return c - '0';
  if (c >= 'A' && c <= 'F')
    return c - 'A' + 10;
  if (c >= 'a' && c <= 'f')
    return c - 'a' + 10;
  return 0;
}

// Append a human-readable time in micros.
int AppendHumanMicros(uint64_t micros, char* output, int len,
                      bool fixed_format);

// Append a human-readable size in bytes
int AppendHumanBytes(uint64_t bytes, char* output, int len);

// Append a human-readable printout of "num" to *str
void AppendNumberTo(string* str, uint64_t num);

string NumberToString(int64_t num);

string NumberToString(int num);

string NumberToString(uint32_t num);

string NumberToString(double num);

// Return a human-readable version of num.
// for num >= 10.000, prints "xxK"
// for num >= 10.000.000, prints "xxM"
// for num >= 10.000.000.000, prints "xxG"
string NumberToHumanString(int64_t num);

// Return a human-readable version of bytes
// ex: 1048576 -> 1.00 GB
string BytesToHumanString(uint64_t bytes);

bool ParseBoolean(const string& type, const string& value);

uint32_t ParseUint32(const string& value);

uint64_t ParseUint64(const string& value);

int ParseInt(const string& value);

double ParseDouble(const string& value);

size_t ParseSizeT(const string& value);

std::vector<int> ParseVectorInt(const string& value);

bool SerializeIntVector(const std::vector<int>& vec, string* value);

void ToUpper(string* str);

void ToLower(string* str);

string Hex2Str(const char* _str, unsigned int _len);

string Str2Hex(const char* _str, unsigned int _len);

bool Hex2Bin(const char* hex_str, string* bin_str);

bool Bin2Hex(const char* bin_str, string* hex_str);

void UrlEncode(const string& src_str, string* dst_str);

void UrlDecode(const string& src_str, string* dst_str);

bool StartsWith(const string& str, const string& prefix);

bool EndsWith(const string& str, const string& suffix);

// ----------------------------------------------------------------------
// HasSuffixString()
//    Return true if str ends in suffix.
// ----------------------------------------------------------------------
inline bool HasSuffixString(const string& str,
                            const string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// ----------------------------------------------------------------------
// HasPrefixString()
//    Check if a string begins with a given prefix.
// ----------------------------------------------------------------------
inline bool HasPrefixString(const string& str,
                            const string& prefix) {
  return str.size() >= prefix.size() &&
         str.compare(0, prefix.size(), prefix) == 0;
}

bool StripSuffix(string* str, const string& suffix);

bool StripPrefix(string* str, const string& prefix);

// Returns a version of 'src' where unprintable characters have been
// escaped using C-style escape sequences.
string CEscape(StringPiece src);

// Copies "source" to "dest", rewriting C-style escape sequences --
// '\n', '\r', '\\', '\ooo', etc -- to their ASCII equivalents.
//
// Errors: Sets the description of the first encountered error in
// 'error'. To disable error reporting, set 'error' to NULL.
//
// NOTE: Does not support \u or \U!
bool CUnescape(StringPiece source, string* dest, string* error);

// Removes any trailing whitespace from "*s".
void StripTrailingWhitespace(string* s);

// Removes leading ascii_isspace() characters.
// Returns number of characters removed.
size_t RemoveLeadingWhitespace(StringPiece* text);

// Removes trailing ascii_isspace() characters.
// Returns number of characters removed.
size_t RemoveTrailingWhitespace(StringPiece* text);

// Removes leading and trailing ascii_isspace() chars.
// Returns number of chars removed.
size_t RemoveWhitespaceContext(StringPiece* text);

// Consume a leading positive integer value.  If any digits were
// found, store the value of the leading unsigned number in "*val",
// advance "*s" past the consumed number, and return true.  If
// overflow occurred, returns false.  Otherwise, returns false.
bool ConsumeLeadingDigits(StringPiece* s, uint64* val);

// Consume a leading token composed of non-whitespace characters only.
// If *s starts with a non-zero number of non-whitespace characters, store
// them in *val, advance *s past them, and return true.  Else return false.
bool ConsumeNonWhitespace(StringPiece* s, StringPiece* val);

// If "*s" starts with "expected", consume it and return true.
// Otherwise, return false.
bool ConsumePrefix(StringPiece* s, StringPiece expected);

// If "*s" ends with "expected", remove it and return true.
// Otherwise, return false.
bool ConsumeSuffix(StringPiece* s, StringPiece expected);

string& Ltrim(string& str); // NOLINT

string& Rtrim(string& str); // NOLINT

string& Trim(string& str); // NOLINT

void Trim(std::vector<string>* str_list);

string TrimString(const string& str, const string& trim);

// Trims any whitespace from either end of the input string.  Returns where
// whitespace was found.
// The non-wide version has two functions:
// * TrimWhitespaceASCII()
//   This function is for ASCII strings and only looks for ASCII whitespace;
// Please choose the best one according to your usage.
// NOTE: Safe to use the same variable for both input and output.
enum TrimPositions {
  TRIM_NONE     = 0,
  TRIM_LEADING  = 1 << 0,
  TRIM_TRAILING = 1 << 1,
  TRIM_ALL      = TRIM_LEADING | TRIM_TRAILING,
};

bool TrimString(const string& input,
                const StringPiece& trim_chars,
                string* output);

TrimPositions TrimWhitespaceASCII(const string& input,
                                  TrimPositions positions,
                                  string* output);

// Deprecated. This function is only for backward compatibility and calls
// TrimWhitespaceASCII().
TrimPositions TrimWhitespace(const string& input,
                             TrimPositions positions,
                             string* output);

// Return lower-cased version of s.
string Lowercase(StringPiece s);

// Return upper-cased version of s.
string Uppercase(StringPiece s);

// Converts "^2ILoveYou!" to "i_love_you_". More specifically:
// - converts all non-alphanumeric characters to underscores
// - replaces each occurence of a capital letter (except the very
//   first character and if there is already an '_' before it) with '_'
//   followed by this letter in lower case
// - Skips leading non-alpha characters
// This method is useful for producing strings matching "[a-z][a-z0-9_]*"
// as required by OpDef.ArgDef.name. The resulting string is either empty or
// matches this regex.
string ArgDefCase(StringPiece s);

// Capitalize first character of each word in "*s".  "delimiters" is a
// set of characters that can be used as word boundaries.
void TitlecaseString(string* s, StringPiece delimiters);

// Replaces the first occurrence (if replace_all is false) or all occurrences
// (if replace_all is true) of oldsub in s with newsub.
string StringReplace(StringPiece s, StringPiece oldsub, StringPiece newsub,
                     bool replace_all);

void string_replace(const string &sub_str1,
                    const string &sub_str2, string *str);

// Join functionality
template <typename T>
string Join(const T& s, const char* sep);

// A variant of Join where for each element of "s", f(&dest_string, elem)
// is invoked (f is often constructed with a lambda of the form:
//   [](string* result, ElemType elem)
template <typename T, typename Formatter>
string Join(const T& s, const char* sep, Formatter f);

struct AllowEmpty {
  bool operator()(StringPiece sp) const { return true; }
};
struct SkipEmpty {
  bool operator()(StringPiece sp) const { return !sp.empty(); }
};
struct SkipWhitespace {
  bool operator()(StringPiece sp) const {
    RemoveTrailingWhitespace(&sp);
    return !sp.empty();
  }
};

string DebugString(const string& src);

void StringSplit(const string& str,
                 const string& delim,
                 std::vector<string>* result);

// |str| should not be in a multi-byte encoding like Shift-JIS or GBK in which
// the trailing byte of a multi-byte character can be in the ASCII range.
// UTF-8, and other single/multi-byte ASCII-compatible encodings are OK.
// Note: |c| must be in the ASCII range.
void SplitString(const string& str,
                 char c,
                 std::vector<string>* r);

// The same as SplitString, but use a substring delimiter instead of a char.
void SplitStringUsingSubstr(const string& str,
                            const string& s,
                            std::vector<string>* r);

// |str| should not be in a multi-byte encoding like Shift-JIS or GBK in which
// the trailing byte of a multi-byte character can be in the ASCII range.
// UTF-8, and other single/multi-byte ASCII-compatible encodings are OK.
// Note: |c| must be in the ASCII range.
void SplitStringDontTrim(const string& str,
                         char c,
                         std::vector<string>* r);

// WARNING: this uses whitespace as defined by the HTML5 spec. If you need
// a function similar to this but want to trim all types of whitespace, then
// factor this out into a function that takes a string containing the characters
// that are treated as whitespace.
//
// Splits the string along whitespace (where whitespace is the five space
// characters defined by HTML 5). Each contiguous block of non-whitespace
// characters is added to result.
void SplitStringAlongWhitespace(const string& str,
                                std::vector<string>* result);

// Splits |line| into key value pairs according to the given delimiters and
// removes whitespace leading each key and trailing each value. Returns true
// only if each pair has a non-empty key and value. |key_value_pairs| will
// include ("","") pairs for entries without |key_value_delimiter|.
typedef std::vector<std::pair<string, string> > StringPairs;
bool SplitStringIntoKeyValuePairs(const string& line,
                                  char key_value_delimiter,
                                  char key_value_pair_delimiter,
                                  StringPairs* key_value_pairs);

static const uint32_t MAX_PATH_LENGTH = 10240;
bool SplitPath(const string& path,
               std::vector<string>* element,
               bool* isdir = nullptr);

// Split strings using any of the supplied delimiters. For example:
// Split("a,b.c,d", ".,") would return {"a", "b", "c", "d"}.
std::vector<string> Split(StringPiece text, StringPiece delims);

template <typename Predicate>
std::vector<string> Split(StringPiece text, StringPiece delims, Predicate p);

// Split "text" at "delim" characters, and parse each component as
// an integer.  If successful, adds the individual numbers in order
// to "*result" and returns true.  Otherwise returns false.
bool SplitAndParseAsInts(StringPiece text, char delim,
                         std::vector<int32>* result);
bool SplitAndParseAsInts(StringPiece text, char delim,
                         std::vector<int64>* result);
bool SplitAndParseAsFloats(StringPiece text, char delim,
                           std::vector<float>* result);

// ------------------------------------------------------------------
// Implementation details below
template <typename T>
string Join(const T& s, const char* sep) {
  string result;
  bool first = true;
  for (const auto& x : s) {
    strings::StrAppend(&result, (first ? "" : sep), x);
    first = false;
  }
  return result;
}

template <typename T>
class Formatter {
 public:
  Formatter(std::function<void(string*, T)> f) : f_(f) {}
  void operator()(string* out, const T& t) { f_(out, t); }

 private:
  std::function<void(string*, T)> f_;
};

template <typename T, typename Formatter>
string Join(const T& s, const char* sep, Formatter f) {
  string result;
  bool first = true;
  for (const auto& x : s) {
    if (!first) {
      result.append(sep);
    }
    f(&result, x);
    first = false;
  }
  return result;
}

// ----------------------------------------------------------------------
// Join()
//    These methods concatenate a range of components into a C++ string, using
//    the C-string "delim" as a separator between components.
// ----------------------------------------------------------------------
template <typename Iterator>
void Join(Iterator start, Iterator end,
          const char* delim, string* result) {
  for (Iterator it = start; it != end; ++it) {
    if (it != start) {
      result->append(delim);
    }
    strings::StrAppend(result, *it);
  }
}

inline std::vector<string> Split(StringPiece text, StringPiece delims) {
  return Split(text, delims, AllowEmpty());
}

template <typename Predicate>
std::vector<string> Split(StringPiece text, StringPiece delims, Predicate p) {
  std::vector<string> result;
  size_t token_start = 0;
  if (!text.empty()) {
    for (size_t i = 0; i < text.size() + 1; i++) {
      if ((i == text.size()) || (delims.find(text[i]) != StringPiece::npos)) {
        StringPiece token(text.data() + token_start, i - token_start);
        if (p(token)) {
          result.push_back(token.ToString());
        }
        token_start = i + 1;
      }
    }
  }
  return result;
}

inline std::vector<string> Split(StringPiece text, char delim) {
  return Split(text, StringPiece(&delim, 1));
}

template <typename Predicate>
std::vector<string> Split(StringPiece text, char delims, Predicate p) {
  return Split(text, StringPiece(&delims, 1), p);
}

}  // namespace str_util
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_STR_UTIL_H_