// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/common/base/string_format.h

#ifndef BUBBLEFS_UTILS_BDCOM_STRING_FORMAT_H_
#define BUBBLEFS_UTILS_BDCOM_STRING_FORMAT_H_

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <iomanip>
#include <sstream>
#include <string>

namespace bubblefs {
namespace mybdcom {

size_t StringFormatAppendVA(std::string* dst, const char* format, va_list ap);

size_t StringFormatAppend(std::string* dst, const char* format, ...);

size_t StringFormatTo(std::string* dst, const char* format, ...);

std::string StringFormat(const char* format, ...);

// caffe/include/caffe/util/format.hpp
inline std::string format_int(int n, int numberOfLeadingZeros = 0 ) {
  std::ostringstream s;
  s << std::setw(numberOfLeadingZeros) << std::setfill('0') << n;
  return s.str();
}

} // namespace mybdcom
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOM_STRING_FORMAT_H_