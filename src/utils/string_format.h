// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/common/base/string_format.h

#ifndef BUBBLEFS_UTILS_STRING_FORMAT_H_
#define BUBBLEFS_UTILS_STRING_FORMAT_H_

#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <string>

namespace bubblefs {
namespace bdcommon {

size_t StringFormatAppendVA(std::string* dst, const char* format, va_list ap);

size_t StringFormatAppend(std::string* dst, const char* format, ...);

size_t StringFormatTo(std::string* dst, const char* format, ...);

std::string StringFormat(const char* format, ...);

} // namespace bdcommon
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_STRING_FORMAT_H_