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

// tensorflow/tensorflow/core/lib/strings/stringprintf.cc

#include "utils/stringprintf.h"
#include <errno.h>
#include <stdarg.h>  // For va_list and related operations
#include <stdio.h>   // MSVC requires this for _vsnprintf
#include "platform/logging.h"

namespace bubblefs {
namespace strings {

#ifdef COMPILER_MSVC
enum { IS_COMPILER_MSVC = 1 };
#else
enum { IS_COMPILER_MSVC = 0 };
#endif

void Appendv(string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer
  static const int kSpaceLength = 1024;
  char space[kSpaceLength];

  // It's possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(space, kSpaceLength, format, backup_ap);
  va_end(backup_ap);

  if (result < kSpaceLength) {
    if (result >= 0) {
      // Normal case -- everything fit.
      dst->append(space, result);
      return;
    }

    if (IS_COMPILER_MSVC) {
      // Error or MSVC running out of space.  MSVC 8.0 and higher
      // can be asked about space needed with the special idiom below:
      va_copy(backup_ap, ap);
      result = vsnprintf(nullptr, 0, format, backup_ap);
      va_end(backup_ap);
    }

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  int length = result + 1;
  char* buf = new char[length];

  // Restore the va_list before we use it again
  va_copy(backup_ap, ap);
  result = vsnprintf(buf, length, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < length) {
    // It fit
    dst->append(buf, result);
  }
  delete[] buf;
}

string Printf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  string result;
  Appendv(&result, format, ap);
  va_end(ap);
  return result;
}

const string& SPrintf(string* dst, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  dst->clear();
  Appendv(dst, format, ap);
  va_end(ap);
  return *dst;
}

void Appendf(string* dst, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  Appendv(dst, format, ap);
  va_end(ap);
}

// Max arguments supported by StringPrintVector
const int kStringPrintfVectorMaxArgs = 32;

// An empty block of zero for filler arguments.  This is const so that if
// printf tries to write to it (via %n) then the program gets a SIGSEGV
// and we can fix the problem or protect against an attack.
static const char string_printf_empty_block[256] = { '\0' };

string PrintfVector(const char* format, const std::vector<string>& v) {
  CHECK_LE(v.size(), kStringPrintfVectorMaxArgs)
      << "StringPrintfVector currently only supports up to "
      << kStringPrintfVectorMaxArgs << " arguments. "
      << "Feel free to add support for more if you need it.";

  // Add filler arguments so that bogus format+args have a harder time
  // crashing the program, corrupting the program (%n),
  // or displaying random chunks of memory to users.

  const char* cstr[kStringPrintfVectorMaxArgs];
  for (int i = 0; i < v.size(); ++i) {
    cstr[i] = v[i].c_str();
  }
  for (int i = v.size(); i < TF_ARRAYSIZE(cstr); ++i) {
    cstr[i] = &string_printf_empty_block[0];
  }

  // I do not know any way to pass kStringPrintfVectorMaxArgs arguments,
  // or any way to build a va_list by hand, or any API for printf
  // that accepts an array of arguments.  The best I can do is stick
  // this COMPILE_ASSERT right next to the actual statement.

  COMPILE_ASSERT(kStringPrintfVectorMaxArgs == 32, arg_count_mismatch);
  return Printf(format,
                cstr[0], cstr[1], cstr[2], cstr[3], cstr[4],
                cstr[5], cstr[6], cstr[7], cstr[8], cstr[9],
                cstr[10], cstr[11], cstr[12], cstr[13], cstr[14],
                cstr[15], cstr[16], cstr[17], cstr[18], cstr[19],
                cstr[20], cstr[21], cstr[22], cstr[23], cstr[24],
                cstr[25], cstr[26], cstr[27], cstr[28], cstr[29],
                cstr[30], cstr[31]);
}

}  // namespace strings
}  // namespace bubblefs