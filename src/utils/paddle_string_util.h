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

// Paddle/paddle/utils/StringUtil.h

#ifndef BUBBLEFS_UTILS_PADDLE_STRING_UTIL_H_
#define BUBBLEFS_UTILS_PADDLE_STRING_UTIL_H_

#include <sstream>
#include <string>
#include <vector>
#include "platform/base_error.h"

namespace bubblefs {
namespace mypaddle {

namespace str {
/// test whether a string ends with another string
bool endsWith(const std::string& str, const std::string& ext);

bool startsWith(const std::string& str, const std::string& prefix);

/**
 * Use sep to split str into pieces.
 * If str is empty, *pieces will be empty.
 * If str ends with sep, the last piece will be an empty string.
 */
void split(const std::string& str, char sep, std::vector<std::string>* pieces);

/**
 * Cast string to type T with status.
 *
 * @param [in] s input string.
 * @param [out] ok status, return true if there is no error in casting. Set
 *              nullptr if user don't care error at all.
 * @return result of casting. If error occurred, a default value of T() will
 *         return.
 */
template <class T>
inline T toWithStatus(const std::string& s, bool* ok = nullptr) {
  std::istringstream sin(s);
  T v;
  sin >> v;
  if (ok) {
    *ok = sin.eof() && !sin.fail();
  }
  return v;
}

/**
 * Cast type T to string with status.
 *
 * @param [in] v input value of type T.
 * @param [out] ok status, return true if there is no error in casting. Set
 *              nullptr if user don't care error at all.
 * @return result of casting. If error occurred, a empty string will be
 *              returned.
 */
template <class T>
inline std::string toWithStatus(const T v, bool* ok = nullptr) {
  std::ostringstream sout;
  sout << v;
  if (ok) {
    *ok = !sout.fail();
  }
  return sout.str();
}

/// Convert string to type T. It makes sure all the characters in s are used.
/// Otherwise it will abort.
///
/// @tparam T type of return
/// @param s string input.
template <class T>
inline T to(const std::string& s) {
  bool ok;
  T v = toWithStatus<T>(s, &ok);
  PANIC_ENFORCE(ok, "Cannot convert s(%s) to type", s.c_str());
  return v;
}

/// Convert type T to string.
///
/// @tparam T type of input value
/// @param v input value of type T
template <class T>
std::string to_string(T v) {
  bool ok;
  std::string s = toWithStatus<T>(v, &ok);
  PANIC_ENFORCE(ok, "Cannot convert v(%s) to type std::string", v.c_str());
  return s;
}

}  // namespace str

#undef DEFINE_STRING_CONVERSION

}  // namespace mypaddle
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PADDLE_STRING_UTIL_H_