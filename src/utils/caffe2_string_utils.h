/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/core/common.h
// caffe2/caffe2/core/logging.h
// caffe2/caffe2/utils/string_utils.h

#ifndef BUBBLEFS_UTILS_CAFFE2_STRING_UTILS_H_
#define BUBBLEFS_UTILS_CAFFE2_STRING_UTILS_H_

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace bubblefs {
namespace mycaffe2 {

// Note: template functions should be defined in namespace.  
  
/**
 * A utility to allow one to show log info to stderr after the program starts.
 *
 * This is similar to calling GLOG's --logtostderr, or setting caffe2_log_level
 * to smaller than INFO. You are recommended to only use this in a few sparse
 * cases, such as when you want to write a tutorial or something. Normally, use
 * the commandline flags to set the log level.
 */

template <typename T>
inline void MakeStringInternal(std::stringstream& ss, const T& t) {
  ss << t;
}

template <typename T, typename... Args>
inline void
MakeStringInternal(std::stringstream& ss, const T& t, const Args&... args) {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
std::string MakeString(const Args&... args) {
  std::stringstream ss;
  MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

// Specializations for already-a-string types.
template <>
inline std::string MakeString(const std::string& str) {
  return str;
}
inline std::string MakeString(const char* c_str) {
  return std::string(c_str);
}

template <class Container>
inline std::string JoinToString(const std::string& delimiter, const Container& v) {
  std::stringstream s;
  s << typeid(v).name() << "{";
  int cnt = static_cast<int64_t>(v.size()) - 1;
  for (auto i = v.begin(); i != v.end(); ++i, --cnt) {
    s << (*i) << (cnt ? delimiter : "");
  }
  s << "}";
  return s.str();
}  

template <typename T>
std::string to_string(T value)
{
  std::ostringstream os;
  os << value;
  return os.str();
}

inline int stoi(const std::string& str) {
  std::stringstream ss;
  int n = 0;
  ss << str;
  ss >> n;
  return n;
}

inline double stod(const std::string& str, std::size_t* pos = 0) {
  std::stringstream ss;
  ss << str;
  double val = 0;
  ss >> val;
  if (pos) {
    if (ss.tellg() == -1) {
      *pos = str.size();
    } else {
      *pos = ss.tellg();
    }
  }
  return val;
}
  
std::vector<std::string> split(char separator, const std::string& string);
size_t editDistance(
  const std::string& s1, const std::string& s2, size_t max_distance = 0);

int32_t editDistanceHelper(const char* s1,
  size_t s1_len,
  const char* s2,
  size_t s2_len,
  std::vector<size_t> &current,
  std::vector<size_t> &previous,
  std::vector<size_t> &previous1,
  size_t max_distance);

}  // namespace mycaffe2
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_STRING_UTILS_H_