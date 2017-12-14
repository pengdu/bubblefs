/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

// apollo/modules/common/util/util.h
// apollo/modules/common/util/string_util.h

/**
 * @file
 * @brief Some string util functions.
 */

#ifndef BUBBLEFS_UTILS_APOLLO_STRING_UTIL_H_
#define BUBBLEFS_UTILS_APOLLO_STRING_UTIL_H_

#include <string.h>
#include <memory>
#include <sstream>
#include <string>

/**
 * @namespace apollo::common::util
 * @brief apollo::common::util
 */
namespace bubblefs {
namespace myapollo {
namespace common {
namespace util {
  
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}  

struct PairHash {
  template <typename T, typename U>
  size_t operator()(const std::pair<T, U>& pair) const {
    return std::hash<T>()(pair.first) ^ std::hash<U>()(pair.second);
  }
};
  
// ----------------------------------------------------------------------
// Join()
//    These methods concatenate a range of components into a C++ string, using
//    the C-string "delim" as a separator between components.
// ----------------------------------------------------------------------
template <typename Iterator>
void Join(Iterator start, Iterator end,
          const char* delim, std::string* result) {
  for (Iterator it = start; it != end; ++it) {
    if (it != start) {
      result->append(delim);
    }
    result->append((*it).data(), (*it).size());
  }
}
  
/**
 * @brief Check if a string ends with a pattern.
 * @param ori The original string. To see if it ends with a specified pattern.
 * @param pat The target pattern. To see if the original string ends with it.
 * @return Whether the original string ends with the specified pattern.
 */
inline bool EndWith(const std::string& ori, const std::string& pat) {
  return ((ori.size() >= pat.size()) &&
          (memcmp(ori.data() + (ori.size() - pat.size()), pat.data(), pat.size()) == 0));
}

template <typename T>
std::string Print(const T& val) {
  std::ostringstream oss;
  oss << val;
  return oss.str();
}

/**
 * @brief Make arrays, conatiners and iterators printable.
 *
 * Usage:
 *   vector<int> vec = {1, 2, 3};
 *   std::cout << PrintIter(vec);
 *   std::cout << PrintIter(vec, ",");
 *   std::cout << PrintIter(vec.begin(), vec.end());
 *   std::cout << PrintIter(vec.begin(), vec.end(), "|");
 *
 *   int array[] = {1, 2, 3};
 *   std::cout << PrintIter(array);
 *   std::cout << PrintIter(array, "|");
 *   std::cout << PrintIter(array + 0, array + 10, "|");
 */
template <typename Iter>
std::string PrintIter(const Iter& begin, const Iter& end,
                      const std::string& delimiter = " ") {
  std::string result;
  Join(begin, end, delimiter.c_str(), &result);
  return result;
}

template <typename Container>
std::string PrintIter(const Container& container,
                      const std::string& delimiter = " ") {
  return PrintIter(container.begin(), container.end(), delimiter);
}

template <typename T, int length>
std::string PrintIter(T (&array)[length], T* end,
                      const std::string& delimiter = " ") {
  std::string result;
  Join(array, end, delimiter.c_str(), &result);
  return result;
}

template <typename T, int length>
std::string PrintIter(T (&array)[length], const std::string& delimiter = " ") {
  return PrintIter(array, array + length, delimiter);
}

/**
 * @brief Make conatiners and iterators printable. Similar to PrintIter but
 *        output the DebugString().
 */
template <typename Iter>
std::string PrintDebugStringIter(const Iter& begin, const Iter& end,
                                 const std::string& delimiter = " ") {
  std::string result;
  for (auto iter = begin; iter != end; ++iter) {
    if (iter == begin) {
      result.append(iter->DebugString());
    } else {
      result.append(delimiter).append(iter->DebugString());
    }
  }
  return result;
}

template <typename Container>
std::string PrintDebugStringIter(const Container& container,
                                 const std::string& delimiter = " ") {
  return PrintDebugStringIter(container.begin(), container.end(), delimiter);
}

}  // namespace util
}  // namespace common
}  // namespace myapollo
}  // namespace bubblefs

/*
template <typename A, typename B>
std::ostream& operator<<(std::ostream& os, std::pair<A, B>& p) {
  return os << "first: " << p.first << ", second: " << p.second;
}
*/

#endif  // BUBBLEFS_UTILS_APOLLO_STRING_UTIL_H_