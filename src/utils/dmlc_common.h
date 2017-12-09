/*!
 *  Copyright (c) 2015 by Contributors
 * \file common.h
 * \brief defines some common utility function.
 */

// dmlc-core/include/dmlc/base.h
// dmlc-core/include/dmlc/common.h

#ifndef BUBBLEFS_UTILS_DMLC_COMMON_H_
#define BUBBLEFS_UTILS_DMLC_COMMON_H_

#include <string>
#include <sstream>
#include <vector>

namespace bubblefs {
namespace mydmlc {

/*!
 * \brief safely get the beginning address of a vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template<typename T>
inline T *BeginPtr(std::vector<T> &vec) {  // NOLINT(*)
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*!
 * \brief get the beginning address of a const vector
 * \param vec input vector
 * \return beginning address of a vector
 */
template<typename T>
inline const T *BeginPtr(const std::vector<T> &vec) {
  if (vec.size() == 0) {
    return NULL;
  } else {
    return &vec[0];
  }
}
/*!
 * \brief get the beginning address of a string
 * \param str input string
 * \return beginning address of a string
 */
inline char* BeginPtr(std::string &str) {  // NOLINT(*)
  if (str.length() == 0) return NULL;
  return &str[0];
}
/*!
 * \brief get the beginning address of a const string
 * \param str input string
 * \return beginning address of a string
 */
inline const char* BeginPtr(const std::string &str) {
  if (str.length() == 0) return NULL;
  return &str[0];
}  
  
/*!
 * \brief Split a string by delimiter
 * \param s String to be splitted.
 * \param delim The delimiter.
 * \return a splitted vector of strings.
 */
inline std::vector<std::string> Split(const std::string& s, char delim) {
  std::string item;
  std::istringstream is(s);
  std::vector<std::string> ret;
  while (std::getline(is, item, delim)) {
    ret.push_back(item);
  }
  return ret;
}

/*!
 * \brief hash an object and combines the key with previous keys
 */
template<typename T>
inline size_t HashCombine(size_t key, const T& value) {
  std::hash<T> hash_func;
  return key ^ (hash_func(value) + 0x9e3779b9 + (key << 6) + (key >> 2));
}

/*!
 * \brief specialize for size_t
 */
template<>
inline size_t HashCombine<size_t>(size_t key, const size_t& value) {
  return key ^ (value + 0x9e3779b9 + (key << 6) + (key >> 2));
}

}  // namespace mydmlc
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_DMLC_COMMON_H_