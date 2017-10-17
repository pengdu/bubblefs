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

// caffe2/caffe2/core/logging.h
// ceph/src/include/types.h

#ifndef BUBBLEFS_UTILS_STL_SSTREAM_H_
#define BUBBLEFS_UTILS_STL_SSTREAM_H_

#include <deque>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bubblefs {

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

// -- io helpers --
// Forward declare all the I/O helpers so strict ADL can find them in
// the case of containers of containers. I'm tempted to abstract this
// stuff using template templates like I did for denc.

template<class A, class B>
inline std::ostream& operator<<(std::ostream&out, const std::pair<A,B>& v);
template<class A, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::vector<A,Alloc>& v);
template<class A, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::deque<A,Alloc>& v);
template<class A, class B, class C>
inline std::ostream& operator<<(std::ostream&out, const std::tuple<A, B, C> &t);
template<class A, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::list<A,Alloc>& ilist);
template<class A, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::set<A, Comp, Alloc>& iset);
template<class A, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::multiset<A,Comp,Alloc>& iset);
template<class A, class B, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::map<A,B,Comp,Alloc>& m);
template<class A, class B, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::multimap<A,B,Comp,Alloc>& m);

template<class A, class B>
inline std::ostream& operator<<(std::ostream& out, const std::pair<A,B>& v) {
  return out << "pair(" << v.first << ", " << v.second << ")";
}

template<class A, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::vector<A,Alloc>& v) {
  out << "vector[";
  for (auto p = v.begin(); p != v.end(); ++p) {
    if (p != v.begin()) out << ", ";
    out << *p;
  }
  out << "]";
  return out;
}

template<class A, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::deque<A,Alloc>& v) {
  out << "deque<";
  for (auto p = v.begin(); p != v.end(); ++p) {
    if (p != v.begin()) out << ", ";
    out << *p;
  }
  out << ">";
  return out;
}

template<class A, class B, class C>
inline std::ostream& operator<<(std::ostream& out, const std::tuple<A, B, C> &t) {
  out << std::get<0>(t) <<"," << std::get<1>(t) << "," << std::get<2>(t);
  return out;
}

template<class A, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::list<A,Alloc>& ilist) {
  out << "list[";
  for (auto it = ilist.begin();
       it != ilist.end();
       ++it) {
    if (it != ilist.begin()) out << ", ";
    out << *it;
  }
  out << "]";
  return out;
}

template<class A, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::set<A, Comp, Alloc>& iset) {
  out << "set(";
  for (auto it = iset.begin();
       it != iset.end();
       ++it) {
    if (it != iset.begin()) out << ", ";
    out << *it;
  }
  out << ")";
  return out;
}

template<class A, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::multiset<A,Comp,Alloc>& iset) {
  out << "multiset(";
  for (auto it = iset.begin();
       it != iset.end();
       ++it) {
    if (it != iset.begin()) out << ", ";
    out << *it;
  }
  out << ")";
  return out;
}

template<class A, class B, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::map<A,B,Comp,Alloc>& m)
{
  out << "map{";
  for (auto it = m.begin();
       it != m.end();
       ++it) {
    if (it != m.begin()) out << ", ";
    out << it->first << "=" << it->second;
  }
  out << "}";
  return out;
}

template<class A, class B, class Comp, class Alloc>
inline std::ostream& operator<<(std::ostream& out, const std::multimap<A,B,Comp,Alloc>& m)
{
  out << "multimap{{";
  for (auto it = m.begin();
       it != m.end();
       ++it) {
    if (it != m.begin()) out << ", ";
    out << it->first << "=" << it->second;
  }
  out << "}}";
  return out;
}

} // namespace bubblefs

#endif // BUBBLEFS_UTILS_STL_SSTREAM_H_