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

// Paddle/paddle/utils/Util.h
// tensorflow/tensorflow/core/lib/gtl/stl_util.h
// ceph/src/include/types.h

// This file provides utility functions for use with STL

#ifndef BUBBLEFS_UTILS_STL_UTIL_H_
#define BUBBLEFS_UTILS_STL_UTIL_H_

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <algorithm>
#include <deque>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

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

namespace bubblefs {
namespace gtl {
  
/**
 * find the value given a key k from container c.
 * If the key can be found, the value is stored in *value
 * return true if the key can be found. false otherwise.
 */
template <class K, class V, class C>
bool MapGet(const K& k, const C& c, V* value) {
  auto it = c.find(k);
  if (it != c.end()) {
    *value = it->second;
    return true;
  } else {
    return false;
  }
}

template <class Container, class T>
static bool Contains(const Container& container, const T& val) {
  return std::find(container.begin(), container.end(), val) != container.end();
}

/**
 * pop and get the front element of a container
 */
template <typename Container>
typename Container::value_type pop_get_front(Container& c) {
  typename Container::value_type v;
  std::swap(v, c.front());
  c.pop_front();
  return v;
}

/**
 * sort and unique ids vector.
 */
inline void UniqueIds(std::vector<uint32_t>& ids) {
  std::sort(ids.begin(), ids.end());
  auto endpos = std::unique(ids.begin(), ids.end());
  ids.erase(endpos, ids.end());
}

// Returns a mutable char* pointing to a string's internal buffer, which may not
// be null-terminated. Returns NULL for an empty string. If not non-null,
// writing through this pointer will modify the string.
//
// string_as_array(&str)[i] is valid for 0 <= i < str.size() until the
// next call to a string method that invalidates iterators.
//
// In C++11 you may simply use &str[0] to get a mutable char*.
//
// Prior to C++11, there was no standard-blessed way of getting a mutable
// reference to a string's internal buffer. The requirement that string be
// contiguous is officially part of the C++11 standard [string.require]/5.
// According to Matt Austern, this should already work on all current C++98
// implementations.
inline char* string_as_array(std::string* str) {
  return str->empty() ? NULL : &*str->begin();
}

// Returns the T* array for the given vector, or NULL if the vector was empty.
//
// Note: If you know the array will never be empty, you can use &*v.begin()
// directly, but that is may dump core if v is empty. This function is the most
// efficient code that will work, taking into account how our STL is actually
// implemented. THIS IS NON-PORTABLE CODE, so use this function instead of
// repeating the nonportable code everywhere. If our STL implementation changes,
// we will need to change this as well.
template <typename T, typename Allocator>
inline T* vector_as_array(std::vector<T, Allocator>* v) {
#if defined NDEBUG && !defined _GLIBCXX_DEBUG
  return &*v->begin();
#else
  return v->empty() ? NULL : &*v->begin();
#endif
}
// vector_as_array overload for const std::vector<>.
template <typename T, typename Allocator>
inline const T* vector_as_array(const std::vector<T, Allocator>* v) {
#if defined NDEBUG && !defined _GLIBCXX_DEBUG
  return &*v->begin();
#else
  return v->empty() ? NULL : &*v->begin();
#endif
}

// Like str->resize(new_size), except any new characters added to "*str" as a
// result of resizing may be left uninitialized, rather than being filled with
// '0' bytes. Typically used when code is then going to overwrite the backing
// store of the string with known data. Uses a Google extension to ::string.
inline void STLStringResizeUninitialized(std::string* s, size_t new_size) {
#if __google_stl_resize_uninitialized_string
  s->resize_uninitialized(new_size);
#else
  s->resize(new_size);
#endif
}

// Calls delete (non-array version) on the SECOND item (pointer) in each pair in
// the range [begin, end).
//
// Note: If you're calling this on an entire container, you probably want to
// call STLDeleteValues(&container) instead, or use ValueDeleter.
template <typename ForwardIterator>
void STLDeleteContainerPairSecondPointers(ForwardIterator begin,
                                          ForwardIterator end) {
  while (begin != end) {
    ForwardIterator temp = begin;
    ++begin;
    delete temp->second;
  }
}

// Deletes all the elements in an STL container and clears the container. This
// function is suitable for use with a vector, set, hash_set, or any other STL
// container which defines sensible begin(), end(), and clear() methods.
//
// If container is NULL, this function is a no-op.
template <typename T>
void STLDeleteElements(T* container) {
  if (!container) return;
  auto it = container->begin();
  while (it != container->end()) {
    auto temp = it;
    ++it;
    delete *temp;
  }
  container->clear();
}

// Given an STL container consisting of (key, value) pairs, STLDeleteValues
// deletes all the "value" components and clears the container. Does nothing in
// the case it's given a NULL pointer.
template <typename T>
void STLDeleteValues(T* container) {
  if (!container) return;
  auto it = container->begin();
  while (it != container->end()) {
    auto temp = it;
    ++it;
    delete temp->second;
  }
  container->clear();
}

// Sorts and removes duplicates from a sequence container.
template <typename T>
inline void STLSortAndRemoveDuplicates(T* v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}
}  // namespace gtl
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_STL_UTIL_H_