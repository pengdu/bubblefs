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
// chromium/base/stl_util.h

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

namespace bubblefs {
namespace gtl {

template <class T, class A>
T Joinify(const A &begin, const A &end, const T &t)
{
  T result;
  for (A it = begin; it != end; it++) {
    if (!result.empty())
      result.append(t);
    result.append(*it);
  }
  return result;
}

// Clears internal memory of an STL object.
// STL clear()/reserve(0) does not always free internal memory allocated
// This function uses swap/destructor to ensure the internal memory is freed.
template<class T>
void STLClearObject(T* obj) {
  T tmp;
  tmp.swap(*obj);
  // Sometimes "T tmp" allocates objects with memory (arena implementation?).
  // Hence using additional reserve(0) even if it doesn't always work.
  obj->reserve(0);
}
  
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
// be null-terminated. Returns nullptr for an empty string. If not non-null,
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
  return str->empty() ? nullptr : &*str->begin();
}

// Returns the T* array for the given vector, or nullptr if the vector was empty.
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
  return v->empty() ? nullptr : &*v->begin();
#endif
}
// vector_as_array overload for const std::vector<>.
template <typename T, typename Allocator>
inline const T* vector_as_array(const std::vector<T, Allocator>* v) {
#if defined NDEBUG && !defined _GLIBCXX_DEBUG
  return &*v->begin();
#else
  return v->empty() ? nullptr : &*v->begin();
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

// For a range within a container of pointers, calls delete (non-array version)
// on these pointers.
// NOTE: for these three functions, we could just implement a DeleteObject
// functor and then call for_each() on the range and functor, but this
// requires us to pull in all of algorithm.h, which seems expensive.
// For hash_[multi]set, it is important that this deletes behind the iterator
// because the hash_set may call the hash function on the iterator when it is
// advanced, which could result in the hash function trying to deference a
// stale pointer.
template <class ForwardIterator>
void STLDeleteContainerPointers(ForwardIterator begin, ForwardIterator end) {
  while (begin != end) {
    ForwardIterator temp = begin;
    ++begin;
    delete *temp;
  }
}

// For a range within a container of pairs, calls delete (non-array version) on
// BOTH items in the pairs.
// NOTE: Like STLDeleteContainerPointers, it is important that this deletes
// behind the iterator because if both the key and value are deleted, the
// container may call the hash function on the iterator when it is advanced,
// which could result in the hash function trying to dereference a stale
// pointer.
template <class ForwardIterator>
void STLDeleteContainerPairPointers(ForwardIterator begin,
                                    ForwardIterator end) {
  while (begin != end) {
    ForwardIterator temp = begin;
    ++begin;
    delete temp->first;
    delete temp->second;
  }
}

// For a range within a container of pairs, calls delete (non-array version) on
// the FIRST item in the pairs.
// NOTE: Like STLDeleteContainerPointers, deleting behind the iterator.
template <class ForwardIterator>
void STLDeleteContainerPairFirstPointers(ForwardIterator begin,
                                         ForwardIterator end) {
  while (begin != end) {
    ForwardIterator temp = begin;
    ++begin;
    delete temp->first;
  }
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

// The following classes provide a convenient way to delete all elements or
// values from STL containers when they goes out of scope.  This greatly
// simplifies code that creates temporary objects and has multiple return
// statements.  Example:
//
// vector<MyProto *> tmp_proto;
// STLElementDeleter<vector<MyProto *> > d(&tmp_proto);
// if (...) return false;
// ...
// return success;

// Given a pointer to an STL container this class will delete all the element
// pointers when it goes out of scope.
template<class T>
class STLElementDeleter {
 public:
  STLElementDeleter<T>(T* container) : container_(container) {}
  ~STLElementDeleter<T>() { STLDeleteElements(container_); }

 private:
  T* container_;
};

// Given a pointer to an STL container this class will delete all the value
// pointers when it goes out of scope.
template<class T>
class STLValueDeleter {
 public:
  STLValueDeleter<T>(T* container) : container_(container) {}
  ~STLValueDeleter<T>() { STLDeleteValues(container_); }

 private:
  T* container_;
};

// Sorts and removes duplicates from a sequence container.
template <typename T>
inline void STLSortAndRemoveDuplicates(T* v) {
  std::sort(v->begin(), v->end());
  v->erase(std::unique(v->begin(), v->end()), v->end());
}

// Counts the number of instances of val in a container.
template <typename Container, typename T>
typename std::iterator_traits<
    typename Container::const_iterator>::difference_type
STLCount(const Container& container, const T& val) {
  return std::count(container.begin(), container.end(), val);
}

// Test to see if a set, map, hash_set or hash_map contains a particular key.
// Returns true if the key is in the collection.
template <typename Collection, typename Key>
bool ContainsKey(const Collection& collection, const Key& key) {
  return collection.find(key) != collection.end();
}

// Test to see if a collection like a vector contains a particular value.
// Returns true if the value is in the collection.
template <typename Collection, typename Value>
bool ContainsValue(const Collection& collection, const Value& value) {
  return std::find(collection.begin(), collection.end(), value) !=
      collection.end();
}

// Returns true if the container is sorted.
template <typename Container>
bool STLIsSorted(const Container& cont) {
  // Note: Use reverse iterator on container to ensure we only require
  // value_type to implement operator<.
  return std::adjacent_find(cont.rbegin(), cont.rend(),
                            std::less<typename Container::value_type>())
      == cont.rend();
}


// Returns a new ResultType containing the difference of two sorted containers.
template <typename ResultType, typename Arg1, typename Arg2>
ResultType STLSetDifference(const Arg1& a1, const Arg2& a2) {
  //DCHECK(STLIsSorted(a1));
  //DCHECK(STLIsSorted(a2));
  ResultType difference;
  std::set_difference(a1.begin(), a1.end(),
                      a2.begin(), a2.end(),
                      std::inserter(difference, difference.end()));
  return difference;
}

// Returns a new ResultType containing the union of two sorted containers.
template <typename ResultType, typename Arg1, typename Arg2>
ResultType STLSetUnion(const Arg1& a1, const Arg2& a2) {
  //DCHECK(STLIsSorted(a1));
  //DCHECK(STLIsSorted(a2));
  ResultType result;
  std::set_union(a1.begin(), a1.end(),
                 a2.begin(), a2.end(),
                 std::inserter(result, result.end()));
  return result;
}

// Returns a new ResultType containing the intersection of two sorted
// containers.
template <typename ResultType, typename Arg1, typename Arg2>
ResultType STLSetIntersection(const Arg1& a1, const Arg2& a2) {
  //DCHECK(STLIsSorted(a1));
  //DCHECK(STLIsSorted(a2));
  ResultType result;
  std::set_intersection(a1.begin(), a1.end(),
                        a2.begin(), a2.end(),
                        std::inserter(result, result.end()));
  return result;
}

// Returns true if the sorted container |a1| contains all elements of the sorted
// container |a2|.
template <typename Arg1, typename Arg2>
bool STLIncludes(const Arg1& a1, const Arg2& a2) {
  //DCHECK(STLIsSorted(a1));
  //DCHECK(STLIsSorted(a2));
  return std::includes(a1.begin(), a1.end(),
                       a2.begin(), a2.end());
}

}  // namespace gtl
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_STL_UTIL_H_