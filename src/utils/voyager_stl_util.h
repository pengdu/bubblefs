// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/util/stl_util.h

#ifndef BUBBLEFS_UTILS_VOYAGER_STL_UTIL_H_
#define BUBBLEFS_UTILS_VOYAGER_STL_UTIL_H_

#include <string>

namespace bubblefs {
namespace myvoyager {

inline char* string_as_array(std::string* str) {
  return str->empty() ? nullptr : &*str->begin();
}

template <typename ForwardIterator>
void STLDeleteContainerPointers(ForwardIterator begin, ForwardIterator end) {
  while (begin != end) {
    ForwardIterator temp = begin;
    ++begin;
    delete *temp;
  }
}

template <typename T>
void STLDeleteElements(T* container) {
  if (!container) return;
  STLDeleteContainerPointers(container->begin(), container->end());
  container->clear();
}

template <typename T>
void STLDeleteValues(T* v) {
  if (!v) return;
  for (typename T::iterator it = v->begin(); it != v->end(); ++it) {
    delete it->second;
  }
  v->clear();
}

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_STL_UTIL_H_