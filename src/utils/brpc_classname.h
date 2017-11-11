// Copyright (c) 2011 Baidu, Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// brpc/src/butil/class_name.h

// Get name of a class. For example, class_name<T>() returns the name of T
// (with namespace prefixes). This is useful in template classes.

#ifndef BUBBLEFS_UTILS_BRPC_CLASS_NAME_H_
#define BUBBLEFS_UTILS_BRPC_CLASS_NAME_H_

#include <string>
#include <typeinfo>

namespace bubblefs {
namespace mybrpc {

std::string demangle(const char* name);

namespace detail {
template <typename T> struct ClassNameHelper { static std::string name; };
template <typename T> std::string ClassNameHelper<T>::name = demangle(typeid(T).name());
}

// Get name of class |T|, in std::string.
template <typename T> const std::string& class_name_str() {
    // We don't use static-variable-inside-function because before C++11
    // local static variable is not guaranteed to be thread-safe.
    return detail::ClassNameHelper<T>::name;
}

// Get name of class |T|, in const char*.
// Address of returned name never changes.
template <typename T> const char* class_name() {
    return class_name_str<T>().c_str();
}

// Get typename of |obj|, in std::string
template <typename T> std::string class_name_str(T const& obj) {
    return demangle(typeid(obj).name());
}

}  // namespace mybrpc
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_BRPC_CLASS_NAME_H_