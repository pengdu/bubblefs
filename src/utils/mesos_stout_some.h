// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// mesos/3rdparty/stout/include/stout/none.hpp
// mesos/3rdparty/stout/include/stout/nothing.hpp
// mesos/3rdparty/stout/include/stout/some.hpp

#ifndef BUBBLEFS_UTILS_MESOS_STOUT_SOME_H_
#define BUBBLEFS_UTILS_MESOS_STOUT_SOME_H_

#include <type_traits>
#include <utility>

namespace bubblefs {
namespace mymesos {
  
// A useful type that can be used to represent an Option or Result.
//
// Examples:
//
//   Result<int> result = None();
//   Option<std::string> = None();
//
//   void foo(Option<int> o = None()) {}
//
//   foo(None());

struct None {};

struct Nothing {};

// A useful type that can be used to represent an Option or Result.
//
// Examples:
//
//   Result<int> result = Some(42);
//   Option<std::string> = Some("hello world");
//
//   void foo(Option<std::string> o) {}
//
//   foo(Some("hello world"));

// NOTE: We use an intermediate type, _Some, so that one doesn't need
// to explicitly define the template type when doing 'Some(value)'.
template <typename T>
struct _Some
{
  _Some(T _t) : t(std::move(_t)) {}

  T t;
};


template <typename T>
_Some<typename std::decay<T>::type> Some(T&& t)
{
  return _Some<typename std::decay<T>::type>(std::forward<T>(t));
}

} // namespace mymesos
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MESOS_STOUT_SOME_H_