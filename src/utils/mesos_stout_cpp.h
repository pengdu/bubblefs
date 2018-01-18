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

// mesos/3rdparty/stout/include/stout/cpp14.hpp
// mesos/3rdparty/stout/include/stout/cpp17.hpp

#ifndef BUBBLEFS_UTILS_MESOS_STOUT_CPP_H_
#define BUBBLEFS_UTILS_MESOS_STOUT_CPP_H_

#include <cstddef>
#include <utility>

namespace bubblefs {
namespace mymesos {
  
// This file contains implementation of C++14 standard library features.
// Once we adopt C++14, this file should be removed and usages of its
// functionality should be replaced with the standard library equivalents
// by replacing `cpp14` with `std` and including the appropriate headers.

// Note that code in this file may not follow stout conventions strictly
// as it uses names as defined in C++ standard.

namespace cpp14 {

// This is a simplified implementation of C++14 `std::index_sequence`
// and `std::make_index_sequence` from <utility>.
template <typename T, T... Is>
struct integer_sequence
{
  static constexpr std::size_t size() noexcept { return sizeof...(Is); }
};


namespace internal {

template <typename T, std::size_t N, std::size_t... Is>
struct IntegerSequenceGen : IntegerSequenceGen<T, N - 1, N - 1, Is...> {};


template <typename T, std::size_t... Is>
struct IntegerSequenceGen<T, 0, Is...>
{
  using type = integer_sequence<T, Is...>;
};

} // namespace internal {


template <typename T, std::size_t N>
using make_integer_sequence = typename internal::IntegerSequenceGen<T, N>::type;


template <std::size_t... Is>
using index_sequence = integer_sequence<std::size_t, Is...>;


template <std::size_t N>
using make_index_sequence = make_integer_sequence<std::size_t, N>;

} // namespace cpp14

// This file contains implementation of C++17 standard library features.
// Once we adopt C++17, this file should be removed and usages of its
// functionality should be replaced with the standard library equivalents
// by replacing `cpp17` with `std` and including the appropriate headers.

// Note that code in this file may not follow stout conventions strictly
// as it uses names as defined in C++ standard.

namespace cpp17 {

// <functional>

// `std::invoke`

#ifdef __WINDOWS__
using std::invoke;
#else
#define RETURN(...) -> decltype(__VA_ARGS__) { return __VA_ARGS__; }

// NOTE: This implementation is not strictly conforming currently
// as attempting to use it with `std::reference_wrapper` will result
// in compilation failure.

template <typename F, typename... As>
auto invoke(F&& f, As&&... as)
  RETURN(std::forward<F>(f)(std::forward<As>(as)...))

template <typename B, typename T, typename D>
auto invoke(T B::*pmv, D&& d)
  RETURN(std::forward<D>(d).*pmv)

template <typename Pmv, typename Ptr>
auto invoke(Pmv pmv, Ptr&& ptr)
  RETURN((*std::forward<Ptr>(ptr)).*pmv)

template <typename B, typename T, typename D, typename... As>
auto invoke(T B::*pmf, D&& d, As&&... as)
  RETURN((std::forward<D>(d).*pmf)(std::forward<As>(as)...))

template <typename Pmf, typename Ptr, typename... As>
auto invoke(Pmf pmf, Ptr&& ptr, As&&... as)
  RETURN(((*std::forward<Ptr>(ptr)).*pmf)(std::forward<As>(as)...))

#undef RETURN
#endif

} // namespace cpp17

} // namespace mymesos
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MESOS_STOUT_CPP_H_