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
// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This file contains macros and macro-like constructs (e.g., templates) that
// are commonly used throughout Chromium source. (It may also contain things
// that are closely related to things that are commonly used that belong in this
// file.)

// tensorflow/tensorflow/core/platform/macros.h
// tensorflow/tensorflow/core/platform/default/dynamic_annotations.h
// brpc/src/butil/macros.h

#ifndef BUBBLEFS_PLATFORM_MACROS_H_
#define BUBBLEFS_PLATFORM_MACROS_H_

#include <assert.h>
#include <inttypes.h> // PRId64
#include <stddef.h>  // For size_t.
#include <string.h>  // For memcpy.
#include <unistd.h>
#include "platform/base_export.h"
#include "platform/compiler_specific.h"
#include "platform/dynamic_annotations.h"
#include "platform/platform.h"

// There must be many copy-paste versions of these macros which are same
// things, undefine them to avoid conflict.
#undef DISALLOW_COPY
#undef DISALLOW_ASSIGN
#undef DISALLOW_COPY_AND_ASSIGN
#undef DISALLOW_EVIL_CONSTRUCTORS
#undef DISALLOW_IMPLICIT_CONSTRUCTORS

#if !defined(BASE_CXX11_ENABLED)
#define BASE_DELETE_FUNCTION(decl) decl
#else
#define BASE_DELETE_FUNCTION(decl) decl = delete
#endif

// Put this in the private: declarations for a class to be uncopyable.
#define DISALLOW_COPY(TypeName)                         \
    BASE_DELETE_FUNCTION(TypeName(const TypeName&))

// Put this in the private: declarations for a class to be unassignable.
#define DISALLOW_ASSIGN(TypeName)                               \
    BASE_DELETE_FUNCTION(void operator=(const TypeName&))

// A macro to disallow the copy constructor and operator= functions
// This should be used in the private: declarations for a class
#define DISALLOW_COPY_AND_ASSIGN(TypeName)                      \
    BASE_DELETE_FUNCTION(TypeName(const TypeName&));            \
    BASE_DELETE_FUNCTION(void operator=(const TypeName&))

// An older, deprecated, politically incorrect name for the above.
// NOTE: The usage of this macro was banned from our code base, but some
// third_party libraries are yet using it.
// TODO(tfarina): Figure out how to fix the usage of this macro in the
// third_party libraries and get rid of it.
#define DISALLOW_EVIL_CONSTRUCTORS(TypeName) DISALLOW_COPY_AND_ASSIGN(TypeName)

// A macro to disallow all the implicit constructors, namely the
// default constructor, copy constructor and operator= functions.
//
// This should be used in the private: declarations for a class
// that wants to prevent anyone from instantiating it. This is
// especially useful for classes containing only static methods.
#define DISALLOW_IMPLICIT_CONSTRUCTORS(TypeName) \
    BASE_DELETE_FUNCTION(TypeName());            \
    DISALLOW_COPY_AND_ASSIGN(TypeName)
   
#undef arraysize
// The arraysize(arr) macro returns the # of elements in an array arr.
// The expression is a compile-time constant, and therefore can be
// used in defining new arrays, for example.  If you use arraysize on
// a pointer by mistake, you will get a compile-time error.
//
// One caveat is that arraysize() doesn't accept any array of an
// anonymous type or a type defined inside a function.  In these rare
// cases, you have to use the unsafe ARRAYSIZE_UNSAFE() macro below.  This is
// due to a limitation in C++'s template system.  The limitation might
// eventually be removed, but it hasn't happened yet.

// This template function declaration is used in defining arraysize.
// Note that the function doesn't need an implementation, as we only
// use its type.
namespace bubblefs {
namespace base {
template <typename T, size_t N>
char (&ArraySizeHelper(T (&array)[N]))[N];
} // namespace base
} // namespace bubblefs

// That gcc wants both of these prototypes seems mysterious. VC, for
// its part, can't decide which to use (another mystery). Matching of
// template overloads: the final frontier.
#ifndef _MSC_VER
namespace bubblefs {
namespace base {
template <typename T, size_t N>
char (&ArraySizeHelper(const T (&array)[N]))[N];
} // namespace base
} // namespace bubblefs
#endif

#define arraysize(array) (sizeof(::bubblefs::base::ArraySizeHelper(array)))

// gejun: Following macro was used in other modules.
#undef ARRAY_SIZE
#define ARRAY_SIZE(array) arraysize(array)

// ARRAYSIZE_UNSAFE performs essentially the same calculation as arraysize,
// but can be used on anonymous types or types defined inside
// functions.  It's less safe than arraysize as it accepts some
// (although not all) pointers.  Therefore, you should use arraysize
// whenever possible.
//
// The expression ARRAYSIZE_UNSAFE(a) is a compile-time constant of type
// size_t.
//
// ARRAYSIZE_UNSAFE catches a few type errors.  If you see a compiler error
//
//   "warning: division by zero in ..."
//
// when using ARRAYSIZE_UNSAFE, you are (wrongfully) giving it a pointer.
// You should only use ARRAYSIZE_UNSAFE on statically allocated arrays.
//
// The following comments are on the implementation details, and can
// be ignored by the users.
//
// ARRAYSIZE_UNSAFE(arr) works by inspecting sizeof(arr) (the # of bytes in
// the array) and sizeof(*(arr)) (the # of bytes in one array
// element).  If the former is divisible by the latter, perhaps arr is
// indeed an array, in which case the division result is the # of
// elements in the array.  Otherwise, arr cannot possibly be an array,
// and we generate a compiler error to prevent the code from
// compiling.
//
// Since the size of bool is implementation-defined, we need to cast
// !(sizeof(a) & sizeof(*(a))) to size_t in order to ensure the final
// result has type size_t.
//
// This macro is not perfect as it wrongfully accepts certain
// pointers, namely where the pointer size is divisible by the pointee
// size.  Since all our code has to go through a 32-bit compiler,
// where a pointer is 4 bytes, this means all pointers to a type whose
// size is 3 or greater than 4 will be (righteously) rejected.
#undef ARRAYSIZE_UNSAFE
#define ARRAYSIZE_UNSAFE(a) \
    ((sizeof(a) / sizeof(*(a))) / \
     static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))
     
// Concatenate numbers in c/c++ macros.
#ifndef BASE_CONCAT
# define BASE_CONCAT(a, b) BASE_CONCAT_HELPER(a, b)
# define BASE_CONCAT_HELPER(a, b) a##b
#endif
  
#if defined(BASE_CXX11_ENABLED)
// C++11 supports compile-time assertion directly
#define BASE_CASSERT(expr, msg) static_assert(expr, #msg)
#else
// Assert constant boolean expressions at compile-time
// Params:
//   expr     the constant expression to be checked
//   msg      an error infomation conforming name conventions of C/C++
//            variables(alphabets/numbers/underscores, no blanks). For
//            example "cannot_accept_a_number_bigger_than_128" is valid
//            while "this number is out-of-range" is illegal.
//
// when an asssertion like "BASE_CASSERT(false, you_should_not_be_here)"
// breaks, a compilation error is printed:
//   
//   foo.cpp:401: error: enumerator value for `you_should_not_be_here___19' not
//   integer constant
//
// You can call BASE_CASSERT at global scope, inside a class or a function
// 
//   BASE_CASSERT(false, you_should_not_be_here);
//   int main () { ... }
//
//   struct Foo {
//       BASE_CASSERT(1 == 0, Never_equals);
//   };
//
//   int bar(...)
//   {
//       BASE_CASSERT (value < 10, invalid_value);
//   }
//
namespace bubblefs {
namespace base {
template <bool> struct CAssert { static const int x = 1; };
template <> struct CAssert<false> { static const char * x; };
} // namespace base
} // namespace bubblefs

#define BASE_CASSERT(expr, msg)                                \
    enum { BASE_CONCAT(BASE_CONCAT(LINE_, __LINE__), __##msg) \
           = ::bubblefs::base::CAssert<!!(expr)>::x };

#endif  // BASE_CXX11_ENABLED

// The impl. of chrome does not work for offsetof(Object, private_filed)
#undef COMPILE_ASSERT
#define COMPILE_ASSERT(expr, msg)  BASE_CASSERT(expr, msg)
           
#ifndef ASSERT
#define ASSERT(x) assert(x)
#endif

// Used to explicitly mark the return value of a function as unused. If you are
// really sure you don't want to do anything with the return value of a function
// that has been marked WARN_UNUSED_RESULT, wrap it with this. Example:
//
//   scoped_ptr<MyType> my_var = ...;
//   if (TakeOwnership(my_var.get()) == SUCCESS)
//     ignore_result(my_var.release());
//
namespace bubblefs {
namespace base {
template<typename T>
inline void ignore_result(const T&) {
}
} // namespace bubblefs
} // namespace base

// The following enum should be used only as a constructor argument to indicate
// that the variable has static storage class, and that the constructor should
// do nothing to its state.  It indicates to the reader that it is legal to
// declare a static instance of the class, provided the constructor is given
// the base::LINKER_INITIALIZED argument.  Normally, it is unsafe to declare a
// static variable that has a constructor or a destructor because invocation
// order is undefined.  However, IF the type can be initialized by filling with
// zeroes (which the loader does for static variables), AND the destructor also
// does nothing to the storage, AND there are no virtual methods, then a
// constructor declared as
//       explicit MyClass(base::LinkerInitialized x) {}
// and invoked as
//       static MyClass my_variable_name(base::LINKER_INITIALIZED);
namespace bubblefs {
namespace base {
enum LinkerInitialized { LINKER_INITIALIZED };

// Use these to declare and define a static local variable (static T;) so that
// it is leaked so that its destructors are not called at exit. If you need
// thread-safe initialization, use base/lazy_instance.h instead.
#undef CR_DEFINE_STATIC_LOCAL
#define CR_DEFINE_STATIC_LOCAL(type, name, arguments) \
  static type& name = *new type arguments

}  // namespace base
}  // namespace bubblefs

#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L
// Define this to 1 if the code is compiled in C++11 mode; leave it
// undefined otherwise.  Do NOT define it to 0 -- that causes
// '#ifdef LANG_CXX11' to behave differently from '#if LANG_CXX11'.
#define LANG_CXX11 1
#endif

#if defined(__clang__) && defined(LANG_CXX11) && defined(__has_warning)
#if __has_feature(cxx_attributes) && __has_warning("-Wimplicit-fallthrough")
#define FALLTHROUGH_INTENDED [[clang::fallthrough]]  // NOLINT
#endif
#endif

#ifndef FALLTHROUGH_INTENDED
#define FALLTHROUGH_INTENDED \
  do {                          \
  } while (0)
#endif

// tensorflow/tensorflow/core/platform/default/dynamic_annotations.h, Do nothing for this platform.

#define ANNOTATE_MEMORY_IS_INITIALIZED(ptr, bytes) \
  do {                                                \
  } while (0)

#ifdef _MSC_VER
#define LONGLONG(x) x##I64
#define ULONGLONG(x) x##UI64
#define LL_FORMAT "I64"  // As in printf("%I64d", ...)
#else
// By long long, we actually mean int64.
#define LONGLONG(x) x##LL
#define ULONGLONG(x) x##ULL
// Used to format real long long integers.
#define LL_FORMAT "ll"  // As in "%lld". Note that "q" is poor form also.
#endif

#define PRId64_FORMAT "%" PRId64
#define PRIu64_FORMAT "%" PRIu64

// This is not very useful as it does not expand defined symbols if
// called directly. Use its counterpart without the _NO_EXPANSION
// suffix, below.
#define STRINGIZE_NO_EXPANSION(x) #x

// Use this to quote the provided parameter, first expanding it if it
// is a preprocessor symbol.
//
// For example, if:
//   #define A FOO
//   #define B(x) myobj->FunctionCall(x)
//
// Then:
//   STRINGIZE(A) produces "FOO"
//   STRINGIZE(B(y)) produces "myobj->FunctionCall(y)"
#define STRINGIZE(x) STRINGIZE_NO_EXPANSION(x)

// Convert symbol to string
#ifndef BASE_SYMBOLSTR
# define BASE_SYMBOLSTR(a) BASE_SYMBOLSTR_HELPER(a)
# define BASE_SYMBOLSTR_HELPER(a) #a
#endif

#ifndef BASE_TYPEOF
# if defined(BASE_CXX11_ENABLED)
#  define BASE_TYPEOF decltype
# else
#  define BASE_TYPEOF typeof
# endif // BASE_CXX11_ENABLED
#endif  // BASE_TYPEOF

// ptr:     the pointer to the member.
// type:    the type of the container struct this is embedded in.
// member:  the name of the member within the struct.
#ifndef container_of
# define container_of(ptr, type, member) ({                             \
            const BASE_TYPEOF( ((type *)0)->member ) *__mptr = (ptr);  \
            (type *)( (char *)__mptr - offsetof(type,member) );})
#endif

// common macros and data structures
#ifndef FIELD_OFFSET
#define FIELD_OFFSET(s, field) (((size_t) & ((s *)(10))->field) - 10)
#endif

#ifndef CONTAINING_RECORD
#define CONTAINING_RECORD(address, type, field)                                                    \
    ((type *)((char *)(address)-FIELD_OFFSET(type, field)))
#endif

# define VOID_TEMP_FAILURE_RETRY(expression) \
    static_cast<void>(TEMP_FAILURE_RETRY(expression))
    
// remove 'unused parameter' warning    
#define EXPR_UNUSED(expr) do { (void)(expr); } while (0)

/*
   can be used like #if (QT_VERSION >= QT_VERSION_CHECK(4, 4, 0))
*/
#define BASE_VERSION_CHECK(major, minor, patch) ((major<<16)|(minor<<8)|(patch))

#endif // BUBBLEFS_PLATFORM_MACROS_H_