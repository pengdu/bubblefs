// Modifications copyright (C) 2017, Baidu.com, Inc.
// Copyright 2017 The Apache Software Foundation

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
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

// tensorflow/tensorflow/core/platform/macros.h
// tensorflow/tensorflow/core/platform/default/dynamic_annotations.h

#ifndef BUBBLEFS_PLATFORM_MACROS_H_
#define BUBBLEFS_PLATFORM_MACROS_H_

// Compiler attributes
#if (defined(__GNUC__) || defined(__APPLE__)) && !defined(SWIG)
// Compiler supports GCC-style attributes
#define TF_ATTRIBUTE_NORETURN __attribute__((noreturn))
#define TF_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define TF_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define TF_ATTRIBUTE_UNUSED __attribute__((unused))
#define TF_ATTRIBUTE_COLD __attribute__((cold))
#define TF_ATTRIBUTE_WEAK __attribute__((weak))
#define TF_PACKED __attribute__((packed))
#define TF_MUST_USE_RESULT __attribute__((warn_unused_result))
#define TF_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#define TF_SCANF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__scanf__, string_index, first_to_check)))
#elif defined(COMPILER_MSVC)
// Non-GCC equivalents
#define TF_ATTRIBUTE_NORETURN __declspec(noreturn)
#define TF_ATTRIBUTE_ALWAYS_INLINE
#define TF_ATTRIBUTE_NOINLINE
#define TF_ATTRIBUTE_UNUSED
#define TF_ATTRIBUTE_COLD
#define TF_MUST_USE_RESULT
#define TF_PACKED
#define TF_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define TF_SCANF_ATTRIBUTE(string_index, first_to_check)
#else
// Non-GCC equivalents
#define TF_ATTRIBUTE_NORETURN
#define TF_ATTRIBUTE_ALWAYS_INLINE
#define TF_ATTRIBUTE_NOINLINE
#define TF_ATTRIBUTE_UNUSED
#define TF_ATTRIBUTE_COLD
#define TF_ATTRIBUTE_WEAK
#define TF_MUST_USE_RESULT
#define TF_PACKED
#define TF_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define TF_SCANF_ATTRIBUTE(string_index, first_to_check)
#endif

// Control visiblity outside .so
#if defined(COMPILER_MSVC)
#ifdef TF_COMPILE_LIBRARY
#define TF_EXPORT __declspec(dllexport)
#else
#define TF_EXPORT __declspec(dllimport)
#endif  // TF_COMPILE_LIBRARY
#else
#define TF_EXPORT __attribute__((visibility("default")))
#endif  // COMPILER_MSVC

// GCC can be told that a certain branch is not likely to be taken (for
// instance, a CHECK failure), and use that information in static analysis.
// Giving it this information can help it optimize for the common case in
// the absence of better information (ie. -fprofile-arcs).
#if defined(__GNUC__) && __GNUC__ >= 4
#define TF_PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define TF_PREDICT_TRUE(x) (__builtin_expect(!!(x), 1))
#define TF_LIKELY(x)   (__builtin_expect((x), 1))
#define TF_UNLIKELY(x) (__builtin_expect((x), 0))
#define TF_ALIGN_AS(n) alignas(n)
#define TF_PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
#define TF_SYNC_SYNCHRONIZE __sync_synchronize();
#define TF_SYNC_ADD_AND_FETCH(x, y) __sync_add_and_fetch(x, y);
#define TF_SYNC_FETCH_AND_ADD(x, y) __sync_fetch_and_add(x, y);
#define TF_SYNC_BOOL_COMPARE_AND_SWAP(x, y, z) __sync_bool_compare_and_swap(x, y, x);
#define TF_SYNC_VAL_COMPARE_AND_SWAP(x, y, z) __sync_val_compare_and_swap(x, y, z);
#define TF_SYNC_LOCK_TEST_AND_SET(x, y, z) __sync_lock_test_and_set(x, y);
#else
#define TF_PREDICT_FALSE(x) (x)
#define TF_PREDICT_TRUE(x) (x)
#define TF_LIKELY(x)   (x)
#define TF_UNLIKELY(x) (x)
#define TF_ALIGN_AS(n)
#define TF_PREFETCH(addr, rw, locality)
#define TF_SYNC_SYNCHRONIZE
#define TF_SYNC_ADD_AND_FETCH(x, y)
#define TF_SYNC_FETCH_AND_ADD(x, y)
#define TF_SYNC_BOOL_COMPARE_AND_SWAP(x, y, z)
#define TF_SYNC_VAL_COMPARE_AND_SWAP(x, y, z)
#define TF_SYNC_LOCK_TEST_AND_SET(x, y, z)
#endif

// A macro to disallow the copy constructor and operator= functions
// This is usually placed in the private: declarations for a class.
#define TF_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete

// The TF_ARRAYSIZE(arr) macro returns the # of elements in an array arr.
//
// The expression TF_ARRAYSIZE(a) is a compile-time constant of type
// size_t.
#define TF_ARRAYSIZE(a)         \
  ((sizeof(a) / sizeof(*(a))) / \
   static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))

#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L
// Define this to 1 if the code is compiled in C++11 mode; leave it
// undefined otherwise.  Do NOT define it to 0 -- that causes
// '#ifdef LANG_CXX11' to behave differently from '#if LANG_CXX11'.
#define LANG_CXX11 1
#endif

#if defined(__clang__) && defined(LANG_CXX11) && defined(__has_warning)
#if __has_feature(cxx_attributes) && __has_warning("-Wimplicit-fallthrough")
#define TF_FALLTHROUGH_INTENDED [[clang::fallthrough]]  // NOLINT
#endif
#endif

#ifndef TF_FALLTHROUGH_INTENDED
#define TF_FALLTHROUGH_INTENDED \
  do {                          \
  } while (0)
#endif

// dynamic_annotations.h, Do nothing for this platform.

#define TF_ANNOTATE_MEMORY_IS_INITIALIZED(ptr, bytes) \
  do {                                                \
  } while (0)

#define TF_ANNOTATE_BENIGN_RACE(ptr, description) \
  do {                                            \
  } while (0)

#define TF_ATTRIBUTE_NO_SANITIZE_MEMORY

#define TF_NOEXCEPT noexcept

#endif // BUBBLEFS_PLATFORM_MACROS_H_