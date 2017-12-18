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

// tensorflow/tensorflow/core/platform/default/integral_types.h
// tensorflow/tensorflow/core/platform/types.h

#ifndef BUBBLEFS_PLATFORM_TYPES_H_
#define BUBBLEFS_PLATFORM_TYPES_H_

#include <limits.h>  // So we can set the bounds of our types.
#include <stddef.h>  // For size_t.
#include <stdint.h>  // For intptr_t.
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include "platform/macros.h"

/*
// complete all missing definitions in case previous <stdint.h> inclusion
// without __STDC_LIMIT_MACROS

#ifndef __WORDSIZE
# ifndef _WIN32
#  ifdef _WIN64
#   define __WORDSIZE 64
#  else
#   define __WORDSIZE 32
#  endif
# endif
#endif

#ifndef INT64_MIN
// Minimum of signed integral types.  
# define INT8_MIN       (-128)
# define INT16_MIN      (-32767-1)
# define INT32_MIN      (-2147483647-1)
# define INT64_MIN      (-__INT64_C(9223372036854775807)-1)
// Maximum of signed integral types.  
# define INT8_MAX       (127)
# define INT16_MAX      (32767)
# define INT32_MAX      (2147483647)
# define INT64_MAX      (__INT64_C(9223372036854775807))

// Maximum of unsigned integral types.  
# define UINT8_MAX      (255)
# define UINT16_MAX     (65535)
# define UINT32_MAX     (4294967295U)
# define UINT64_MAX     (__UINT64_C(18446744073709551615))

// Values to test for integral types holding `void *' pointer.  
# if __WORDSIZE == 64
#  define INTPTR_MIN        (-9223372036854775807L-1)
#  define INTPTR_MAX        (9223372036854775807L)
#  define UINTPTR_MAX       (18446744073709551615UL)
# else
#  define INTPTR_MIN        (-2147483647-1)
#  define INTPTR_MAX        (2147483647)
#  define UINTPTR_MAX       (4294967295U)
# endif

// Minimum for largest signed integral type.  
# define INTMAX_MIN     (-__INT64_C(9223372036854775807)-1)
// Maximum for largest signed integral type.  
# define INTMAX_MAX     (__INT64_C(9223372036854775807))

// Maximum for largest unsigned integral type. 
# define UINTMAX_MAX        (__UINT64_C(18446744073709551615))

// Limits of other integer types.  

// Limits of `ptrdiff_t' type.  
# if __WORDSIZE == 64
#  define PTRDIFF_MIN       (-9223372036854775807L-1)
#  define PTRDIFF_MAX       (9223372036854775807L)
# else
#  define PTRDIFF_MIN       (-2147483647-1)
#  define PTRDIFF_MAX       (2147483647)
# endif

// Limit of `size_t' type.
# ifndef SIZE_MAX
#  if __WORDSIZE == 64
#   define SIZE_MAX      (18446744073709551615UL)
#  else
#   define SIZE_MAX      (4294967295U)
#  endif
# endif

// Limits of `wchar_t'.
# ifndef WCHAR_MIN
// These constants might also be defined in <wchar.h>. 
#  define WCHAR_MIN     __WCHAR_MIN
#  define WCHAR_MAX     __WCHAR_MAX
# endif

// Limits of `wint_t'. 
# define WINT_MIN       (0u)
# define WINT_MAX       (4294967295u)
*/

#ifndef  INT8_MAX
#define  INT8_MAX   0x7f
#endif
#ifndef  INT8_MIN
#define  INT8_MIN   (-INT8_MAX - 1)
#endif
#ifndef  UINT8_MAX
#define  UINT8_MAX   (INT8_MAX * 2 + 1)
#endif
#ifndef  INT16_MAX
#define  INT16_MAX   0x7fff
#endif
#ifndef  INT16_MIN
#define  INT16_MIN   (-INT16_MAX - 1)
#endif
#ifndef  UINT16_MAX
#define  UINT16_MAX  0xffff
#endif
#ifndef  INT32_MAX
#define  INT32_MAX   0x7fffffffL
#endif
#ifndef  INT32_MIN
#define  INT32_MIN   (-INT32_MAX - 1L)
#endif
#ifndef  UINT32_MAX
#define  UINT32_MAX  0xffffffffUL
#endif
#ifndef  INT64_MAX
#define  INT64_MAX   0x7fffffffffffffffLL
#endif
#ifndef  INT64_MIN
#define  INT64_MIN   (-INT64_MAX - 1LL)
#endif
#ifndef  UINT64_MAX
#define  UINT64_MAX  0xffffffffffffffffULL
#endif

// KB, MB, GB to bytes
#define KBYTES (1024L)
#define MBYTES (1024L*1024L)
#define GBYTES (1024L*1024L*1024)

// Include appropriate platform-dependent implementations
#if defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) || \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_WINDOWS) 
namespace bubblefs {
//------------------------------------------------------------------------------
// Basis POD types.
//------------------------------------------------------------------------------  
/*
typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

#ifdef _MSC_VER
typedef __int8  int8;
typedef __int16 int16;
typedef __int32 int32;
typedef __int64 int64;

typedef unsigned __int8  uint8;
typedef unsigned __int16 uint16;
typedef unsigned __int32 uint32;
typedef unsigned __int64 uint64;
#endif // _MSC_VER
*/

typedef unsigned int uint;

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef float float32;
typedef double float64;

// in linux/limits.h PATH_MAX    
constexpr int PATH_MAX_LEN = 4096;

}  // namespace bubblefs
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

namespace bubblefs {

// Using statements for common classes that we refer to in caffe2 very often.
// Note that we only place it inside caffe2 so the global namespace is not
// polluted.
/* using override */
using ::std::dynamic_pointer_cast;
using ::std::make_pair;
using ::std::pair;
using ::std::shared_ptr;
using ::std::string;
using ::std::weak_ptr;
using ::std::wstring;
using ::std::unique_ptr;

static const uint8 kuint8max = ((uint8)0xFF);
static const uint16 kuint16max = ((uint16)0xFFFF);
static const uint32 kuint32max = ((uint32)0xFFFFFFFF);
static const uint64 kuint64max = ((uint64)0xFFFFFFFFFFFFFFFFull);
static const int8 kint8min = ((int8)~0x7F);
static const int8 kint8max = ((int8)0x7F);
static const int16 kint16min = ((int16)~0x7FFF);
static const int16 kint16max = ((int16)0x7FFF);
static const int32 kint32min = ((int32)~0x7FFFFFFF);
static const int32 kint32max = ((int32)0x7FFFFFFF);
static const int64 kint64min = ((int64)~0x7FFFFFFFFFFFFFFFll);
static const int64 kint64max = ((int64)0x7FFFFFFFFFFFFFFFll);

static const int32 kInt32Max = 0x7FFFFFFF;
static const int32 kInt32Min = -kInt32Max - 1;
static const int64 kInt64Max = 0x7FFFFFFFFFFFFFFFll;
static const int64 kInt64Min = -kInt64Max - 1;
static const uint32 kUInt32Max = 0xFFFFFFFFu;
static const uint64 kUInt64Max = 0xFFFFFFFFFFFFFFFFull;

static const float kFloatMax = std::numeric_limits<float>::max();
static const float kFloatMin = std::numeric_limits<float>::min();

/* To avoid dividing by zero */
static const float kVerySmallNumber = 1e-15;
static const double kVerySmallNumberDouble = 1e-15;

// A typedef for a uint64 used as a short fingerprint.
typedef uint64 Fprint;

}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TYPES_H_