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
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <type_traits>
#include <vector>
#include "platform/macros.h"

// Include appropriate platform-dependent implementations
#if defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) || \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_WINDOWS)
namespace bubblefs {
/*
typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long long int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;
*/

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

}  // namespace bubblefs
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

namespace bubblefs {

// Using statements for common classes that we refer to in caffe2 very often.
// Note that we only place it inside caffe2 so the global namespace is not
// polluted.
/* using override */
using std::set;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

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

// A typedef for a uint64 used as a short fingerprint.
typedef uint64 Fprint;

// Data type for caffe2 Index/Size. We use size_t to be safe here as well as for
// large matrices that are common in sparse math.
typedef int64_t TIndex;

// Note(Yangqing): NVCC does not play well with unordered_map on some platforms,
// forcing us to use std::map instead of unordered_map. This may affect speed
// in some cases, but in most of the computation code we do not access map very
// often, so it should be fine for us. I am putting a CaffeMap alias so we can
// change it more easily if things work out for unordered_map down the road.
template <typename Key, typename Value>
using CaffeMap = std::map<Key, Value>;
// using CaffeMap = std::unordered_map;

}  // namespace bubblefs

// KB, MB, GB to bytes
#define KBYTES (1024L)
#define MBYTES (1024L*1024L)
#define GBYTES (1024L*1024L*1024)

#endif // BUBBLEFS_PLATFORM_TYPES_H_