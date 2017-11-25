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
// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Paddle/paddle/utils/Util.h
// cocos2d-x/cocos/math/CCMathBase.h
// protobuf/src/google/protobuf/stubs/mathutil.h
// tensorflow/tensorflow/stream_executor/lib/mathutil.h

#ifndef BUBBLEFS_UTILS_MATH_UTIL_H_
#define BUBBLEFS_UTILS_MATH_UTIL_H_

#include <assert.h>
#include <float.h>
#include <math.h>
#include <limits>
#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>
#include "platform/platform.h"

namespace bubblefs {
namespace mathutil {
  
/**Util macro for conversion from degrees to radians.*/
#define MATH_DEG_TO_RAD(x)          ((x) * 0.0174532925f)
/**Util macro for conversion from radians to degrees.*/
#define MATH_RAD_TO_DEG(x)          ((x)* 57.29577951f)
/**
@{ Util macro for const float such as epsilon, small float and float precision tolerance.
*/
#define MATH_FLOAT_SMALL            1.0e-37f
#define MATH_TOLERANCE              2e-37f
#define MATH_PIOVER2                1.57079632679489661923f
#define MATH_EPSILON                0.000001f
/**@}*/

// Bit constants for fast bitwise calculation.
static unsigned char kBits[] = {
  1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6, 1 << 7
};

template <typename Float>
inline bool IsFinite(const Float& number) {
#if defined(OS_POSIX)
  return isfinite(number) != 0;
#elif defined(OS_WIN)
  return _finite(number) != 0;
#endif
}

template <typename Float>
inline bool IsNaN(const Float& number) {
#if defined(OS_POSIX)
  return isnan(number) != 0;
#elif defined(OS_WIN)
  return _isnan(number) != 0;
#endif
}  
  
/**
 * calculate the non-negative remainder of a/b
 * @param[in] a
 * @param[in] b, should be positive
 * @return the non-negative remainder of a / b
 */
inline int Mod(int a, int b) {
  int r = a % b;
  return r >= 0 ? r : r + b;
}

/**
 * split array by index.
 * used by sync multi thread task,
 * each thread call CalcSplitArrayInterval with thread id,
 * get a interval as return.
 * input:
 * *totalSize* is array size,
 * *tId* is thread id, *tSize* is total worker thread num
 * output:
 * start and end index as a std::pair
 */
inline std::pair<size_t, size_t> CalcSplitArrayInterval(size_t totalSize,
                                                        size_t tId,
                                                        size_t tSize) {
  size_t start = totalSize * tId / tSize;
  size_t end = totalSize * (tId + 1) / tSize;
  return std::make_pair(start, end);
}

/**
 * same as above, but split at boundary of block.
 */
inline std::pair<size_t, size_t> CalcSplitArrayInterval(size_t totalSize,
                                                        size_t tId,
                                                        size_t tSize,
                                                        size_t blockSize) {
  size_t numBlocks = totalSize / blockSize;
  if (numBlocks * blockSize < totalSize) {
    numBlocks++;
  }

  auto interval = CalcSplitArrayInterval(numBlocks, tId, tSize);
  size_t start = std::min(interval.first * blockSize, totalSize);
  size_t end = std::min(interval.second * blockSize, totalSize);

  return std::make_pair(start, end);
}

template<typename T>
bool AlmostEquals(T a, T b) {
  return a == b;
}
template<>
inline bool AlmostEquals(float a, float b) {
  return fabs(a - b) < 32 * FLT_EPSILON;
}

template<>
inline bool AlmostEquals(double a, double b) {
  return fabs(a - b) < 32 * DBL_EPSILON;
}

class MathUtil {
 public:
  template <typename IntegralType>
  static IntegralType CeilOfRatio(IntegralType numerator,
                                  IntegralType denominator) {
    return CeilOrFloorOfRatio<IntegralType, true>(numerator, denominator);
  }
  template <typename IntegralType>
  static IntegralType FloorOfRatio(IntegralType numerator,
                                   IntegralType denominator) {
    return CeilOrFloorOfRatio<IntegralType, false>(numerator, denominator);
  }
  template <typename IntegralType, bool ceil>
  static IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                         IntegralType denominator);
};

// ---- CeilOrFloorOfRatio ----
// This is a branching-free, cast-to-double-free implementation.
//
// Casting to double is in general incorrect because of loss of precision
// when casting an int64 into a double.
//
// There's a bunch of 'recipes' to compute a integer ceil (or floor) on the web,
// and most of them are incorrect.
template<typename IntegralType, bool ceil>
IntegralType MathUtil::CeilOrFloorOfRatio(IntegralType numerator,
                                          IntegralType denominator) {
  static_assert(std::is_integral<IntegralType>::value,
                 "CeilOfRatio_is_only_defined_for_integral_types");
  assert(denominator != 0);
  // Dividing the smallest signed integer by -1 is not supported: it would
  // SIGFPE
  assert(!std::is_signed<IntegralType>::value ||
         numerator != std::numeric_limits<IntegralType>::min() ||
         denominator != -1);

  const IntegralType rounded_toward_zero = numerator / denominator;
  const IntegralType intermediate_product = rounded_toward_zero * denominator;

  if (ceil) {  // Compile-time condition: not an actual branching
    // When rounded_toward_zero is negative, then an adjustment is never needed:
    // the real ratio is negative, and so rounded toward zero is the ceil.
    // When rounded_toward_zero is non-negative, an adjustment is needed if the
    // sign of the difference numerator - intermediate_product is the same as
    // the sign of the denominator.
    //
    // Using a bool and then a static_cast to IntegralType is not strictly
    // necessary, but it makes the code clear, and anyway the compiler should
    // get rid of it.
    const bool needs_adjustment = (rounded_toward_zero >= 0) &&
        ((denominator > 0 && numerator > intermediate_product) ||
            (denominator < 0 && numerator < intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType ceil_of_ratio = rounded_toward_zero + adjustment;
    return ceil_of_ratio;
  } else {
    // Floor case: symmetrical to the previous one
    const bool needs_adjustment = (rounded_toward_zero <= 0) &&
        ((denominator > 0 && numerator < intermediate_product) ||
         (denominator < 0 && numerator > intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType floor_of_ratio = rounded_toward_zero - adjustment;
    return floor_of_ratio;
  }
}

inline bool FloatEqual(float a, float b) {
  const float EPSILON = 1e-5;
  if (fabs(a - b) < EPSILON) {
    return true;
  }
  return false;
}

inline int ToLog2(int value) {
  return static_cast<int>(floor(log2(value)));
}

// The number of bits necessary to hold the given index.
//
// ------------------------
//   sample input/output
// ------------------------
//   0           -->  0
//   1           -->  1
//   2,3         -->  2
//   4,5,6,7     -->  3
//   128,129,255 -->  8
// ------------------------
inline int ToRadix(int index) {
  assert(index >= 0);
  return index == 0 ? 0 : 1 + ToLog2(index);
}


//===--------------------------------------------------------------------===//
// Count the number of leading zeroes in a given 64-bit unsigned number
//===--------------------------------------------------------------------===//
inline uint64_t CountLeadingZeroes(uint64_t i) {
#if defined __GNUC__ || defined __clang__
  return __builtin_clzl(i);
#else
#error get a better compiler to CountLeadingZeroes
#endif
}

//===--------------------------------------------------------------------===//
// Find the next power of two higher than the provided value
//===--------------------------------------------------------------------===//
inline uint64_t NextPowerOf2(uint64_t n) {
#if defined __GNUC__ || defined __clang__
  assert(n > 0);
  return 1ul << (64 - CountLeadingZeroes(n - 1));
#else
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return ++n;
#endif
}

inline int NextPOT(int x)
{
    x = x - 1;
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >>16);
    return x + 1;
}

} // namespace mathutil 
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MATH_UTIL_H_