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
// protobuf/src/google/protobuf/stubs/mathutil.h

#ifndef BUBBLEFS_UTILS_MATH_UTIL_H_
#define BUBBLEFS_UTILS_MATH_UTIL_H_

#include <float.h>
#include <math.h>
#include <algorithm>
#include <utility>

namespace bubblefs {
namespace mathutil {

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

} // namespace mathutil 
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MATH_UTIL_H_