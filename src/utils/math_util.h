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

// Paddle/paddle/utils/Util.h

#ifndef BUBBLEFS_UTILS_MATH_UTIL_H_
#define BUBBLEFS_UTILS_MATH_UTIL_H_

#include <math.h>
#include <algorithm>
#include <utility>

namespace bubblefs {
namespace math {

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

} // namespace math 
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MATH_UTIL_H_