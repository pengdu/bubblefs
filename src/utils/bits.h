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
// tensorflow/tensorflow/core/lib/core/bits.h
// mesos/3rdparty/stout/include/stout/bits.hpp

#ifndef BUBBLEFS_UTILS_BITS_H_
#define BUBBLEFS_UTILS_BITS_H_

#include <type_traits>
#include "platform/logging.h"
#include "platform/types.h"

namespace bubblefs {
  
#if defined(__GNUC__)
/**
 * return the 1-based index of the highest bit set
 *
 * for x > 0:
 * \f[
 *    FindLastSet(x) = 1 + \floor*{\log_{2}x}
 * \f]
 */
inline constexpr size_t FindLastSet(size_t x) {
  return std::is_same<size_t, unsigned int>::value
             ? (x ? 8 * sizeof(x) - __builtin_clz(x) : 0)
             : (std::is_same<size_t, unsigned long>::value  // NOLINT
                    ? (x ? 8 * sizeof(x) - __builtin_clzl(x) : 0)
                    : (x ? 8 * sizeof(x) - __builtin_clzll(x) : 0));
}
#endif

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
int Log2Floor(uint32 n);
int Log2Floor64(uint64 n);

// Return ceiling(log2(n)) for positive integer n.  Returns -1 iff n == 0.
int Log2Ceiling(uint32 n);
int Log2Ceiling64(uint64 n);

// ------------------------------------------------------------------------
// Implementation details follow
// ------------------------------------------------------------------------

#if defined(__GNUC__)

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor(uint32 n) { return n == 0 ? -1 : 31 ^ __builtin_clz(n); }

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor64(uint64 n) {
  return n == 0 ? -1 : 63 ^ __builtin_clzll(n);
}

#else

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor(uint32 n) {
  if (n == 0) return -1;
  int log = 0;
  uint32 value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32 x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  assert(value == 1);
  return log;
}

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
// Log2Floor64() is defined in terms of Log2Floor32()
inline int Log2Floor64(uint64 n) {
  const uint32 topbits = static_cast<uint32>(n >> 32);
  if (topbits == 0) {
    // Top bits are zero, so scan in bottom bits
    return Log2Floor(static_cast<uint32>(n));
  } else {
    return 32 + Log2Floor(topbits);
  }
}

#endif

inline int Log2Ceiling(uint32 n) {
  int floor = Log2Floor(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}

inline int Log2Ceiling64(uint64 n) {
  int floor = Log2Floor64(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}

inline uint32 NextPowerOfTwo(uint32 value) {
  int exponent = Log2Ceiling(value);
  DCHECK_LT(exponent, std::numeric_limits<uint32>::digits);
  return 1 << exponent;
}

inline uint64 NextPowerOfTwo64(uint64 value) {
  int exponent = Log2Ceiling(value);
  DCHECK_LT(exponent, std::numeric_limits<uint64>::digits);
  return 1LL << exponent;
}

inline unsigned GetBitsOf(int v) {
  unsigned n = 0;
  while (v) {
    ++n;
    v >>= 1;
  }
  return n;
}

// Provides efficient bit operations.
// More details can be found at:
// http://graphics.stanford.edu/~seander/bithacks.html
// Counts set bits from a 32 bit unsigned integer using Hamming weight.
inline int countSetBits(uint32_t value)
{
  int count = 0;
  value = value - ((value >> 1) & 0x55555555);
  value = (value & 0x33333333) + ((value >> 2) & 0x33333333);
  count = (((value + (value >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;

  return count;
}

}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_BITS_H_