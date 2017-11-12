/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

/* mfbt maths algorithms. */

#ifndef BUBBLEFS_UTILS_MOZILLA_MATH_ALGORITHMS_H_
#define BUBBLEFS_UTILS_MOZILLA_MATH_ALGORITHMS_H_

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include "utils/mozilla_attributes.h"

namespace bubblefs {
namespace mymozilla {

// Greatest Common Divisor
template<typename IntegerType>
MOZ_ALWAYS_INLINE IntegerType
EuclidGCD(IntegerType aA, IntegerType aB)
{
  // Euclid's algorithm; O(N) in the worst case.  (There are better
  // ways, but we don't need them for the current use of this algo.)
  MOZ_ASSERT(aA > IntegerType(0));
  MOZ_ASSERT(aB > IntegerType(0));

  while (aA != aB) {
    if (aA > aB) {
      aA = aA - aB;
    } else {
      aB = aB - aA;
    }
  }

  return aA;
}

// Least Common Multiple
template<typename IntegerType>
MOZ_ALWAYS_INLINE IntegerType
EuclidLCM(IntegerType aA, IntegerType aB)
{
  // Divide first to reduce overflow risk.
  return (aA / EuclidGCD(aA, aB)) * aB;
}

inline float
AbsFloat(const float aFloat)
{
  return fabs(aFloat);
}

inline double
AbsDouble(const double aDouble)
{
  return fabs(aDouble);
}

inline long double
AbsLongDouble(const long double aLongDouble)
{
  return fabs(aLongDouble);
}

#if defined(_MSC_VER) && \
    (defined(_M_IX86) || defined(_M_AMD64) || defined(_M_X64))
#  define MOZ_BITSCAN_WINDOWS

#  include <intrin.h>
#  pragma intrinsic(_BitScanForward, _BitScanReverse)

#  if defined(_M_AMD64) || defined(_M_X64)
#    define MOZ_BITSCAN_WINDOWS64
#   pragma intrinsic(_BitScanForward64, _BitScanReverse64)
#  endif

#endif

namespace detail {

#if defined(MOZ_BITSCAN_WINDOWS)

inline uint_fast8_t
CountLeadingZeroes32(uint32_t aValue)
{
  unsigned long index;
  if (!_BitScanReverse(&index, static_cast<unsigned long>(aValue)))
      return 32;
  return uint_fast8_t(31 - index);
}


inline uint_fast8_t
CountTrailingZeroes32(uint32_t aValue)
{
  unsigned long index;
  if (!_BitScanForward(&index, static_cast<unsigned long>(aValue)))
      return 32;
  return uint_fast8_t(index);
}

inline uint_fast8_t
CountPopulation32(uint32_t aValue)
{
  uint32_t x = aValue - ((aValue >> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  return (((x + (x >> 4)) & 0xf0f0f0f) * 0x1010101) >> 24;
}
inline uint_fast8_t
CountPopulation64(uint64_t aValue)
{
  return uint_fast8_t(CountPopulation32(aValue & 0xffffffff) +
                      CountPopulation32(aValue >> 32));
}

inline uint_fast8_t
CountLeadingZeroes64(uint64_t aValue)
{
#if defined(MOZ_BITSCAN_WINDOWS64)
  unsigned long index;
  if (!_BitScanReverse64(&index, static_cast<unsigned __int64>(aValue)))
      return 64;
  return uint_fast8_t(63 - index);
#else
  uint32_t hi = uint32_t(aValue >> 32);
  if (hi != 0) {
    return CountLeadingZeroes32(hi);
  }
  return 32u + CountLeadingZeroes32(uint32_t(aValue));
#endif
}

inline uint_fast8_t
CountTrailingZeroes64(uint64_t aValue)
{
#if defined(MOZ_BITSCAN_WINDOWS64)
  unsigned long index;
  if (!_BitScanForward64(&index, static_cast<unsigned __int64>(aValue)))
      return 64;
  return uint_fast8_t(index);
#else
  uint32_t lo = uint32_t(aValue);
  if (lo != 0) {
    return CountTrailingZeroes32(lo);
  }
  return 32u + CountTrailingZeroes32(uint32_t(aValue >> 32));
#endif
}

#  ifdef MOZ_HAVE_BITSCAN64
#    undef MOZ_HAVE_BITSCAN64
#  endif

#elif defined(__clang__) || defined(__GNUC__)

#  if defined(__clang__)
#    if !__has_builtin(__builtin_ctz) || !__has_builtin(__builtin_clz)
#      error "A clang providing __builtin_c[lt]z is required to build"
#    endif
#  else
     // gcc has had __builtin_clz and friends since 3.4: no need to check.
#  endif

inline uint_fast8_t
CountLeadingZeroes32(uint32_t aValue)
{
  return __builtin_clz(aValue);
}

inline uint_fast8_t
CountTrailingZeroes32(uint32_t aValue)
{
  return __builtin_ctz(aValue);
}

inline uint_fast8_t
CountPopulation32(uint32_t aValue)
{
  return __builtin_popcount(aValue);
}

inline uint_fast8_t
CountPopulation64(uint64_t aValue)
{
  return __builtin_popcountll(aValue);
}

inline uint_fast8_t
CountLeadingZeroes64(uint64_t aValue)
{
  return __builtin_clzll(aValue);
}

inline uint_fast8_t
CountTrailingZeroes64(uint64_t aValue)
{
  return __builtin_ctzll(aValue);
}

#else
#  error "Implement these!"
inline uint_fast8_t CountLeadingZeroes32(uint32_t aValue) = delete;
inline uint_fast8_t CountTrailingZeroes32(uint32_t aValue) = delete;
inline uint_fast8_t CountPopulation32(uint32_t aValue) = delete;
inline uint_fast8_t CountPopulation64(uint64_t aValue) = delete;
inline uint_fast8_t CountLeadingZeroes64(uint64_t aValue) = delete;
inline uint_fast8_t CountTrailingZeroes64(uint64_t aValue) = delete;
#endif

} // namespace detail

/**
 * Compute the number of high-order zero bits in the NON-ZERO number |aValue|.
 * That is, looking at the bitwise representation of the number, with the
 * highest- valued bits at the start, return the number of zeroes before the
 * first one is observed.
 *
 * CountLeadingZeroes32(0xF0FF1000) is 0;
 * CountLeadingZeroes32(0x7F8F0001) is 1;
 * CountLeadingZeroes32(0x3FFF0100) is 2;
 * CountLeadingZeroes32(0x1FF50010) is 3; and so on.
 */
inline uint_fast8_t
CountLeadingZeroes32(uint32_t aValue)
{
  MOZ_ASSERT(aValue != 0);
  return detail::CountLeadingZeroes32(aValue);
}

/**
 * Compute the number of low-order zero bits in the NON-ZERO number |aValue|.
 * That is, looking at the bitwise representation of the number, with the
 * lowest- valued bits at the start, return the number of zeroes before the
 * first one is observed.
 *
 * CountTrailingZeroes32(0x0100FFFF) is 0;
 * CountTrailingZeroes32(0x7000FFFE) is 1;
 * CountTrailingZeroes32(0x0080FFFC) is 2;
 * CountTrailingZeroes32(0x0080FFF8) is 3; and so on.
 */
inline uint_fast8_t
CountTrailingZeroes32(uint32_t aValue)
{
  MOZ_ASSERT(aValue != 0);
  return detail::CountTrailingZeroes32(aValue);
}

/**
 * Compute the number of one bits in the number |aValue|,
 */
inline uint_fast8_t
CountPopulation32(uint32_t aValue)
{
  return detail::CountPopulation32(aValue);
}

/** Analogous to CoutPopulation32, but for 64-bit numbers */
inline uint_fast8_t
CountPopulation64(uint64_t aValue)
{
  return detail::CountPopulation64(aValue);
}

/** Analogous to CountLeadingZeroes32, but for 64-bit numbers. */
inline uint_fast8_t
CountLeadingZeroes64(uint64_t aValue)
{
  MOZ_ASSERT(aValue != 0);
  return detail::CountLeadingZeroes64(aValue);
}

/** Analogous to CountTrailingZeroes32, but for 64-bit numbers. */
inline uint_fast8_t
CountTrailingZeroes64(uint64_t aValue)
{
  MOZ_ASSERT(aValue != 0);
  return detail::CountTrailingZeroes64(aValue);
}

namespace detail {

template<typename T, size_t Size = sizeof(T)>
class CeilingLog2;

template<typename T>
class CeilingLog2<T, 4>
{
public:
  static uint_fast8_t compute(const T aValue)
  {
    // Check for <= 1 to avoid the == 0 undefined case.
    return aValue <= 1 ? 0u : 32u - CountLeadingZeroes32(aValue - 1);
  }
};

template<typename T>
class CeilingLog2<T, 8>
{
public:
  static uint_fast8_t compute(const T aValue)
  {
    // Check for <= 1 to avoid the == 0 undefined case.
    return aValue <= 1 ? 0u : 64u - CountLeadingZeroes64(aValue - 1);
  }
};

} // namespace detail

/**
 * Compute the log of the least power of 2 greater than or equal to |aValue|.
 *
 * CeilingLog2(0..1) is 0;
 * CeilingLog2(2) is 1;
 * CeilingLog2(3..4) is 2;
 * CeilingLog2(5..8) is 3;
 * CeilingLog2(9..16) is 4; and so on.
 */
template<typename T>
inline uint_fast8_t
CeilingLog2(const T aValue)
{
  return detail::CeilingLog2<T>::compute(aValue);
}

/** A CeilingLog2 variant that accepts only size_t. */
inline uint_fast8_t
CeilingLog2Size(size_t aValue)
{
  return CeilingLog2(aValue);
}

namespace detail {

template<typename T, size_t Size = sizeof(T)>
class FloorLog2;

template<typename T>
class FloorLog2<T, 4>
{
public:
  static uint_fast8_t compute(const T aValue)
  {
    return 31u - CountLeadingZeroes32(aValue | 1);
  }
};

template<typename T>
class FloorLog2<T, 8>
{
public:
  static uint_fast8_t compute(const T aValue)
  {
    return 63u - CountLeadingZeroes64(aValue | 1);
  }
};

} // namespace detail

/**
 * Compute the log of the greatest power of 2 less than or equal to |aValue|.
 *
 * FloorLog2(0..1) is 0;
 * FloorLog2(2..3) is 1;
 * FloorLog2(4..7) is 2;
 * FloorLog2(8..15) is 3; and so on.
 */
template<typename T>
inline uint_fast8_t
FloorLog2(const T aValue)
{
  return detail::FloorLog2<T>::compute(aValue);
}

/** A FloorLog2 variant that accepts only size_t. */
inline uint_fast8_t
FloorLog2Size(size_t aValue)
{
  return FloorLog2(aValue);
}

/*
 * Compute the smallest power of 2 greater than or equal to |x|.  |x| must not
 * be so great that the computed value would overflow |size_t|.
 */
inline size_t
RoundUpPow2(size_t aValue)
{
  static_assert(aValue <= (size_t(1) << (sizeof(size_t) * CHAR_BIT - 1)),
               "can't round up -- will overflow!");
  return size_t(1) << CeilingLog2(aValue);
}

/**
 * Rotates the bits of the given value left by the amount of the shift width.
 */
template<typename T>
inline T
RotateLeft(const T aValue, uint_fast8_t aShift)
{
  static_assert(aShift < sizeof(T) * CHAR_BIT, "Shift value is too large!");
  static_assert(aShift > 0,
               "Rotation by value length is undefined behavior, but compilers "
               "do not currently fold a test into the rotate instruction. "
               "Please remove this restriction when compilers optimize the "
               "zero case (http://blog.regehr.org/archives/1063).");
  return (aValue << aShift) | (aValue >> (sizeof(T) * CHAR_BIT - aShift));
}

/**
 * Rotates the bits of the given value right by the amount of the shift width.
 */
template<typename T>
inline T
RotateRight(const T aValue, uint_fast8_t aShift)
{
  MOZ_ASSERT(aShift < sizeof(T) * CHAR_BIT, "Shift value is too large!");
  MOZ_ASSERT(aShift > 0,
             "Rotation by value length is undefined behavior, but compilers "
             "do not currently fold a test into the rotate instruction. "
             "Please remove this restriction when compilers optimize the "
             "zero case (http://blog.regehr.org/archives/1063).");
  return (aValue >> aShift) | (aValue << (sizeof(T) * CHAR_BIT - aShift));
}

/**
 * Returns true if |x| is a power of two.
 * Zero is not an integer power of two. (-Inf is not an integer)
 */
template<typename T>
inline bool
IsPowerOfTwo(T x)
{
    return x && (x & (x - 1)) == 0;
}

template<typename T>
inline T
Clamp(const T aValue, const T aMin, const T aMax)
{
    MOZ_ASSERT(aMin <= aMax);

    if (aValue <= aMin)
        return aMin;
    if (aValue >= aMax)
        return aMax;
    return aValue;
}

} // namespace mymozilla
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_MOZILLA_MATH_ALGORITHMS_H_