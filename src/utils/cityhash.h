// Copyright (c) 2011 Google, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// CityHash, by Geoff Pike and Jyrki Alakuijala
//
// http://code.google.com/p/cityhash/
//
// This file provides a few functions for hashing strings.  All of them are
// high-quality functions in the sense that they pass standard tests such
// as Austin Appleby's SMHasher.  They are also fast.
//
// For 64-bit x86 code, on short strings, we don't know of anything faster than
// CityHash64 that is of comparable quality.  We believe our nearest competitor
// is Murmur3.  For 64-bit x86 code, CityHash64 is an excellent choice for hash
// tables and most other hashing (excluding cryptography).
//
// For 64-bit x86 code, on long strings, the picture is more complicated.
// On many recent Intel CPUs, such as Nehalem, Westmere, Sandy Bridge, etc.,
// CityHashCrc128 appears to be faster than all competitors of comparable
// quality.  CityHash128 is also good but not quite as fast.  We believe our
// nearest competitor is Bob Jenkins' Spooky.  We don't have great data for
// other 64-bit CPUs, but for long strings we know that Spooky is slightly
// faster than CityHash on some relatively recent AMD x86-64 CPUs, for example.
// Note that CityHashCrc128 is declared in citycrc.h.
//
// For 32-bit x86 code, we don't know of anything faster than CityHash32 that
// is of comparable quality.  We believe our nearest competitor is Murmur3A.
// (On 64-bit CPUs, it is typically faster to use the other CityHash variants.)
//
// Functions in the CityHash family are not suitable for cryptography.
//
// Please see CityHash's README file for more details on our performance
// measurements and so on.
//
// WARNING: This code has been only lightly tested on big-endian platforms!
// It is known to work well on little-endian platforms that have a small penalty
// for unaligned reads, such as current Intel and AMD moderate-to-high-end CPUs.
// It should work on all 32-bit and 64-bit platforms that allow unaligned reads;
// bug reports are welcome.
//
// By the way, for some hash functions, given strings a and b, the hash
// of a+b is easily derived from the hashes of a and b.  This property
// doesn't hold for any hash functions in this file.

/*
Introduction
============

CityHash provides hash functions for strings.  The functions mix the
input bits thoroughly but are not suitable for cryptography.  See
"Hash Quality," below, for details on how CityHash was tested and so on.

We provide reference implementations in C++, with a friendly MIT license.

CityHash32() returns a 32-bit hash.

CityHash64() and similar return a 64-bit hash.

CityHash128() and similar return a 128-bit hash and are tuned for
strings of at least a few hundred bytes.  Depending on your compiler
and hardware, it's likely faster than CityHash64() on sufficiently long
strings.  It's slower than necessary on shorter strings, but we expect
that case to be relatively unimportant.

CityHashCrc128() and similar are variants of CityHash128() that depend
on _mm_crc32_u64(), an intrinsic that compiles to a CRC32 instruction
on some CPUs.  However, none of the functions we provide are CRCs.

CityHashCrc256() is a variant of CityHashCrc128() that also depends
on _mm_crc32_u64().  It returns a 256-bit hash.

All members of the CityHash family were designed with heavy reliance
on previous work by Austin Appleby, Bob Jenkins, and others.
For example, CityHash32 has many similarities with Murmur3a.

Performance on long strings: 64-bit CPUs
========================================
 
We are most excited by the performance of CityHash64() and its variants on
short strings, but long strings are interesting as well.

CityHash is intended to be fast, under the constraint that it hash very
well.  For CPUs with the CRC32 instruction, CRC is speedy, but CRC wasn't
designed as a hash function and shouldn't be used as one.  CityHashCrc128()
is not a CRC, but it uses the CRC32 machinery.

On a single core of a 2.67GHz Intel Xeon X5550, CityHashCrc256 peaks at about
5 to 5.5 bytes/cycle.  The other CityHashCrc functions are wrappers around
CityHashCrc256 and should have similar performance on long strings.
(CityHashCrc256 in v1.0.3 was even faster, but we decided it wasn't as thorough
as it should be.)  CityHash128 peaks at about 4.3 bytes/cycle.  The fastest
Murmur variant on that hardware, Murmur3F, peaks at about 2.4 bytes/cycle.
We expect the peak speed of CityHash128 to dominate CityHash64, which is
aimed more toward short strings or use in hash tables.

For long strings, a new function by Bob Jenkins, SpookyHash, is just
slightly slower than CityHash128 on Intel x86-64 CPUs, but noticeably
faster on AMD x86-64 CPUs.  For hashing long strings on AMD CPUs
and/or CPUs without the CRC instruction, SpookyHash may be just as
good or better than any of the CityHash variants.

Performance on short strings: 64-bit CPUs
=========================================

For short strings, e.g., most hash table keys, CityHash64 is faster than
CityHash128, and probably faster than all the aforementioned functions,
depending on the mix of string lengths.  Here are a few results from that
same hardware, where we (unrealistically) tested a single string length over
and over again:

Hash              Results
------------------------------------------------------------------------------
CityHash64 v1.0.3 7ns for 1 byte, or 6ns for 8 bytes, or 9ns for 64 bytes
Murmur2 (64-bit)  6ns for 1 byte, or 6ns for 8 bytes, or 15ns for 64 bytes
Murmur3F          14ns for 1 byte, or 15ns for 8 bytes, or 23ns for 64 bytes

We don't have CityHash64 benchmarks results for v1.1, but we expect the
numbers to be similar.

Performance: 32-bit CPUs
========================

CityHash32 is the newest variant of CityHash.  It is intended for
32-bit hardware in general but has been mostly tested on x86.  Our benchmarks
suggest that Murmur3 is the nearest competitor to CityHash32 on x86.
We don't know of anything faster that has comparable quality.  The speed rankings
in our testing: CityHash32 > Murmur3f > Murmur3a (for long strings), and
CityHash32 > Murmur3a > Murmur3f (for short strings).
*/

// cityhash/src/city.h

#ifndef BUBBLEFS_UTILS_CITY_HASH_H_
#define BUBBLEFS_UTILS_CITY_HASH_H_

#include <stdlib.h>  // for size_t.
#include <stdint.h>
#include <utility>

typedef uint8_t uint8;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef std::pair<uint64, uint64> uint128;

inline uint64 Uint128Low64(const uint128& x) { return x.first; }
inline uint64 Uint128High64(const uint128& x) { return x.second; }

// Hash function for a byte array.
uint64 CityHash64(const char *buf, size_t len);

// Hash function for a byte array.  For convenience, a 64-bit seed is also
// hashed into the result.
uint64 CityHash64WithSeed(const char *buf, size_t len, uint64 seed);

// Hash function for a byte array.  For convenience, two seeds are also
// hashed into the result.
uint64 CityHash64WithSeeds(const char *buf, size_t len,
                           uint64 seed0, uint64 seed1);

// Hash function for a byte array.
uint128 CityHash128(const char *s, size_t len);

// Hash function for a byte array.  For convenience, a 128-bit seed is also
// hashed into the result.
uint128 CityHash128WithSeed(const char *s, size_t len, uint128 seed);

// Hash function for a byte array.  Most useful in 32-bit binaries.
uint32 CityHash32(const char *buf, size_t len);

// Hash 128 input bits down to 64 bits of output.
// This is intended to be a reasonably good hash function.
inline uint64 Hash128to64(const uint128& x) {
  // Murmur-inspired hashing.
  const uint64 kMul = 0x9ddfea08eb382d69ULL;
  uint64 a = (Uint128Low64(x) ^ Uint128High64(x)) * kMul;
  a ^= (a >> 47);
  uint64 b = (Uint128High64(x) ^ a) * kMul;
  b ^= (b >> 47);
  b *= kMul;
  return b;
}

#endif  // BUBBLEFS_UTILS_CITY_HASH_H_