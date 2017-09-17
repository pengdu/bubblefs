// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// chromium/base/sys_byteorder.h

#ifndef BUBBLEFS_PLATFORM_BYTEORDER_H_
#define BUBBLEFS_PLATFORM_BYTEORDER_H_

#include <stdint.h>
#include <stdlib.h>
#include "platform/cpu_info.h"
#include "platform/macros.h"

#if defined(OS_WIN)
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

// Note: use gcc

// This header defines cross-platform ByteSwap() implementations for 16, 32 and
// 64-bit values, and NetToHostXX() / HostToNextXX() functions equivalent to
// the traditional ntohX() and htonX() functions.
// Use the functions defined here rather than using the platform-specific
// functions directly.

namespace bubblefs {
namespace base {
  
// Returns a value with all bytes in |x| swapped, i.e. reverses the endianness.
/*
inline uint16 ByteSwap(uint16 x) {
  return ((x & 0x00ff) << 8) | ((x & 0xff00) >> 8);
}

inline uint32 ByteSwap(uint32 x) {
  return ((x & 0x000000fful) << 24) | ((x & 0x0000ff00ul) << 8) |
      ((x & 0x00ff0000ul) >> 8) | ((x & 0xff000000ul) >> 24);
}

inline uint64 ByteSwap(uint64 x) {
  return ((x & 0x00000000000000ffull) << 56) |
      ((x & 0x000000000000ff00ull) << 40) |
      ((x & 0x0000000000ff0000ull) << 24) |
      ((x & 0x00000000ff000000ull) << 8) |
      ((x & 0x000000ff00000000ull) >> 8) |
      ((x & 0x0000ff0000000000ull) >> 24) |
      ((x & 0x00ff000000000000ull) >> 40) |
      ((x & 0xff00000000000000ull) >> 56);
}
)
*/
inline uint16_t ByteSwap(uint16_t x) {
#if defined(COMPILER_MSVC)
  return _byteswap_ushort(x);
#else
  return __builtin_bswap16(x);
#endif
}

inline uint32_t ByteSwap(uint32_t x) {
#if defined(COMPILER_MSVC)
  return _byteswap_ulong(x);
#else
  return __builtin_bswap32(x);
#endif
}

inline uint64_t ByteSwap(uint64_t x) {
#if defined(COMPILER_MSVC)
  return _byteswap_uint64(x);
#else
  return __builtin_bswap64(x);
#endif
}

// Converts the bytes in |x| from host order (endianness) to little endian, and
// returns the result.
inline uint16_t ByteSwapToLE16(uint16_t x) {
#if defined(ARCH_CPU_LITTLE_ENDIAN)
  return x;
#else
  return ByteSwap(x);
#endif
}
inline uint32_t ByteSwapToLE32(uint32_t x) {
#if defined(ARCH_CPU_LITTLE_ENDIAN)
  return x;
#else
  return ByteSwap(x);
#endif
}
inline uint64_t ByteSwapToLE64(uint64_t x) {
#if defined(ARCH_CPU_LITTLE_ENDIAN)
  return x;
#else
  return ByteSwap(x);
#endif
}

// Converts the bytes in |x| from network to host order (endianness), and
// returns the result.
inline uint16_t NetToHost16(uint16_t x) {
#if defined(ARCH_CPU_LITTLE_ENDIAN)
  return ByteSwap(x);
#else
  return x;
#endif
}
inline uint32_t NetToHost32(uint32_t x) {
#if defined(ARCH_CPU_LITTLE_ENDIAN)
  return ByteSwap(x);
#else
  return x;
#endif
}
inline uint64_t NetToHost64(uint64_t x) {
#if defined(ARCH_CPU_LITTLE_ENDIAN)
  return ByteSwap(x);
#else
  return x;
#endif
}

// Converts the bytes in |x| from host to network order (endianness), and
// returns the result.
inline uint16_t HostToNet16(uint16_t x) {
#if defined(ARCH_CPU_LITTLE_ENDIAN)
  return ByteSwap(x);
#else
  return x;
#endif
}
inline uint32_t HostToNet32(uint32_t x) {
#if defined(ARCH_CPU_LITTLE_ENDIAN)
  return ByteSwap(x);
#else
  return x;
#endif
}
inline uint64_t HostToNet64(uint64_t x) {
#if defined(ARCH_CPU_LITTLE_ENDIAN)
  return ByteSwap(x);
#else
  return x;
#endif
}  
  
} // namespace base  
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_BYTEORDER_H_