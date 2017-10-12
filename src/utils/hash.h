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

// ceph/src/include/hash.h
// tensorflow/tensorflow/core/lib/hash/hash.h

// Simple hash functions used for internal data structures

#ifndef BUBBLEFS_UTILS_HASH_H_
#define BUBBLEFS_UTILS_HASH_H_

#include <stddef.h>
#include <stdint.h>
#include <limits>
#include <string>
#include <utility>
#include "platform/macros.h"
#include "platform/types.h"
#include "utils/stringpiece.h"

namespace bubblefs {
  
// Robert Jenkins' function for mixing 32-bit values
// http://burtleburtle.net/bob/hash/evahash.html
// a, b = random bits, c = input and output

#define hashmix(a,b,c) \
        a=a-b;  a=a-c;  a=a^(c>>13); \
        b=b-c;  b=b-a;  b=b^(a<<8);  \
        c=c-a;  c=c-b;  c=c^(b>>13); \
        a=a-b;  a=a-c;  a=a^(c>>12); \
        b=b-c;  b=b-a;  b=b^(a<<16); \
        c=c-a;  c=c-b;  c=c^(b>>5);  \
        a=a-b;  a=a-c;  a=a^(c>>3); \
        b=b-c;  b=b-a;  b=b^(a<<10); \
        c=c-a;  c=c-b;  c=c^(b>>15);
        
inline uint32_t rjhash32(uint32_t a) {
  a = (a+0x7ed55d16) + (a<<12);
  a = (a^0xc761c23c) ^ (a>>19);
  a = (a+0x165667b1) + (a<<5);
  a = (a+0xd3a2646c) ^ (a<<9);
  a = (a+0xfd7046c5) + (a<<3);
  a = (a^0xb55a4f09) ^ (a>>16);
  return a;
}        
        
inline uint64_t rjhash64(uint64_t key) {
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

// use like: rjhash<uint32_t> H;
template <class _Key> struct rjhash { };

template<> struct rjhash<uint32_t> {
  inline size_t operator()(const uint32_t x) const {
    return rjhash32(x);
  }
};

template<> struct rjhash<uint64_t> {
  inline size_t operator()(const uint64_t x) const {
    return rjhash64(x);
  }
};

/*
 *
template <>
struct hash< entity_addr_t >
{
  size_t operator()( const entity_addr_t& x ) const {
    static blobhash H;
    return H((const char*)&x, sizeof(x));
  }
};
 * */
class blobhash {
public:
  uint32_t operator()(const char *p, unsigned len) {
    static rjhash<uint32_t> H;
    uint32_t acc = 0;
    while (len >= sizeof(acc)) {
      acc ^= *(uint32_t*)p;
      p += sizeof(uint32_t);
      len -= sizeof(uint32_t);
    }
    int sh = 0;
    while (len) {
      acc ^= (uint32_t)*p << sh;
      sh += 8;
      len--;
      p++;
    }
    return H(acc);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////
  
extern uint32_t Hash(const char* data, size_t n, uint32_t seed);
extern uint32 Hash32(const char* data, size_t n, uint32 seed);
extern uint64 Hash64(const char* data, size_t n, uint64 seed);

inline uint64 Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

inline uint64 Hash64(const string& str) {
  return Hash64(str.data(), str.size());
}

inline uint64 Hash64Combine(uint64 a, uint64 b) {
  return a ^ (b + 0x9e3779b97f4a7800ULL + (a << 10) + (a >> 4));
}

inline uint32_t BloomHash(const StringPiece& key) {
  return Hash32(key.data(), key.size(), 0xbc9f1d34);
}

inline uint32_t GetStringPieceHash(const StringPiece& s) {
  return Hash32(s.data(), s.size(), 397);
}

// std::hash compatible interface.
// std::unordered_map<StringPiece, void*, StringPieceHasher> insert_hints_;
struct StringPieceHasher {
  uint32_t operator()(const StringPiece& s) const { return GetStringPieceHash(s); }
};

// Hash functor suitable for use with power-of-two sized hashtables.  Use
// instead of std::hash<T>.
//
// In particular, tensorflow::hash is not the identity function for pointers.
// This is important for power-of-two sized hashtables like FlatMap and FlatSet,
// because otherwise they waste the majority of their hash buckets.
template <typename T>
struct hash {
  size_t operator()(const T& t) const { return std::hash<T>()(t); }
};

template <typename T>
struct hash<T*> {
  size_t operator()(const T* t) const {
    // Hash pointers as integers, but bring more entropy to the lower bits.
    size_t k = static_cast<size_t>(reinterpret_cast<uintptr_t>(t));
    return k + (k >> 6);
  }
};

template <>
struct hash<string> {
  size_t operator()(const string& s) const {
    return static_cast<size_t>(Hash64(s));
  }
};

template <typename T, typename U>
struct hash<std::pair<T, U>> {
  size_t operator()(const std::pair<T, U>& p) const {
    return Hash64Combine(hash<T>()(p.first), hash<U>()(p.second));
  }
};

template <>
struct hash<StringPiece> {
  size_t operator()(StringPiece sp) const {
    return static_cast<size_t>(Hash64(sp.data(), sp.size()));
  }
};

}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_HASH_H_