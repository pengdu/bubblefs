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
//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
/*
 * Ceph - scalable distributed file system
 *
 * Copyright (C) 2004-2006 Sage Weil <sage@newdream.net>
 *
 * This is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1, as published by the Free Software 
 * Foundation.  See file COPYING.
 * 
 */

// tensorflow/tensorflow/core/lib/random/random.h
// cocos2d-x/cocos/base/ccRandom.h
// ceph/src/include/Distribution.h

#ifndef BUBBLEFS_UTILS_RANDOM_H_
#define BUBBLEFS_UTILS_RANDOM_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <random>
#include <string>
#include <vector>
#include "platform/macros.h"
#include "platform/types.h"

namespace bubblefs {
namespace random {

// Return a 64-bit random value.  Different sequences are generated
// in different processes.
// Returns a random number in range [0, UINT64_MAX]. Thread-safe.
uint64 New64();

// Return a 64-bit random value. Uses
// std::mersenne_twister_engine::default_seed as seed value.
uint64 New64DefaultSeed();

// Returns a random number in range [0, range).  Thread-safe.
//
// Note that this can be used as an adapter for std::random_shuffle():
// Given a pre-populated |std::vector<int> myvector|, shuffle it as
//   std::random_shuffle(myvector.begin(), myvector.end(), base::RandGenerator);
BASE_EXPORT uint64_t RandGenerator(uint64_t range);

// Returns a random number between min and max (inclusive). Thread-safe.
BASE_EXPORT int RandInt(int min, int max);

// Returns a random double in range [0, 1). Thread-safe.
BASE_EXPORT double RandDouble();

///////////////////////////////////////////////////////////

// A very simple random number generator.  Not especially good at
// generating truly random bits, but good enough for our needs in this
// package.
class Random {
 private:
  enum : uint32_t {
    M = 2147483647L  // 2^31-1
  };
  enum : uint64_t {
    A = 16807  // bits 14, 8, 7, 5, 2, 1, 0
  };

  uint32_t seed_;

  static uint32_t GoodSeed(uint32_t s) { return (s & M) != 0 ? (s & M) : 1; }

 public:
  // This is the largest value that can be returned from Next()
  enum : uint32_t { kMaxNext = M };

  explicit Random(uint32_t s) : seed_(GoodSeed(s)) {}

  void Reset(uint32_t s) { seed_ = GoodSeed(s); }

  uint32_t Next() {
    // We are computing
    //       seed_ = (seed_ * A) % M,    where M = 2^31-1
    //
    // seed_ must not be zero or M, or else all subsequent computed values
    // will be zero or M respectively.  For all other values, seed_ will end
    // up cycling through every number in [1,M-1]
    uint64_t product = seed_ * A;

    // Compute (product % M) using the fact that ((x << 31) % M) == x.
    seed_ = static_cast<uint32_t>((product >> 31) + (product & M));
    // The first reduction may overflow by 1 bit, so we may need to
    // repeat.  mod == M is not possible; using > allows the faster
    // sign-bit-based test.
    if (seed_ > M) {
      seed_ -= M;
    }
    return seed_;
  }

  // Returns a uniformly distributed value in the range [0..n-1]
  // REQUIRES: n > 0
  uint32_t Uniform(int n) { return Next() % n; }

  // Randomly returns true ~"1/n" of the time, and false otherwise.
  // REQUIRES: n > 0
  bool OneIn(int n) { return (Next() % n) == 0; }

  // Skewed: pick "base" uniformly from range [0,max_log] and then
  // return "base" random bits.  The effect is to pick a number in the
  // range [0,2^max_log-1] with exponential bias towards smaller numbers.
  uint32_t Skewed(int max_log) {
    return Uniform(1 << Uniform(max_log + 1));
  }

  // Returns a Random instance for use by the current thread without
  // additional locking
  static Random* GetTLSInstance();
};

// A simple 64bit random number generator based on std::mt19937_64
class Random64 {
 private:
  std::mt19937_64 generator_;

 public:
  explicit Random64(uint64_t s) : generator_(s) { }

  // Generates the next random number
  uint64_t Next() { return generator_(); }

  // Returns a uniformly distributed value in the range [0..n-1]
  // REQUIRES: n > 0
  uint64_t Uniform(uint64_t n) {
    return std::uniform_int_distribution<uint64_t>(0, n - 1)(generator_);
  }

  // Randomly returns true ~"1/n" of the time, and false otherwise.
  // REQUIRES: n > 0
  bool OneIn(uint64_t n) { return Uniform(n) == 0; }

  // Skewed: pick "base" uniformly from range [0,max_log] and then
  // return "base" random bits.  The effect is to pick a number in the
  // range [0,2^max_log-1] with exponential bias towards smaller numbers.
  uint64_t Skewed(int max_log) {
    return Uniform(uint64_t(1) << Uniform(max_log + 1));
  }
};

class RandomString {
public:
    static std::string Rand(size_t len = 128) {
        std::string s;
        s.resize(len);
        static Random r(uint32_t(time(NULL)));
        const char* end = &s[0] + s.size();
        for (char* p = &s[0]; p < end; ++p) {
            *p = r.Next() % 255;
        }
        return s;
    }
};

// peloton/src/include/common/generator.h
class UniformGenerator {
 public:
  UniformGenerator() { unif = std::uniform_real_distribution<double>(0, 1); }

  UniformGenerator(double lower_bound, double upper_bound) {
    unif = std::uniform_real_distribution<double>(lower_bound, upper_bound);
  }

  double GetSample() { return unif(rng); }

 private:
  // Random number generator
  std::mt19937_64 rng;

  // Distribution
  std::uniform_real_distribution<double> unif;
};

class Distribution {
  std::vector<float> p;
  std::vector<int> v;

 public:  
  unsigned GetWidth() {
    return p.size();
  }

  void Clear() {
    p.clear();
    v.clear();
  }
  void Add(int val, float pr) {
    p.push_back(pr);
    v.push_back(val);
  }

  void Random() {
    float sum = 0.0;
    for (unsigned i = 0; i < p.size(); i++) {
      p[i] = (float)(rand() % 10000);
      sum += p[i];
    }
    for (unsigned i = 0; i < p.size(); i++) 
      p[i] /= sum;
  }

  int Sample() {
    float s = (float)(rand() % 10000) / 10000.0;
    for (unsigned i = 0; i < p.size(); i++) {
      if (s < p[i]) return v[i];
      s -= p[i];
    }
    abort();
    return v[p.size() - 1];
  }

  float Normalize() {
    float s = 0.0;
    for (unsigned i = 0; i < p.size(); i++)
      s += p[i];
    for (unsigned i=0; i < p.size(); i++)
      p[i] /= s;
    return s;
  }

};

}  // namespace random
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_RANDOM_H_