// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

//  Get code from re2 project

// toft/base/random.h

#ifndef BUBBLEFS_UTILS_TOFT_BASE_RANDOM_H_
#define BUBBLEFS_UTILS_TOFT_BASE_RANDOM_H_

#include <stdint.h>
#include "utils/toft_base_uncopyable.h"

namespace bubblefs {
namespace mytoft {

// ACM minimal standard random number generator.  (re-entrant.)
class Random {
    DECLARE_UNCOPYABLE(Random);

public:
    explicit Random(int32_t seed) : seed_(seed) {}

    int32_t Next();
    int32_t Uniform(int32_t n);

    void Reset(int32_t seed) {
        seed_ = seed;
    }

    // Randomly returns true ~"1/n" of the time, and false otherwise.
    // REQUIRES: n > 0
    bool OneIn(int n) { return (Next() % n) == 0; }

    // Skewed: pick "base" uniformly from range [0,max_log] and then
    // return "base" random bits.  The effect is to pick a number in the
    // range [0,2^max_log-1] with exponential bias towards smaller numbers.
    uint32_t Skewed(int max_log) {
        return Uniform(1 << Uniform(max_log + 1));
    }

private:
    int32_t seed_;
};

}  // namespace mytoft
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_TOFT_BASE_RANDOM_H_