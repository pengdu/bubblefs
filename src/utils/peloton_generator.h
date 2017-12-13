//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// generator.h
//
// Identification: src/include/common/generator.h
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/include/common/generator.h

#ifndef BUBBLEFS_UTILS_PELOTON_GENERATOR_H_
#define BUBBLEFS_UTILS_PELOTON_GENERATOR_H_

#include <random>

//===--------------------------------------------------------------------===//
// Generator
//===--------------------------------------------------------------------===//

namespace bubblefs {
namespace mypeloton {
  
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

} // namespace mypeloton
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_GENERATOR_H_