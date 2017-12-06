// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// AirSim/AirLib/include/common/common_utils/RandomGenerator.hpp

#ifndef BUBBLEFS_UTILS_AIRSIM_RANDOM_GENERATOR_H_
#define BUBBLEFS_UTILS_AIRSIM_RANDOM_GENERATOR_H_

#include <random>

namespace bubblefs {
namespace myairsim {
namespace common_utils {

template<typename TReturn, typename TDistribution, unsigned int Seed=42>
class RandomGenerator {
public:
    //for uniform distribution supply min and max (inclusive)
    //for gaussian distribution supply mean and sigma
    template<typename... DistArgs>
    RandomGenerator(DistArgs... dist_args)
        : dist_(dist_args...), rand_(Seed)
    {
    }

    void seed(int val)
    {
        rand_.seed(val);
    }

    TReturn next()
    {
        return static_cast<TReturn>(dist_(rand_));
    }

    void reset()
    {
        rand_.seed(Seed);
        dist_.reset();
    }

private:
    TDistribution dist_;
    std::mt19937 rand_;
};

typedef RandomGenerator<double, std::uniform_real_distribution<double>> RandomGeneratorD;
typedef RandomGenerator<float, std::uniform_real_distribution<float>> RandomGeneratorF;
typedef RandomGenerator<int, std::uniform_int_distribution<int>> RandomGeneratorI;
//TODO: below we should have float instead of double but VC++2017 has a bug :(
typedef RandomGenerator<float, std::normal_distribution<double>> RandomGeneratorGaussianF;
typedef RandomGenerator<double, std::normal_distribution<double>> RandomGeneratorGaussianD;

} // namespace common_utils
} // namespace myairsim
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_AIRSIM_RANDOM_GENERATOR_H_