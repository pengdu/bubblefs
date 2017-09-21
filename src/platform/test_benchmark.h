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

// tensorflow/tensorflow/core/platform/test_benchmark.h

// Simple benchmarking facility.
#ifndef BUBBLEFS_PLATFORM_TEST_BENCHMARK_H_
#define BUBBLEFS_PLATFORM_TEST_BENCHMARK_H_

#include <utility>
#include <vector>
#include "platform/macros.h"
#include "platform/platform.h"
#include "platform/types.h"

#define BENCHMARK(n)                                            \
  static ::bubblefs::testing::Benchmark* TF_BENCHMARK_CONCAT( \
      __benchmark_, n, __LINE__) ALLOW_UNUSED =          \
      (new ::bubblefs::testing::Benchmark(#n, (n)))
#define TF_BENCHMARK_CONCAT(a, b, c) TF_BENCHMARK_CONCAT2(a, b, c)
#define TF_BENCHMARK_CONCAT2(a, b, c) a##b##c

namespace bubblefs {
namespace testing {

#if defined(PLATFORM_GOOGLE)

using ::testing::Benchmark;

#else

class Benchmark {
 public:
  Benchmark(const char* name, void (*fn)(int));
  Benchmark(const char* name, void (*fn)(int, int));
  Benchmark(const char* name, void (*fn)(int, int, int));

  Benchmark* Arg(int x);
  Benchmark* ArgPair(int x, int y);
  Benchmark* Range(int lo, int hi);
  Benchmark* RangePair(int lo1, int hi1, int lo2, int hi2);
  static void Run(const char* pattern);

 private:
  string name_;
  int num_args_;
  std::vector<std::pair<int, int>> args_;
  void (*fn0_)(int) = nullptr;
  void (*fn1_)(int, int) = nullptr;
  void (*fn2_)(int, int, int) = nullptr;

  void Register();
  void Run(int arg1, int arg2, int* run_count, double* run_seconds);
};
#endif // PLATFORM_GOOGLE

void RunBenchmarks();
void SetLabel(const std::string& label);
void BytesProcessed(int64);
void ItemsProcessed(int64);
void StartTiming();
void StopTiming();
void UseRealTime();

}  // namespace testing
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_TEST_BENCHMARK_H_