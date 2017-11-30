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

// tensorflow/tensorflow/core/lib/random/random.cc
// Pebble/src/common/random.cpp

#include "utils/random.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <functional>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>
#include "platform/macros.h"
#include "platform/mutex.h"
#include "platform/types.h"

namespace bubblefs {
namespace random {

namespace {
std::mt19937_64* InitRngWithRandomSeed() {
  std::random_device device("/dev/urandom");
  return new std::mt19937_64(device());
}
std::mt19937_64 InitRngWithDefaultSeed() { return std::mt19937_64(); }

}  // anonymous namespace

uint64 New64() {
  static std::mt19937_64* rng = InitRngWithRandomSeed();
  static mutex mu;
  mutex_lock l(mu);
  return (*rng)();
}

uint64 New64DefaultSeed() {
  static std::mt19937_64 rng = InitRngWithDefaultSeed();
  static mutex mu;
  mutex_lock l(mu);
  return rng();
}

double BitsToOpenEndedUnitInterval(uint64_t bits) {
  // We try to get maximum precision by masking out as many bits as will fit
  // in the target type's mantissa, and raising it to an appropriate power to
  // produce output in the range [0, 1).  For IEEE 754 doubles, the mantissa
  // is expected to accommodate 53 bits.

  static_assert(std::numeric_limits<double>::radix == 2,
                "otherwise use scalbn");
  static const int kBits = std::numeric_limits<double>::digits;
  uint64_t random_bits = bits & ((UINT64_C(1) << kBits) - 1);
  double result = ldexp(static_cast<double>(random_bits), -1 * kBits);
  if (result <= 0.0)
    return 0.0;
  if (result >= 1.0)
    return 1.0;
  return result;
}

uint64_t RandGenerator(uint64_t range) {
  assert(range > 0);
  // We must discard random results above this number, as they would
  // make the random generator non-uniform (consider e.g. if
  // MAX_UINT64 was 7 and |range| was 5, then a result of 1 would be twice
  // as likely as a result of 3 or 4).
  uint64_t max_acceptable_value =
      (std::numeric_limits<uint64_t>::max() / range) * range - 1;

  uint64_t value;
  do {
    value = New64();
  } while (value > max_acceptable_value);

  return value % range;
}

int RandInt(int min, int max) {
  assert(min < max);

  uint64_t range = static_cast<uint64_t>(max) - min + 1;
  // |range| is at most UINT_MAX + 1, so the result of RandGenerator(range)
  // is at most UINT_MAX.  Hence it's safe to cast it from uint64_t to int64_t.
  int result =
      static_cast<int>(min + static_cast<int64_t>(RandGenerator(range)));
  if (min <= result && result <= max)
    return result;
  return 0; // invalid
}

double RandDouble() {
  return BitsToOpenEndedUnitInterval(New64());
}

// rocksdb/util/random.cc

Random* Random::GetTLSInstance() {
  static __thread Random* tls_instance;
  static __thread std::aligned_storage<sizeof(Random)>::type tls_instance_bytes;

  auto rv = tls_instance;
  if (UNLIKELY(rv == nullptr)) {
    size_t seed = std::hash<std::thread::id>()(std::this_thread::get_id());
    rv = new (&tls_instance_bytes) Random((uint32_t)seed);
    tls_instance = rv;
  }
  return rv;
}

}  // namespace random
}  // namespace bubblefs