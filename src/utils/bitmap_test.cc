/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

// tensorflow/tensorflow/core/lib/core/bitmap_test.cc

#include "utils/bitmap.h"
#include "platform/macros.h"
#include "platform/test.h"
#include "utils/philox_random.h"

namespace bubblefs {
namespace core {
namespace { // namespace anonymous

// Return next size to test after n.
size_t NextSize(size_t n) { return n + ((n < 75) ? 1 : 25); }

TEST(BitmapTest, Basic) {
  for (size_t n = 0; n < 200; n = NextSize(n)) {
    Bitmap bits(n);
    for (size_t i = 0; i < n; i++) {
      EXPECT_FALSE(bits.get(i)) << n << " " << i << " " << bits.ToString();
      bits.set(i);
      EXPECT_TRUE(bits.get(i)) << n << " " << i << " " << bits.ToString();
      bits.clear(i);
      EXPECT_FALSE(bits.get(i)) << n << " " << i << " " << bits.ToString();
    }
  }
}

TEST(BitmapTest, ToString) {
  Bitmap bits(10);
  bits.set(1);
  bits.set(3);
  EXPECT_EQ(bits.ToString(), "0101000000");
}

TEST(BitmapTest, FirstUnset) {
  for (size_t n = 0; n < 200; n = NextSize(n)) {
    for (size_t p = 0; p <= 100; p++) {
      for (size_t q = 0; q <= 100; q++) {
        // Generate a bitmap of length n with long runs of ones.
        Bitmap bitmap(n);
        // Set first p bits to 1.
        int one_count = 0;
        size_t i = 0;
        while (i < p && i < n) {
          one_count++;
          bitmap.set(i);
          i++;
        }
        // Fill rest with a pattern of 0 followed by q 1s.
        while (i < n) {
          i++;
          for (size_t j = 0; j < q && i < n; j++, i++) {
            one_count++;
            bitmap.set(i);
          }
        }

        // Now use FirstUnset to iterate over unset bits and verify
        // that all encountered bits are clear.
        int seen = 0;
        size_t pos = 0;
        while (true) {
          pos = bitmap.FirstUnset(pos);
          if (pos == n) break;
          ASSERT_FALSE(bitmap.get(pos)) << pos << " " << bitmap.ToString();
          seen++;
          pos++;
        }
        EXPECT_EQ(seen, n - one_count) << " " << bitmap.ToString();
      }
    }
  }
}

}  // namespace anonymous
}  // namespace core
}  // namespace bubblefs