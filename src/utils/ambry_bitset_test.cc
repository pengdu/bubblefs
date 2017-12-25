/**
 * Copyright 2016 LinkedIn Corp. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

// ambry/ambry-utils/src/test/java/com.github.ambry.utils/OpenBitSetTest.java

#include "utils/ambry_bitset.h"
#include "platform/base_error.h"

/**
 * OpenBitSet Test
 */
namespace bubblefs {
namespace myambry {
namespace utils { 

class OpenBitSetTest {
  //@Test
 public:
   void testOpenBitSetTest() {
    OpenBitSet bitSet(1000);
    bitSet.set(0);
    bitSet.set(100);
    PRINTF_CHECK_TRUE(bitSet.get(0));
    PRINTF_CHECK_TRUE(bitSet.get(100));
    PRINTF_CHECK_FALSE(bitSet.get(1));
    bitSet.clear(0);
    PRINTF_CHECK_FALSE(bitSet.get(0));
    PRINTF_CHECK_EQ(bitSet.capacity(), 1024);
    PRINTF_CHECK_EQ(bitSet.size(), 1024);
    PRINTF_CHECK_EQ(bitSet.length(), 1024);
    PRINTF_CHECK_EQ(bitSet.isEmpty(), false);
    PRINTF_CHECK_EQ(bitSet.cardinality(), 1);
    OpenBitSet bitSet2(1000);
    bitSet2.set(100);
    bitSet2.set(1);
    bitSet2.andWith(bitSet);
    PRINTF_CHECK_TRUE(bitSet2.get(100));
    PRINTF_CHECK_FALSE(bitSet2.get(1));
    bitSet2.intersect(bitSet);
    PRINTF_CHECK_TRUE(bitSet2.get(100));
    OpenBitSet bitSet3(1000);
    bitSet3.set(100);
    PRINTF_CHECK_TRUE(bitSet2.equals(&bitSet3));
    bitSet3.set(101);
    bitSet3.set(102);
    bitSet3.set(103);
    bitSet3.clear(100, 104);
    PRINTF_CHECK_FALSE(bitSet3.get(100));
    PRINTF_CHECK_FALSE(bitSet3.get(101));
    PRINTF_CHECK_FALSE(bitSet3.get(102));
    PRINTF_CHECK_FALSE(bitSet3.get(103));
  }
};

} // namespace utils
} // namespace myambry
} // namespace bubblefs

int main(int argc, char* argv[]) {
  bubblefs::myambry::utils::OpenBitSetTest t;
  t.testOpenBitSetTest();
  return 0;
}