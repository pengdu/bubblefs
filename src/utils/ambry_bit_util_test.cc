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

// ambry/ambry-utils/src/test/java/com.github.ambry.utils/BitUtilTest.java

#include "utils/ambry_bit_util.h"
#include "platform/base_error.h"

/**
 * Test for all the bit manipulation utils
 */

namespace bubblefs {
namespace myambry {
namespace utils {  
  
class BitUtilTest {
  //@Test
 public:
  void testBitUtil() {
    PRINTF_CHECK_TRUE(BitUtil::isPowerOfTwo(32));
    PRINTF_CHECK_FALSE(BitUtil::isPowerOfTwo(37));
    PRINTF_CHECK_TRUE(BitUtil::isPowerOfTwo(4096L));
    PRINTF_CHECK_FALSE(BitUtil::isPowerOfTwo(4097L));
    PRINTF_CHECK_EQ(BitUtil::nextHighestPowerOfTwo(62), 64);
    PRINTF_CHECK_EQ(BitUtil::nextHighestPowerOfTwo(4092L), 4096);

    PRINTF_CHECK_EQ(BitUtil::ntz(0x24), 2);
    PRINTF_CHECK_EQ(BitUtil::ntz(0x1000), 12);
    PRINTF_CHECK_EQ(BitUtil::ntz2(0x1000), 12);
    PRINTF_CHECK_EQ(BitUtil::ntz3(0x1000), 12);
    PRINTF_CHECK_EQ(BitUtil::pop(0x1000), 1);
    long words1[2];
    words1[0] = 0x4f2;
    words1[1] = 0x754;
    long words2[2];
    words2[0] = 0x1de6;
    words2[1] = 0xa07;
    PRINTF_CHECK_EQ(BitUtil::pop_andnot(words1, words2, 0, 2), 5);
    PRINTF_CHECK_EQ(BitUtil::pop_array(words1, 0, 2), 12);
    PRINTF_CHECK_EQ(BitUtil::pop_intersect(words1, words2, 0, 2), 7);
    PRINTF_CHECK_EQ(BitUtil::pop_union(words1, words2, 0, 2), 19);
    PRINTF_CHECK_EQ(BitUtil::pop_xor(words1, words2, 0, 2), 12);
    
    PRINTF_TEST_DONE();
  }
};

} // namespace utils
} // namespace myambry
} // namespace bubblefs

int main(int argc, char* argv[]) {
  bubblefs::myambry::utils::BitUtilTest t;
  t.testBitUtil();
  return 0;
}