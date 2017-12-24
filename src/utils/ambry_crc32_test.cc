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

// ambry/ambry-utils/src/test/java/com.github.ambry.utils/Crc32Test.java

#include "utils/ambry_crc32.h"
#include "platform/base_error.h"
#include "utils/pebble_random.h"

namespace bubblefs {
namespace myambry {
namespace utils { 

/**
 * Test to ensure that the checksum class works fine
 */
class Crc32Test {
  //@Test
 public:
  void crcTest() {
    Crc32 crc;
    char buf[4000] = { 0 };
    mypebble::TrueRandom rd;
    rd.NextBytes(buf, 4000);
    crc.update(buf, 0, 4000);
    long value1 = crc.getValue();
    crc.reset();
    crc.update(buf, 0, 4000);
    long value2 = crc.getValue();
    PRINTF_CHECK_EQ(value1, value2);
    buf[3999] = (char) (~buf[3999]);
    crc.reset();
    crc.update(buf, 0, 4000);
    long value3 = crc.getValue();
    PRINTF_CHECK_FALSE(value1 == value3);
  }
};

} // namespace utils
} // namespace myambry
} // namespace bubblefs

int main(int argc, char* argv[]) {
  bubblefs::myambry::utils::Crc32Test t;
  t.crcTest();
  return 0;
}