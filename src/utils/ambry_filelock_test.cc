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

// ambry/ambry-utils/src/test/java/com.github.ambry.utils/FileLockTest.java

#include <unistd.h> // close
#include "utils/ambry_filelock.h"
#include "platform/base_error.h"

/**
 * Tests for file lock
 */

namespace bubblefs {
namespace myambry {
namespace utils { 
  
class FileLockTest {
  //@Test
 public:
  void testFileLock() {
    int fd = open("temp", O_RDWR | O_CREAT, 0644);
    if (fd < 0) return; // file.deleteOnExit();
    FileLock lock(fd);
    lock.TryLock();
    PRINTF_CHECK_FALSE(lock.TryLock());
    lock.Unlock();
    PRINTF_CHECK_TRUE(lock.TryLock());
    lock.Unlock();
    close(fd);
  }
};

} // namespace utils
} // namespace myambry
} // namespace bubblefs

int main(int argc, char* argv[]) {
  bubblefs::myambry::utils::FileLockTest t;
  t.testFileLock();
  return 0;
}