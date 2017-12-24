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

// ambry/ambry-utils/src/main/java/com.github.ambry.utils/FileLock.java

#ifndef BUBBLEFS_UTILS_AMBRY_FILELOCK_H_
#define BUBBLEFS_UTILS_AMBRY_FILELOCK_H_

#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <mutex>

/**
 * File Lock helper
 * thread-safe file lock
 */

namespace bubblefs {
namespace myambry {
namespace utils {
  
class FileLock {
 public:
   FileLock(int _fd) : fd_(_fd) {}
   bool Lock();
   bool TryLock();
   bool Unlock();
   
 private:
   static int LockOrUnlock(int fd, bool lock, bool wait);

 private:  
  int fd_; // java.nio.channels.FileChannel = new java.io.RandomAccessFile((java.io.File)file, "rw").getChannel()
  std::mutex mu_; // for thread-safe
};

/**
 * process-safe filelock
 * fcntl: Any number of processes may hold a read
 * lock (shared lock) on a file region, but only one process may hold a
 * write lock (exclusive lock).  An exclusive lock excludes all other
 * locks, both shared and exclusive.  A single process can hold only one
 * type of lock on a file region; if a new lock is applied to an
 * already-locked region, then the existing lock is converted to the new lock type. 
 */
int FileLock::LockOrUnlock(int fd, bool lock, bool wait) {
  struct flock f;
  memset(&f, 0, sizeof(f));
  // As well as being removed by an explicit F_UNLCK,
  // Record locks are automatically released when the process terminates or close(fd)
  f.l_type = (lock ? F_WRLCK : F_UNLCK);
  f.l_whence = SEEK_SET; // the start of the file
  f.l_start = 0;      
  f.l_len = 0;        // the number of bytes to be locked, Lock/unlock entire file
  // 1. F_SETLK: If a conflicting lock is held by another process, 
  // this call returns -1 and sets errno to EACCES or EAGAIN.
  // 2. F_SETLKW: but if a conflicting lock is held on the file,
  // then wait for that lock to be released.  If a signal is caught
  // while waiting, then the call is interrupted and (after the
  // signal handler has returned) returns immediately (with return
  // value -1 and errno set to EINTR;
  int setlk_type = wait ? F_SETLKW : F_SETLK;
  return fcntl(fd, setlk_type, &f);
}

bool FileLock::Lock() {
  std::lock_guard<std::mutex> lock(mu_);
  return LockOrUnlock(fd_, true, true) == 0;
}

bool FileLock::TryLock() {
  std::lock_guard<std::mutex> lock(mu_);
  return LockOrUnlock(fd_, true, false) == 0;
}

bool FileLock::Unlock() {
  std::lock_guard<std::mutex> lock(mu_);
  return LockOrUnlock(fd_, false, false) == 0;
}

} // namespace utils
} // namespace myambry  
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_AMBRY_FILELOCK_H_