// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.
//

// slash/slash/include/cond_lock.h

#ifndef BUBBLEFS_PLATFORM_SLASH_COND_LOCK_H_
#define BUBBLEFS_PLATFORM_SLASH_COND_LOCK_H_

/*
 * CondLock is a wrapper for condition variable.
 * It contain a mutex in it's class, so we don't need other to protect the 
 * condition variable.
 */
#include <pthread.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

namespace bubblefs {
namespace myslash {

class CondLock {
 public:
  CondLock();
  ~CondLock();

  void Lock();
  void Unlock();

  void Wait();
  
  /*
   * timeout is millisecond
   */
  void TimedWait(uint32_t timeout);
  void Signal();
  void Broadcast();

 private:
  pthread_mutex_t mutex_;
  pthread_cond_t cond_;

  CondLock(const CondLock&) {};
  void operator =(const CondLock&) {};
};

}  // namespace myslash
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_SLASH_COND_LOCK_H_