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

// tensorflow/tensorflow/core/platform/mutex.h
// tensorflow/tensorflow/core/platform/default/mutex.h

#ifndef BUBBLEFS_PLATFORM_MUTEX_H_
#define BUBBLEFS_PLATFORM_MUTEX_H_

#include <chrono>
#include <condition_variable>
#include <mutex>
#include "platform/platform.h"
#include "platform/thread_annotations.h"
#include "platform/types.h"

namespace bubblefs {
  
enum ConditionResult { kCond_Timeout, kCond_MaybeNotified };

#undef mutex_lock

enum LinkerInitialized { LINKER_INITIALIZED };

// A class that wraps around the std::mutex implementation, only adding an
// additional LinkerInitialized constructor interface.
class LOCKABLE mutex : public std::mutex {
 public:
  mutex() {}
  // The default implementation of std::mutex is safe to use after the linker
  // initializations
  explicit mutex(LinkerInitialized x) {}

  void lock() ACQUIRE() { std::mutex::lock(); }
  bool try_lock() EXCLUSIVE_TRYLOCK_FUNCTION(true) {
    return std::mutex::try_lock();
  };
  void unlock() RELEASE() { std::mutex::unlock(); }
};

class SCOPED_LOCKABLE mutex_lock : public std::unique_lock<std::mutex> {
 public:
  mutex_lock(class mutex& m) ACQUIRE(m) : std::unique_lock<std::mutex>(m) {}
  mutex_lock(class mutex& m, std::try_to_lock_t t) ACQUIRE(m)
      : std::unique_lock<std::mutex>(m, t) {}
  mutex_lock(mutex_lock&& ml) noexcept
      : std::unique_lock<std::mutex>(std::move(ml)) {}
  ~mutex_lock() RELEASE() {}
};

// Catch bug where variable name is omitted, e.g. mutex_lock (mu);
#define mutex_lock(x) static_assert(0, "mutex_lock_decl_missing_var_name");

using std::condition_variable;

inline ConditionResult WaitForMilliseconds(mutex_lock* mu,
                                           condition_variable* cv, int64 ms) {
  std::cv_status s = cv->wait_for(*mu, std::chrono::milliseconds(ms));
  return (s == std::cv_status::timeout) ? kCond_Timeout : kCond_MaybeNotified;
}

// The mutex library included above defines:
//   class mutex;
//   class mutex_lock;
//   class condition_variable;
// It also defines the following:

// Like "cv->wait(*mu)", except that it only waits for up to "ms" milliseconds.
//
// Returns kCond_Timeout if the timeout expired without this
// thread noticing a signal on the condition variable.  Otherwise may
// return either kCond_Timeout or kCond_MaybeNotified
ConditionResult WaitForMilliseconds(mutex_lock* mu, condition_variable* cv,
                                    int64 ms);
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_MUTEX_H_
