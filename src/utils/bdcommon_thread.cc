// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "utils/bdcommon_thread.h"

namespace bubblefs {
namespace bdcommon {

bool Thread::Start(Proc proc, void* arg, size_t stack_size, bool joinable) {
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  // Pthreads are joinable by default, so only specify the detached
  // attribute if the thread should be non-joinable.
  if (!joinable) {
    pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
  }
  // Get a better default if available.
  if (stack_size <= 0)
    stack_size = 2 * (1 << 23);  // 2 times 8192K (the default stack size on Linux).
  pthread_attr_setstacksize(&attributes, stack_size);
        
  // The child thread will inherit our signal mask.
  // Set our signal mask to the set of signals we want to block ?
        
  int ret = pthread_create(&tid_, nullptr, proc, arg);
        
  pthread_attr_destroy(&attributes);
  return (ret == 0);
}
  
} // namespace bdcommon  
} // namespace bubblefs