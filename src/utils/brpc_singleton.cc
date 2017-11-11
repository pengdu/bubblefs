// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/memory/singleton.cc

#include "utils/brpc_singleton.h"

namespace bubblefs {
namespace mybrpc {
namespace internal {

base::AtomicWord WaitForInstance(base::AtomicWord* instance) {
  // Handle the race. Another thread beat us and either:
  // - Has the object in BeingCreated state
  // - Already has the object created...
  // We know value != NULL.  It could be kBeingCreatedMarker, or a valid ptr.
  // Unless your constructor can be very time consuming, it is very unlikely
  // to hit this race.  When it does, we just spin and yield the thread until
  // the object has been created.
  base::AtomicWord value;
  while (true) {
    // The load has acquire memory ordering as the thread which reads the
    // instance pointer must acquire visibility over the associated data.
    // The pairing Release_Store operation is in Singleton::get().
    value = base::Acquire_Load(instance);
    if (value != kBeingCreatedMarker)
      break;
    sched_yield(); //concurrent::PlatformThread::YieldCurrentThread();
  }
  return value;
}

}  // namespace internal
}  // namespace mybrpc
}  // namespace bubblefs