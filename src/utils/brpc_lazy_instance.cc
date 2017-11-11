// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/lazy_instance.cc

#include "utils/brpc_lazy_instance.h"
#include "platform/atomicops.h"
#include "platform/dynamic_annotations.h"
#include "platform/types.h"
#include "utils/brpc_at_exit.h"

namespace bubblefs {
namespace mybrpc {
namespace internal {

using namespace bubblefs::base;  
  
// TODO(joth): This function could be shared with Singleton, in place of its
// WaitForInstance() call.
bool NeedsLazyInstance(AtomicWord* state) {
  // Try to create the instance, if we're the first, will go from 0 to
  // kLazyInstanceStateCreating, otherwise we've already been beaten here.
  // The memory access has no memory ordering as state 0 and
  // kLazyInstanceStateCreating have no associated data (memory barriers are
  // all about ordering of memory accesses to *associated* data).
  if (base::NoBarrier_CompareAndSwap(state, 0,
                               kLazyInstanceStateCreating) == 0)
    // Caller must create instance
    return true;

  // It's either in the process of being created, or already created. Spin.
  // The load has acquire memory ordering as a thread which sees
  // state_ == STATE_CREATED needs to acquire visibility over
  // the associated data (buf_). Pairing Release_Store is in
  // CompleteLazyInstance().
  while (base::Acquire_Load(state) == kLazyInstanceStateCreating) {
    sched_yield();
  }
  // Someone else created the instance.
  return false;
}

void CompleteLazyInstance(AtomicWord* state,
                          AtomicWord new_instance,
                          void* lazy_instance,
                          void (*dtor)(void*)) {
  // See the comment to the corresponding HAPPENS_AFTER in Pointer().
  ANNOTATE_HAPPENS_BEFORE(state);

  // Instance is created, go from CREATING to CREATED.
  // Releases visibility over private_buf_ to readers. Pairing Acquire_Load's
  // are in NeedsInstance() and Pointer().
  base::Release_Store(state, new_instance);

  // Make sure that the lazily instantiated object will get destroyed at exit.
  if (dtor)
    AtExitManager::RegisterCallback(dtor, lazy_instance);
}

}  // namespace internal
}  // namespace mybrpc
}  // namespace bubblefs