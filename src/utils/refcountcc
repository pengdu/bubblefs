// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/memory/ref_counted.cc

#include "utils/refcount.h"

namespace bubblefs {
namespace core {  
  
bool RefCountedThreadSafeBase::HasOneRef() const {
  return (ref_count_.load(std::memory_order_acquire) == 1);
}

RefCountedThreadSafeBase::RefCountedThreadSafeBase() : ref_count_(0) {
#ifndef NDEBUG
  in_dtor_ = false;
#endif
}

RefCountedThreadSafeBase::~RefCountedThreadSafeBase() {
#ifndef NDEBUG
  DCHECK(in_dtor_) << "RefCountedThreadSafe object deleted without "
                      "calling Release()";
  DCHECK_EQ(ref_.load(), 0);
#endif
}

void RefCountedThreadSafeBase::AddRef() const {
#ifndef NDEBUG
  DCHECK(!in_dtor_);
#endif
  DCHECK_GE(ref_count_.load(), 1);
  ref_count_.fetch_add(1, std::memory_order_relaxed);
}

bool RefCountedThreadSafeBase::Release() const {
#ifndef NDEBUG
  DCHECK(!in_dtor_);
  DCHECK_GT(ref_.load(), 0);
#endif
  if (HasOneRef() || ref_count_.fetch_sub(1) == 1) {
#ifndef NDEBUG
    in_dtor_ = true;
    DCHECK((ref_.store(0), true));
#endif
    return true;
  }
  return false;
}
  
} // namespace core  
} // namespace bubblefs