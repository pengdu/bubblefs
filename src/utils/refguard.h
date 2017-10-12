/*
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/include/pdlfs-common/guard.h

#ifndef BUBBLEFS_UTILS_REFGUARD_H_
#define BUBBLEFS_UTILS_REFGUARD_H_

#include <assert.h>

namespace bubblefs {

template <typename T, typename Ref>
class RefGuard {
  Ref* ref_;
  T* src_;

 public:
  explicit RefGuard(T* t, Ref* r) : ref_(r), src_(t) {}
  ~RefGuard() {
    if (ref_ != nullptr) {
      src_->Release(ref_);
    }
  }

 private: 
  // No copying allowed
  void operator=(const RefGuard&);
  RefGuard(const RefGuard&); 
  
};

}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_REFGUARD_H_