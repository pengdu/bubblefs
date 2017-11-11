/*
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/include/pdlfs-common/guard.h

#ifndef BUBBLEFS_UTILS_PDLFS_GUARD_H_
#define BUBBLEFS_UTILS_PDLFS_GUARD_H_

#include <assert.h>

namespace bubblefs {
namespace mypdlfs {

template <typename T, typename Ref>
class RefGuard {
  Ref* ref_;
  T* src_;

  // No copying allowed
  void operator=(const RefGuard&);
  RefGuard(const RefGuard&);

 public:
  explicit RefGuard(T* t, Ref* r) : ref_(r), src_(t) {}
  ~RefGuard() {
    if (ref_ != NULL) {
      src_->Release(ref_);
    }
  }
};

}  // namespace mypdlfs
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PDLFS_GUARD_H_