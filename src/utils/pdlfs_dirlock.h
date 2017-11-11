/*
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/include/pdlfs-common/dirlock.h

#ifndef BUBLEFS_UTILS_PDLFS_DIRLOCK_H_
#define BUBLEFS_UTILS_PDLFS_DIRLOCK_H_

#include "utils/pdlfs_dcntl.h"

namespace bubblefs {
namespace pdlfs {

// Helper class that locks a directory on construction and
// unlocks the directory when the destructor of the DirLock
// object is invoked.
class DirLock {
 public:
  explicit DirLock(Dir* d) : dir_(d) { dir_->Lock(); }

  ~DirLock() { dir_->Unlock(); }

 private:
  // No copying allowed
  void operator=(const DirLock&);
  DirLock(const DirLock&);

  Dir* dir_;
};

}  // namespace pdlfs
}  // namespace bubblefs

#endif // BUBLEFS_UTILS_PDLFS_DIRLOCK_H_