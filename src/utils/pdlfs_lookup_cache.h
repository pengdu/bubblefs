/*
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/include/pdlfs-common/lookup_cache.h

#ifndef BUBBLEFS_UTILS_PDLFS_LOOKUP_CACHE_H_
#define BUBBLEFS_UTILS_PDLFS_LOOKUP_CACHE_H_

#include "utils/pdlfs_mdb.h"
#include "utils/pdlfs_fstypes.h"
#include "utils/pdlfs_lru.h"
#include "platform/pdlfs_port.h"

namespace bubblefs {
namespace mypdlfs {

// An LRU-cache of pathname lookup leases.
class LookupCache {
  typedef LRUEntry<LookupStat> LookupEntry;

 public:
  // If mu is NULL, the resulting LookupCache requires external synchronization.
  // If mu is not NULL, the resulting LookupCache is implicitly synchronized
  // via it and is thread-safe.
  explicit LookupCache(size_t capacity = 4096, port::Mutex* mu = NULL);
  ~LookupCache();

  struct Handle {};
  void Release(Handle* handle);
  LookupStat* Value(Handle* handle);

  Handle* Lookup(const DirId& pid, const Slice& nhash);
  Handle* Insert(const DirId& pid, const Slice& nhash, LookupStat* stat);
  void Erase(const DirId& pid, const Slice& nhash);

 private:
  static Slice LRUKey(const DirId&, const Slice&, char* scratch);
  LRUCache<LookupEntry> lru_;
  port::Mutex* mu_;

  // No copying allowed
  void operator=(const LookupCache&);
  LookupCache(const LookupCache&);
};

}  // namespace mypdlfs
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PDLFS_LOOKUP_CACHE_H_