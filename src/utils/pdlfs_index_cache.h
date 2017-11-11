/*
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/include/pdlfs-common/index_cache.h

#ifndef BUBBLEFS_UTILS_INDEX_CACHE_H_
#define BUBBLEFS_UTILS_INDEX_CACHE_H_

#include "platform/pdlfs_port.h"
#include "utils/pdlfs_gigaplus.h"
#include "utils/pdlfs_lru.h"
#include "utils/pdlfs_mdb.h"

namespace bubblefs {
namespace mypdlfs {

// An LRU-cache of directory indices.
class IndexCache {
  typedef LRUEntry<DirIndex> IndexEntry;

 public:
  // If mu is NULL, the resulting IndexCache requires external synchronization.
  // If mu is not NULL, the resulting IndexCache is implicitly synchronized
  // via it and is thread-safe.
  explicit IndexCache(size_t capacity = 4096, port::Mutex* mu = NULL);
  ~IndexCache();

  struct Handle {};
  void Release(Handle* handle);
  const DirIndex* Value(Handle* handle);

  Handle* Lookup(const DirId& id);
  Handle* Insert(const DirId& id, DirIndex* index);
  void Erase(const DirId& id);

 private:
  static Slice LRUKey(const DirId&, char* scratch);
  LRUCache<IndexEntry> lru_;
  port::Mutex* mu_;

  // No copying allowed
  void operator=(const IndexCache&);
  IndexCache(const IndexCache&);
};

}  // namespace mypdlfs
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_INDEX_CACHE_H_