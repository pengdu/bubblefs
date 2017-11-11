/*
 * Copyright (c) 2011 The LevelDB Authors.
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/include/pdlfs-common/hash.h

#ifndef BUBBLEFS_UTILS_PDLFS_HASH_H_
#define BUBBLEFS_UTILS_PDLFS_HASH_H_

#include <stddef.h>
#include <stdint.h>

namespace bubblefs {
namespace mypdlfs {

/*
 * Simple hash function used by many of our internal data structures
 * such as LRU caches and bloom filters.
 *
 * Current implementation uses a schema similar to
 * the murmur hash.
 */
extern uint32_t Hash(const char* data, size_t n, uint32_t seed);

}  // namespace mypdlfs
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PDLFS_HASH_H_