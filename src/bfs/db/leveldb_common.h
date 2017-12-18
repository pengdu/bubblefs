// Copyright (c) 2014 The IndexFS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// indexfs/common/options.h

#ifndef BUBBLEFS_DB_LEVELDB_COMMON_H_
#define BUBBLEFS_DB_LEVELDB_COMMON_H_

#include "leveldb/cache.h"
#include "leveldb/db.h"
#include "leveldb/env.h"
#include "leveldb/options.h"
#include "leveldb/status.h"
#include "leveldb/slice.h"
#include "leveldb/write_batch.h"

#define DEFAULT_LEVELDB_COMPRESSION false
#define DEFAULT_LEVELDB_SMALLFILE_THRESHOLD     65536
#define DEFAULT_LEVELDB_MONITORING      false
#define DEFAULT_LEVELDB_SAMPLING_INTERVAL  1
#define DEFAULT_LEVELDB_FILTER_BYTES    14
#define DEFAULT_LEVELDB_MAX_OPEN_FILES  128
#define DEFAULT_LEVELDB_SYNC_INTERVAL   5
#define DEFAULT_LEVELDB_USE_COLUMNDB    false
#define DEFAULT_LEVELDB_ZERO_FACTOR     10.0
#define DEFAULT_LEVELDB_LEVEL_FACTOR    10.0
#define DEFAULT_LEVELDB_BLOCK_SIZE         (64 << 10)
#define DEFAULT_LEVELDB_SSTABLE_SIZE       (32 << 20)
#define DEFAULT_LEVELDB_WRITE_BUFFER_SIZE  (32 << 20)
#define DEFAULT_LEVELDB_CACHE_SIZE         (512 << 20)

#endif // BUBBLEFS_DB_LEVELDB_COMMON_H_