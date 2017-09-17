//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// Abstract interface for allocating memory in blocks. This memory is freed
// when the allocator object is destroyed. See the Arena class for more info.

// rocksdb/util/allocator.h

#ifndef BUBBLEFS_UTILS_ALLOCATOR_H_
#define BUBBLEFS_UTILS_ALLOCATOR_H_

#include <stddef.h>

namespace bubblefs {
namespace core {

class Allocator {
 public:
  virtual ~Allocator() {}

  virtual char* Alloc(const size_t size) = 0;
  virtual char* AllocAligned(const size_t size, const size_t alignment) = 0;

  virtual size_t BlockSize() const = 0;
};   
  
} // namespace core  
} // namespace bubblefs  

#endif // BUBBLEFS_UTILS_ALLOCATOR_H_