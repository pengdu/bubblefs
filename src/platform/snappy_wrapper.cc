//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// rocksdb/port/port_posix.cc

#include "platform/snappy_wrapper.h"
#include "platform/base.h"
#if TF_USE_SNAPPY
#include "snappy.h"
#endif

namespace bubblefs {
namespace port {
 
bool Snappy_Compress(const char* input, size_t length, string* output) {
#if TF_USE_SNAPPY
  output->resize(snappy::MaxCompressedLength(length));
  size_t outlen;
  snappy::RawCompress(input, length, &(*output)[0], &outlen);
  output->resize(outlen);
  return true;
#else
  return false;
#endif
}

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result) {
#if TF_USE_SNAPPY
  return snappy::GetUncompressedLength(input, length, result);
#else
  return false;
#endif
}

bool Snappy_Uncompress(const char* input, size_t length, char* output) {
#if TF_USE_SNAPPY
  return snappy::RawUncompress(input, length, output);
#else
  return false;
#endif
}
  
} // namespace port
} // namespace bubblefs