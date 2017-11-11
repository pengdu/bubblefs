// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// chromium/base/rand_util.h

#ifndef BUBBLEFS_UTILS_BRPC_RANDUINT64_H_
#define BUBBLEFS_UTILS_BRPC_RANDUINT64_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <random>
#include <string>
#include <vector>
#include "platform/macros.h"
#include "platform/types.h"

namespace bubblefs {
namespace mybrpc {

// --------------------------------------------------------------------------
// NOTE(gejun): Functions in this header read from /dev/urandom in posix
// systems and are not proper for performance critical situations.
// For fast random numbers, check fast_rand.h
// --------------------------------------------------------------------------

// Returns a random number in range [0, kuint64max]. Thread-safe.
BASE_EXPORT uint64_t RandUint64();

} // namespace mybrpc
} // namespace bubblefs

#endif //BUBBLEFS_UTILS_BRPC_RANDUINT64_H_