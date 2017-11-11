/*
 * Copyright (c) 2011 The LevelDB Authors.
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/include/pdlfs-common/testutil.h

#ifndef BUBBLEFS_PLATFORM_TESTUTIL_H_
#define BUBBLEFS_PLATFORM_TESTUTIL_H_

#include "platform/pdlfs_env.h"
#include "utils/pdlfs_random.h"

namespace bubblefs {
namespace mypdlfs {
namespace test {
// Return a file name containing a given unique identifier.
extern std::string FileName(int i);

// Return true iff a file ends with a given suffix
extern bool StringEndWith(const std::string& str, const std::string& suffix);

// Store in *dst a random string of length "len" and return a Slice that
// references the generated data.
extern Slice RandomString(Random* rnd, int len, std::string* dst);

// Return a random key with the specified length that may contain interesting
// characters (e.g. \x00, \xff, etc.).
extern std::string RandomKey(Random* rnd, int len);

// Store in *dst a string of length "len" that will compress to
// "N*compressed_fraction" bytes and return a Slice that references
// the generated data.
extern Slice CompressibleString(Random* rnd, double compressed_fraction,
                                size_t len, std::string* dst);

}  // namespace test
}  // namespace mypdlfs
}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TESTUTIL_H_