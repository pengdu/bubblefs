// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// slash/slash/tests/slash_env_test.cc

#include "platform/slash_env.h"
#include "platform/slash_testharness.h"

namespace bubblefs {
namespace myslash {

class EnvTest { };

TEST(EnvTest, SetMaxFileDescriptorNum) {
  ASSERT_EQ(0, SetMaxFileDescriptorNum(10));
  ASSERT_NE(0, SetMaxFileDescriptorNum(2147483647));
}

TEST(EnvTest, FileOps) {
  std::string tmp_dir;
  GetTestDirectory(&tmp_dir);

  ASSERT_TRUE(DeleteDirIfExist(tmp_dir.c_str()));
  ASSERT_TRUE(!FileExists(tmp_dir.c_str()));
  ASSERT_NE(0, SetMaxFileDescriptorNum(2147483647));
}

}  // namespace myslash
}  // namespace bubblefs