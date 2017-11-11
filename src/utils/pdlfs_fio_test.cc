/*
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/src/fio_test.cc

#include "utils/pdlfs_fio.h"
#include "platform/pdlfs_testharness.h"

namespace bubblefs {
namespace pdlfs {

class FioTest {};

TEST(FioTest, EncodeDecode) {
  Fentry entry1;
  Fentry entry2;
  entry1.pid = DirId(1, 2, 3);
  entry1.nhash = "xyz";
  entry1.zserver = 4;
  Stat* stat1 = &entry1.stat;
  stat1->SetInodeNo(5);
  stat1->SetSnapId(6);
  stat1->SetRegId(7);
  stat1->SetFileSize(8);
  stat1->SetFileMode(9);
  stat1->SetUserId(10);
  stat1->SetGroupId(11);
  stat1->SetZerothServer(12);
  stat1->SetChangeTime(13);
  stat1->SetModifyTime(14);
  char tmp1[DELTAFS_FENTRY_BUFSIZE];
  Slice encoding1 = entry1.EncodeTo(tmp1);
  Slice input = encoding1;
  bool ok = entry2.DecodeFrom(&input);
  ASSERT_TRUE(ok);
  char tmp2[DELTAFS_FENTRY_BUFSIZE];
  Slice encoding2 = entry2.EncodeTo(tmp2);
  ASSERT_EQ(encoding1.as_string(), encoding2.as_string());
}

}  // namespace pdlfs
}  // namespace bubblefs

int main(int argc, char** argv) {
  return ::bubblefs::pdlfs::test::RunAllTests(&argc, &argv);
}