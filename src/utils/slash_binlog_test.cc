// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// slash/slash/tests/slash_binlog_test.cc

#include <iostream>
#include "platform/slash_env.h"
#include "platform/slash_testharness.h"
#include "utils/slash_binlog.h"

namespace bubblefs {
namespace slash {

class BinlogTest {
 public:
  BinlogTest()
    : log_(NULL),
      reader_(NULL) {
    GetTestDirectory(&tmpdir_);
    DeleteDirIfExist(tmpdir_.c_str());
    ASSERT_OK(Binlog::Open(tmpdir_, &log_));
  }
  ~BinlogTest() {
    delete reader_;
    delete log_;
    DeleteDirIfExist(tmpdir_.c_str());
  }
 protected:
  Binlog *log_;
  BinlogReader *reader_;
  const std::string test_item_ = "slash_item";
  std::string tmpdir_;
};

TEST(BinlogTest, ReadWrite) {
  std::string item1 = test_item_ + "1";
  std::string item2 = test_item_ + "2";
  std::string item3 = test_item_ + "3";
  // Write
  ASSERT_OK(log_->Append(item1));
  ASSERT_OK(log_->Append(item2));
  ASSERT_OK(log_->Append(item3));

  // Read
  std::string item;
  reader_ = log_->NewBinlogReader(0, 0);
  ASSERT_TRUE(reader_);
  ASSERT_OK(reader_->ReadRecord(item));
  ASSERT_EQ(item, item1);
  ASSERT_OK(reader_->ReadRecord(item));
  ASSERT_EQ(item, item2);
  ASSERT_OK(reader_->ReadRecord(item));
  ASSERT_EQ(item, item3);
}

TEST(BinlogTest, OffsetTest) {
  std::string second_block_item = "sbi";
  uint64_t offset = 0;
  uint32_t filenum = 0;
  uint64_t pro_offset = 0;
  int i = 0;

  while (true) {
    std::string curitem = test_item_ + std::to_string(++i);
    log_->Append(curitem);
    log_->GetProducerStatus(&filenum, &pro_offset);
    offset = filenum * kBinlogSize + pro_offset;
    if (offset > kBlockSize) {
      log_->Append(second_block_item);
      break;
    }
  }

  std::string str;
  reader_ = log_->NewBinlogReader(filenum, pro_offset);
  ASSERT_TRUE(reader_);
  reader_->ReadRecord(str);
  ASSERT_EQ(str, second_block_item);
}

TEST(BinlogTest, ProducerStatusOp) {
  std::cout << "ProducerStatusOp" << std::endl;
  uint32_t filenum = 187;
  uint64_t pro_offset = 8790;
  ASSERT_OK(log_->SetProducerStatus(filenum, pro_offset));
  filenum = 0;
  pro_offset = 0;
  ASSERT_OK(log_->GetProducerStatus(&filenum, &pro_offset));
  ASSERT_EQ(filenum, 187);
  ASSERT_EQ(pro_offset, 8790);
}

} // namespace slash
} // namespace bubblefs 