// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// slash/slash/include/slash_binlog.h

#ifndef BUBBLEFS_UTILS_SLASH_BINLOG_H_
#define BUBBLEFS_UTILS_SLASH_BINLOG_H_

#include <assert.h>
#include <string>
#include "utils/slash_status.h"

namespace bubblefs {
namespace myslash {

class BinlogReader;
  
// SyncPoint is a file number and an offset;

const std::string kBinlogPrefix = "binlog";
const std::string kManifest = "manifest";
const int kBinlogSize = 128;
//const int kBinlogSize = (100 << 20);
const int kBlockSize = (64 << 10);
// Header is Type(1 byte), length (3 bytes), time (4 bytes)
const size_t kHeaderSize = 1 + 3 + 4;

enum RecordType {
  kZeroType = 0,
  kFullType = 1,
  kFirstType = 2,
  kMiddleType = 3,
  kLastType = 4,
  kEof = 5,
  kBadRecord = 6,
  kOldRecord = 7
};

class Binlog {
 public:
  static Status Open(const std::string& path, Binlog** logptr);

  Binlog() { }
  virtual ~Binlog() { }

  // TODO (aa) 
  //   1. maybe add Options
  
  //
  // Basic API
  //
  virtual Status Append(const std::string &item) = 0;
  virtual BinlogReader* NewBinlogReader(uint32_t filenum, uint64_t offset) = 0;

  // Set/Get Producer filenum and offset with lock
  virtual Status GetProducerStatus(uint32_t* filenum, uint64_t* pro_offset) = 0;
  virtual Status SetProducerStatus(uint32_t filenum, uint64_t pro_offset) = 0;

 private:

  // No copying allowed
  Binlog(const Binlog&);
  void operator=(const Binlog&);
};

class BinlogReader {
 public:
  BinlogReader() { }
  virtual ~BinlogReader() { }

  virtual Status ReadRecord(std::string &record) = 0;
  //bool ReadRecord(Slice* record, std::string* scratch) = 0;

 private:

  // No copying allowed;
  BinlogReader(const BinlogReader&);
  void operator=(const BinlogReader&);
};

} // namespace myslash
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SLASH_BINLOG_H_