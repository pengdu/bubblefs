/*
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/include/pdlfs-common/log_scanner.h

#include "platform/pdlfs_env.h"
#include "utils/pdlfs_log_reader.h"
#include "utils/status.h"

namespace bubblefs {
namespace pdlfs {
  
namespace log {

class Scanner {
 public:
  explicit Scanner(SequentialFile* file, bool checksum = true,
                   uint64_t initial_offset = 0)
      : reporter_(NULL), file_(file), reader_(NULL), eof_(false) {
    reporter_ = new Reporter;
    reporter_->status = &status_;
    reader_ = new Reader(file, reporter_, checksum, initial_offset);
    Next();
  }

  ~Scanner() {
    delete reader_;
    delete reporter_;
    delete file_;
  }

  void Next() { eof_ = !reader_->ReadRecord(&record_, &scratch_); }

  bool Valid() { return status_.ok() && !eof_; }

  Slice record() const { return record_; }

  const Status& status() const { return status_; }

 private:
  // No copying allowed
  Scanner(const Scanner&);
  void operator=(const Scanner&);

  struct Reporter : public Reader::Reporter {
    Status* status;

    virtual void Corruption(size_t bytes, const Status& s) {
      if (this->status->ok()) *this->status = s;
    }
  };
  Status status_;
  Slice record_;  // Current record
  std::string scratch_;
  Reporter* reporter_;
  SequentialFile* file_;
  Reader* reader_;
  bool eof_;
};

}  // namespace log
}  // namespace pdlfs
}  // namespace bubblefs