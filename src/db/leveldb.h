/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/db/leveldb.cc

#ifndef BUBBLEFS_DB_LEVELDB_H_
#define BUBBLEFS_DB_LEVELDB_H_

#include <string>
#include "db/db.h"
#include "db/leveldb_common.h"
#include "gflags/gflags.h"
#include "platform/base_error.h"
#include "utils/unique_ptr.h"

namespace bubblefs {
namespace db {
  
class LevelDBCursor : public Cursor {
 public:
  explicit LevelDBCursor(leveldb::DB* db)
      : iter_(db->NewIterator(leveldb::ReadOptions())) {
    SeekToFirst();
    FPRINTF_CHECK(iter_->status().ok(), iter_->status().ToString());
  }
  ~LevelDBCursor() {}
  void Seek(const string& key) override { iter_->Seek(key); }
  bool SupportsSeek() override { return true; }
  void SeekToFirst() override { iter_->SeekToFirst(); }
  void Next() override { iter_->Next(); }
  string key() override { return iter_->key().ToString(); }
  string value() override { return iter_->value().ToString(); }
  bool Valid() override { return iter_->Valid(); }

 private:
  std::unique_ptr<leveldb::Iterator> iter_; // leveldb::Iterator* iter_;
};

class LevelDBTransaction : public Transaction {
 public:
  explicit LevelDBTransaction(leveldb::DB* db) : db_(db) {
    FPRINTF_CHECK(db_, "db_ is NULL");
    batch_.reset(new leveldb::WriteBatch());
  }
  ~LevelDBTransaction() { Commit(); }
  bool Put(const string& key, const string& value) override {
    batch_->Put(key, value);
    return true;
  }
  bool Commit() override {
    leveldb::Status status = db_->Write(leveldb::WriteOptions(), batch_.get());
    batch_.reset(new leveldb::WriteBatch());
    if (!status.ok()) {
      FPRINTF_ERROR("Failed to write batch to leveldb. %s\n", 
                    status.ToString().c_str());
      return false;
    }
    return true;
  }

 private:
  leveldb::DB* db_;
  std::unique_ptr<leveldb::WriteBatch> batch_;

  DISALLOW_COPY_AND_ASSIGN(LevelDBTransaction);
};

class LevelDB : public DB {
 public:
  LevelDB() { }
  
  bool Open(const string& source, Mode mode) override {
    mode_ = mode;
    leveldb::Options options;
    options.block_size = 65536;
    options.write_buffer_size = 268435456;
    options.max_open_files = 100;
    options.error_if_exists = mode == NEW;
    options.create_if_missing = mode != READ;
    leveldb::DB* db_temp;
    leveldb::Status status = leveldb::DB::Open(options, source, &db_temp);
    if (!status.ok()) {
      FPRINTF_ERROR("Failed to open leveldb %s. %s\n", 
                    source.c_str(), status.ToString().c_str());
      return false;
    }
    db_.reset(db_temp);
    FPRINTF_INFO("Opened leveldb %s\n", source.c_str());
    return true;
  }

  bool Close() override { db_.reset(); return true; }
  std::unique_ptr<Cursor> NewCursor() override {
    return make_unique<LevelDBCursor>(db_.get());
  }
  std::unique_ptr<Transaction> NewTransaction() override {
    return make_unique<LevelDBTransaction>(db_.get());
  }

 private:
  std::unique_ptr<leveldb::DB> db_;
};
  
} // namespace db  
} // namespace bubblefs

#endif // BUBBLEFS_DB_LEVELDB_H_
