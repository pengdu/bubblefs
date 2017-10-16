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
  
// std::unique_ptr<DB> db(CreateDB(db_type, name, READ));
// std::unique_ptr<Cursor> cursor(db->NewCursor());
// TestCursor(cursor.get());
class LevelDBCursor : public Cursor {
 public:
  explicit LevelDBCursor(leveldb::DB* db)
      : iter_(db->NewIterator(leveldb::ReadOptions())) { }
  ~LevelDBCursor() {}
  
  Status GetStatus() override;
  Status StartSeek() override;
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

// std::unique_ptr<DB> db(CreateDB(db_type, name, NEW));
//  if (!db.get()) {
//    LOG(ERROR) << "Cannot create db of type " << db_type;
//    return false;
//  }
//  std::unique_ptr<Transaction> trans(db->NewTransaction());
//  for (int i = 0; i < kMaxItems; ++i) {
//    std::stringstream ss;
//    ss << std::setw(2) << std::setfill('0') << i;
//    trans->Put(ss.str(), ss.str());
//  }
//  trans->Commit();
//  trans.reset();
//  db.reset();
class LevelDBTransaction : public Transaction {
 public:
  explicit LevelDBTransaction(leveldb::DB* db) : db_(db) {
    PANIC_ENFORCE(db_, "db_ is NULL");
    batch_.reset(new leveldb::WriteBatch());
  }
  ~LevelDBTransaction() { }
  bool Put(const string& key, const string& value) override {
    batch_->Put(key, value);
    return true;
  }
  Status Commit() override;

 private:
  leveldb::DB* db_;
  std::unique_ptr<leveldb::WriteBatch> batch_;

  DISALLOW_COPY_AND_ASSIGN(LevelDBTransaction);
};

class LevelDB : public DB {
 public:
  LevelDB() : db_(nullptr), db_cache_(nullptr) { }
  
  Status Open(const string& source, Mode mode) override;
  
  Status Open(const string& source, Mode mode, int64_t db_cache_size) override;

  Status Close() override;
  
  std::unique_ptr<Cursor> NewCursor() override {
    return make_unique<LevelDBCursor>(db_.get());
  }
  std::unique_ptr<Transaction> NewTransaction() override {
    return make_unique<LevelDBTransaction>(db_.get());
  }
  
  bool Valid() override;
  
  Status DestroyDB() override;
  
  Status Get(const string& key, string* value) override;
  
  Status Put(const string& key, const string& value) override;
  
  Status Delete(const string& key) override;

 private:
  std::unique_ptr<leveldb::DB> db_;
  std::unique_ptr<leveldb::Cache> db_cache_;
};
  
} // namespace db  
} // namespace bubblefs

#endif // BUBBLEFS_DB_LEVELDB_H_