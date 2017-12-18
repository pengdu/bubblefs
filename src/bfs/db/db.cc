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

// caffe2/caffe2/core/db.cc

#include "db/db.h"
#include <stdio.h>
#include <stdlib.h>
#include <mutex>
#include <string>
#include "db/leveldb.h"
#include "platform/base_error.h"
#include "platform/types.h"
#include "utils/unique_ptr.h"

namespace bubblefs {
namespace db {
    
// Below, we provide a bare minimum database "minidb" as a reference
// implementation as well as a portable choice to store data.
// Note that the MiniDB classes are not exposed via a header file - they should
// be created directly via the db interface. See MiniDB for details.

class MiniDBCursor : public Cursor {
 public:
  explicit MiniDBCursor(FILE* f, std::mutex* mutex)
    : file_(f), lock_(*mutex), valid_(true) {
  }
  ~MiniDBCursor() {}

  Status GetStatus() override {
    if (valid_)
      return Status::OK();
    return Status(error::USER_ERROR, "Cursor is not valid");
  }
  
  void Seek(const string& /*key*/) override {
    PANIC("MiniDB does not support seeking to a specific key.\n");
  }

  void SeekToFirst() override {
    fseek(file_, 0, SEEK_SET);
    PANIC_ENFORCE(!feof(file_), "Hmm, empty file?\n");
    // Read the first item.
    valid_ = true;
    Next();
  }

  void Next() override {
    // First, read in the key and value length.
    if (fread(&key_len_, sizeof(int), 1, file_) == 0) {
      // Reaching EOF.
      PRINTF_INFO("EOF reached, setting valid to false\n");
      valid_ = false;
      return;
    }
    PANIC_ENFORCE_EQ(fread(&value_len_, sizeof(int), 1, file_), 1);
    PANIC_ENFORCE_GT(key_len_, 0);
    PANIC_ENFORCE_GT(value_len_, 0);
    // Resize if the key and value len is larger than the current one.
    if (key_len_ > key_.size()) {
      key_.resize(key_len_);
    }
    if (value_len_ > value_.size()) {
      value_.resize(value_len_);
    }
    // Actually read in the contents.
    PANIC_ENFORCE_EQ(
        fread(key_.data(), sizeof(char), key_len_, file_), key_len_);
    PANIC_ENFORCE_EQ(
        fread(value_.data(), sizeof(char), value_len_, file_), value_len_);
    // Note(Yangqing): as we read the file, the cursor naturally moves to the
    // beginning of the next entry.
  }

  string key() override {
    PANIC_ENFORCE(valid_, "Cursor is at invalid location!\n");
    return string(key_.data(), key_len_);
  }

  string value() override {
    PANIC_ENFORCE(valid_, "Cursor is at invalid location!\n");
    return string(value_.data(), value_len_);
  }

  bool Valid() override { return valid_; }

 private:
  FILE* file_;
  std::lock_guard<std::mutex> lock_;
  bool valid_;
  unsigned key_len_;
  std::vector<char> key_;
  unsigned value_len_;
  std::vector<char> value_;
}; 

class MiniDBTransaction : public Transaction {
 public:
  explicit MiniDBTransaction(FILE* f, std::mutex* mutex)
    : file_(f), lock_(*mutex) {}
  ~MiniDBTransaction() {
    Commit();
  }

  void Put(const string& key, const string& value) override {
    unsigned key_len = key.size();
    unsigned value_len = value.size();
    PANIC_ENFORCE_EQ(fwrite(&key_len, sizeof(int), 1, file_), 1);
    PANIC_ENFORCE_EQ(fwrite(&value_len, sizeof(int), 1, file_), 1);
    PANIC_ENFORCE_EQ(
        fwrite(key.c_str(), sizeof(char), key_len, file_), key_len);
    PANIC_ENFORCE_EQ(
        fwrite(value.c_str(), sizeof(char), value_len, file_), value_len);
  }
  
  void Delete(const string& key) override { }

  Status Commit() override {
    if (file_ != nullptr) {
      PANIC_ENFORCE_EQ(fflush(file_), 0);
      file_ = nullptr;
      return Status::OK();
    }
    return Status(error::USER_ERROR, "MiniDB Commit fail as file_ is nullptr");
  }

 private:
  FILE* file_;
  std::lock_guard<std::mutex> lock_;

  DISALLOW_COPY_AND_ASSIGN(MiniDBTransaction);
};

class MiniDB : public DB {
 public:
  MiniDB() : file_(nullptr) { }
  ~MiniDB() { Close(); }
  
  Status Open(const string& source, Mode mode) override {
    source_ = source;
    mode_ = mode;
    switch (mode) {
      case NEW:
        file_ = fopen(source.c_str(), "wb");
        break;
      case WRITE:
        file_ = fopen(source.c_str(), "ab");
        fseek(file_, 0, SEEK_END);
        break;
      case READ:
        file_ = fopen(source.c_str(), "rb");
        break;
      default:
        PANIC("Cannot open file: %s, as mode: %d is invalid\n", 
              source.c_str(), static_cast<int>(mode));
    }
    if (!file_) {
      PANIC("Cannot open file: %s\n", source.c_str());
      return Status(error::USER_ERROR, "Cannot open MiniDB file");
    }
    PRINTF_INFO("Opened MiniDB %s\n", source.c_str());
    return Status::OK();
  }
  
  Status Open(const string& source, Mode mode, int64_t db_cache_size) override {
    return Status::NotSupported();
  }

  Status Close() override {
    if (file_) {
      fclose(file_);
    }
    file_ = nullptr;
    DB::Close();
    return Status::OK();
  }

  unique_ptr<Cursor> NewCursor() override {
    PANIC_ENFORCE_EQ(this->mode_, Mode::READ);
    return make_unique<MiniDBCursor>(file_, &file_access_mutex_);
  }

  unique_ptr<Transaction> NewTransaction() override {
    PANIC_ENFORCE(this->mode_ == Mode::NEW || this->mode_ ==Mode::WRITE, "mode is not NEW or WRITE");
    return make_unique<MiniDBTransaction>(file_, &file_access_mutex_);
  }
  
  bool Valid() override {
    return (nullptr != file_);
  }
  
  Status DestroyDB() override {
    return Status::NotSupported();
  }
  
  Status Get(const string& key, string* value) override {
    return Status::NotSupported();
  }
  
  Status Put(const string& key, const string& value) override {
    return Status::NotSupported();
  }
  
  Status Delete(const string& key) override {
    return Status::NotSupported();
  }

 private:
  FILE* file_;
  // access mutex makes sure we don't have multiple cursors/transactions
  // reading the same file.
  std::mutex file_access_mutex_;
};

// std::unique_ptr<DB> db(CreateDB(db_type, name, NEW));
DB* NewDB(DBType db_type) {
  if (db_type == DBType::kLevelDB) { // USE_LEVELDB
    return new LevelDB();
  }
  PANIC("Unknown database %d\n", db_type);
  return nullptr;
}

void DeleteDB(DB** db) {
  delete (*db);
  *db = nullptr;
}

// For lazy-minded, one can also call with lower-case name.
//REGISTER_CAFFE2_DB(minidb, MiniDB);
  
} // namespace db  
} // namespace bubblefs