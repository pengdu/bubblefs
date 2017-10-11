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
    // We call Next() to read in the first entry.
    Next();
  }
  ~MiniDBCursor() {}

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
      FPRINTF_INFO("EOF reached, setting valid to false\n");
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
  vector<char> key_;
  unsigned value_len_;
  vector<char> value_;
}; 

class MiniDBTransaction : public Transaction {
 public:
  explicit MiniDBTransaction(FILE* f, std::mutex* mutex)
    : file_(f), lock_(*mutex) {}
  ~MiniDBTransaction() {
    Commit();
  }

  bool Put(const string& key, const string& value) override {
    unsigned key_len = key.size();
    unsigned value_len = value.size();
    PANIC_ENFORCE_EQ(fwrite(&key_len, sizeof(int), 1, file_), 1);
    PANIC_ENFORCE_EQ(fwrite(&value_len, sizeof(int), 1, file_), 1);
    PANIC_ENFORCE_EQ(
        fwrite(key.c_str(), sizeof(char), key_len, file_), key_len);
    PANIC_ENFORCE_EQ(
        fwrite(value.c_str(), sizeof(char), value_len, file_), value_len);
    return true;
  }

  bool Commit() override {
    if (file_ != nullptr) {
      PANIC_ENFORCE_EQ(fflush(file_), 0);
      file_ = nullptr;
      return true;
    }
    return false;
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
  
  bool Open(const string& source, Mode mode) override {
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
    }
    if (!file_) {
      PANIC("Cannot open file: %s\n", source.c_str());
      return false;
    }
    FPRINTF_INFO("Opened MiniDB %s\n", source.c_str());
    return true;
  }

  bool Close() override {
    if (file_) {
      fclose(file_);
    }
    file_ = nullptr;
    return true;
  }

  unique_ptr<Cursor> NewCursor() override {
    PANIC_ENFORCE_EQ(this->mode_, READ);
    return make_unique<MiniDBCursor>(file_, &file_access_mutex_);
  }

  unique_ptr<Transaction> NewTransaction() override {
    PANIC_ENFORCE(this->mode_ == NEW || this->mode_ == WRITE, "mode is not NEW or WRITE");
    return make_unique<MiniDBTransaction>(file_, &file_access_mutex_);
  }

 private:
  FILE* file_;
  // access mutex makes sure we don't have multiple cursors/transactions
  // reading the same file.
  std::mutex file_access_mutex_;
};

REGISTER_CAFFE2_DB(MiniDB, MiniDB);
// For lazy-minded, one can also call with lower-case name.
REGISTER_CAFFE2_DB(minidb, MiniDB);

DB* NewDB(const DBClass backend) {
  if (backend == DBClass::LEVELDB) { // USE_LEVELDB
    return new LevelDB();
  }
  PANIC("Unknown database backend");
  return nullptr;
}

void FreeDB(DB** db) {
  delete (*db);
  *db = nullptr;
}
  
} // namespace db  
} // namespace bubblefs