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

// caffe2/caffe2/core/db.h

#ifndef BUBBLEFS_UTILS_CAFFE2_DB_H_
#define BUBBLEFS_UTILS_CAFFE2_DB_H_

#include <mutex>
#include "platform/base_error.h"
#include "platform/types.h"
#include "utils/caffe2_blob_serializer_base.h"
#include "utils/caffe2_proto_caffe2.h"

namespace bubblefs {
namespace mycaffe2 {
namespace db {

/**
 * The mode of the database, whether we are doing a read, write, or creating
 * a new database.
 */
enum Mode { READ, WRITE, NEW };

/**
 * An abstract class for the cursor of the database while reading.
 */
class Cursor {
 public:
  Cursor() { }
  virtual ~Cursor() { }
  /**
   * Seek to a specific key (or if the key does not exist, seek to the
   * immediate next). This is optional for dbs, and in default, SupportsSeek()
   * returns false meaning that the db cursor does not support it.
   */
  virtual void Seek(const string& key) = 0;
  virtual bool SupportsSeek() { return false; }
  /**
   * Seek to the first key in the database.
   */
  virtual void SeekToFirst() = 0;
  /**
   * Go to the next location in the database.
   */
  virtual void Next() = 0;
  /**
   * Returns the current key.
   */
  virtual string key() = 0;
  /**
   * Returns the current value.
   */
  virtual string value() = 0;
  /**
   * Returns whether the current location is valid - for example, if we have
   * reached the end of the database, return false.
   */
  virtual bool Valid() = 0;

  DISALLOW_COPY_AND_ASSIGN(Cursor);
};

/**
 * An abstract class for the current database transaction while writing.
 */
class Transaction {
 public:
  Transaction() { }
  virtual ~Transaction() { }
  /**
   * Puts the key value pair to the database.
   */
  virtual void Put(const string& key, const string& value) = 0;
  /**
   * Commits the current writes.
   */
  virtual void Commit() = 0;

  DISALLOW_COPY_AND_ASSIGN(Transaction);
};

/**
 * An abstract class for accessing a database of key-value pairs.
 */
class DB {
 public:
  DB(const string& /*source*/, Mode mode) : mode_(mode) {}
  virtual ~DB() { }
  /**
   * Closes the database.
   */
  virtual void Close() = 0;
  /**
   * Returns a cursor to read the database. The caller takes the ownership of
   * the pointer.
   */
  virtual std::unique_ptr<Cursor> NewCursor() = 0;
  /**
   * Returns a transaction to write data to the database. The caller takes the
   * ownership of the pointer.
   */
  virtual std::unique_ptr<Transaction> NewTransaction() = 0;

 protected:
  Mode mode_;

  DISALLOW_COPY_AND_ASSIGN(DB);
};

// Database classes are registered by their names so we can do optional
// dependencies.
//CAFFE_DECLARE_REGISTRY(Caffe2DBRegistry, DB, const string&, Mode);
//#define REGISTER_CAFFE2_DB(name, ...) \
//  CAFFE_REGISTER_CLASS(Caffe2DBRegistry, name, __VA_ARGS__)

/**
 * Returns a database object of the given database type, source and mode. The
 * caller takes the ownership of the pointer. If the database type is not
 * supported, a nullptr is returned. The caller is responsible for examining the
 * validity of the pointer.
 */
inline unique_ptr<DB> CreateDB(
  const string& db_type, const string& source, Mode mode) {
  //auto result = Caffe2DBRegistry()->Create(db_type, source, mode);
  //VLOG(1) << ((!result) ? "not found db " : "found db ") << db_type;
  return nullptr;
}

/**
 * A reader wrapper for DB that also allows us to serialize it.
 */
class DBReader {
 public:

  friend class DBReaderSerializer;
  DBReader() {}

  DBReader(
      const string& db_type,
      const string& source,
      const int32_t num_shards = 1,
      const int32_t shard_id = 0) {
    Open(db_type, source, num_shards, shard_id);
  }

  explicit DBReader(const DBReaderProto& proto) {
    Open(proto.db_type(), proto.source());
    if (!proto.key.empty()) {
      PANIC_ENFORCE(cursor_->SupportsSeek(),
          "Encountering a proto that needs seeking but the db type "
          "does not support it.");
      cursor_->Seek(proto.key());
    }
    num_shards_ = 1;
    shard_id_ = 0;
  }

  explicit DBReader(std::unique_ptr<DB> db)
      : db_type_("<memory-type>"),
        source_("<memory-source>"),
        db_(std::move(db)) {
    PANIC_ENFORCE(db_.get(), "Passed null db");
    cursor_ = db_->NewCursor();
  }

  void Open(
      const string& db_type,
      const string& source,
      const int32_t num_shards = 1,
      const int32_t shard_id = 0) {
    // Note(jiayq): resetting is needed when we re-open e.g. leveldb where no
    // concurrent access is allowed.
    cursor_.reset();
    db_.reset();
    db_type_ = db_type;
    source_ = source;
    db_ = CreateDB(db_type_, source_, READ);
    PANIC_ENFORCE(db_, "Cannot open db: ", source_, " of type ", db_type_);
    InitializeCursor(num_shards, shard_id);
  }

  void Open(
      unique_ptr<DB>&& db,
      const int32_t num_shards = 1,
      const int32_t shard_id = 0) {
    cursor_.reset();
    db_.reset();
    db_ = std::move(db);
    PANIC_ENFORCE(db_.get(), "Passed null db");
    InitializeCursor(num_shards, shard_id);
  }

 public:
  /**
   * Read a set of key and value from the db and move to next. Thread safe.
   *
   * The string objects key and value must be created by the caller and
   * explicitly passed in to this function. This saves one additional object
   * copy.
   *
   * If the cursor reaches its end, the reader will go back to the head of
   * the db. This function can be used to enable multiple input ops to read
   * the same db.
   *
   * Note(jiayq): we loosen the definition of a const function here a little
   * bit: the state of the cursor is actually changed. However, this allows
   * us to pass in a DBReader to an Operator without the need of a duplicated
   * output blob.
   */
  void Read(string* key, string* value) const {
    PANIC_ENFORCE(cursor_ != nullptr, "Reader not initialized.");
    std::unique_lock<std::mutex> mutex_lock(reader_mutex_);
    *key = cursor_->key();
    *value = cursor_->value();

    // In sharded mode, each read skips num_shards_ records
    for (int s = 0; s < num_shards_; s++) {
      cursor_->Next();
      if (!cursor_->Valid()) {
        MoveToBeginning();
        break;
      }
    }
  }

  /**
   * @brief Seeks to the first key. Thread safe.
   */
  void SeekToFirst() const {
    PANIC_ENFORCE(cursor_ != nullptr, "Reader not initialized.");
    std::unique_lock<std::mutex> mutex_lock(reader_mutex_);
    MoveToBeginning();
  }

  /**
   * Returns the underlying cursor of the db reader.
   *
   * Note that if you directly use the cursor, the read will not be thread
   * safe, because there is no mechanism to stop multiple threads from
   * accessing the same cursor. You should consider using Read() explicitly.
   */
  inline Cursor* cursor() const {
    //LOG(ERROR) << "Usually for a DBReader you should use Read() to be "
    //              "thread safe. Consider refactoring your code.";
    return cursor_.get();
  }

 private:
  void InitializeCursor(const int32_t num_shards, const int32_t shard_id) {
    PANIC_ENFORCE(num_shards >= 1, "");
    PANIC_ENFORCE(shard_id >= 0, "");
    PANIC_ENFORCE(shard_id < num_shards, "");
    num_shards_ = num_shards;
    shard_id_ = shard_id;
    cursor_ = db_->NewCursor();
    SeekToFirst();
  }

  void MoveToBeginning() const {
    cursor_->SeekToFirst();
    for (auto s = 0; s < shard_id_; s++) {
      cursor_->Next();
      PANIC_ENFORCE(
          cursor_->Valid(), "Db has less rows than shard id: ", s, shard_id_);
    }
  }

  string db_type_;
  string source_;
  unique_ptr<DB> db_;
  unique_ptr<Cursor> cursor_;
  mutable std::mutex reader_mutex_;
  uint32_t num_shards_;
  uint32_t shard_id_;

  DISALLOW_COPY_AND_ASSIGN(DBReader);
};

class DBReaderSerializer : public BlobSerializerBase {
 public:
  /**
   * Serializes a DBReader. Note that this blob has to contain DBReader,
   * otherwise this function produces a fatal error.
   */
  void Serialize(
      const Blob& blob,
      const string& name,
      BlobSerializerBase::SerializationAcceptor acceptor) override;
};

class DBReaderDeserializer : public BlobDeserializerBase {
 public:
  void Deserialize(const BlobProto& proto, Blob* blob) override;
};

}  // namespace db
}  // namespace mycaffe2
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_CAFFE2_DB_H_