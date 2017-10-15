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

#ifndef BUBBLEFS_DB_DB_H_
#define BUBBLEFS_DB_DB_H_

#include "platform/base_error.h"
#include "platform/macros.h"
#include "platform/types.h"
#include "utils/caffe2_registry.h"
#include "utils/status.h"

namespace bubblefs {
namespace db {
  
enum class DBType { kMiniDB, kLevelDB };  
  
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
  
  virtual Status GetStatus() = 0;
  virtual Status StartSeek() = 0;
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
  virtual bool Put(const string& key, const string& value) = 0;
  /**
   * Commits the current writes.
   */
  virtual Status Commit() = 0;

  DISALLOW_COPY_AND_ASSIGN(Transaction);
};

/**
 * An abstract class for accessing a database of key-value pairs.
 */
class DB {
 public:
  DB() { }
  virtual ~DB() { }
  
  virtual Status Open(const string& source, Mode mode) = 0;
  
  virtual Status Open(const string& source, Mode mode, int64_t db_cache_size) = 0;
  /**
   * Closes the database.
   */
  virtual Status Close() = 0;
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

DB* NewDB(DBType backend);

void DeleteDB(DB** db);

// Database classes are registered by their names so we can do optional
// dependencies.
/*
CAFFE_DECLARE_REGISTRY(Caffe2DBRegistry, DB);
#define REGISTER_CAFFE2_DB(name, ...) \
  CAFFE_REGISTER_CLASS(Caffe2DBRegistry, name, __VA_ARGS__)
*/

/**
 * Returns a database object of the given database type, source and mode. The
 * caller takes the ownership of the pointer. If the database type is not
 * supported, a nullptr is returned. The caller is responsible for examining the
 * validity of the pointer.
 */
/*
inline unique_ptr<DB> CreateDB(const string& db_type) {
  auto result = Caffe2DBRegistry()->Create(db_type);
  if (!result) {
    PRINTF_ERROR("Not found DB %s\n", db_type.c_str());
  }
  return result;
}
*/

  
} // namespace db  
} // namespace bubblefs

#endif // BUBBLEFS_API_DB_H_