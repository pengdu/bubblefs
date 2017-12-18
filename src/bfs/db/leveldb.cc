
#include <sstream>
#include "db/leveldb.h"

namespace bubblefs {
namespace db {

Status LevelDBCursor::GetStatus() {
  if (iter_->status().ok())
    return Status::OK();
  return Status(error::USER_ERROR, iter_->status().ToString());
}

void LevelDBTransaction::Put(const string& key, const string& value) {
  batch_->Put(key, value);
}

void LevelDBTransaction::Delete(const string& key) {
  batch_->Delete(key);
}

Status LevelDBTransaction::Commit() {
  leveldb::Status status = db_->Write(leveldb::WriteOptions(), batch_.get());
  batch_.reset(new leveldb::WriteBatch());
  if (status.ok())
    return Status::OK();
  return Status(error::USER_ERROR, status.ToString());  
}

Status LevelDB::Open(const string& source, Mode mode) {
  source_ = source;
  mode_ = mode;
  leveldb::Options options;
  options.block_size = DEFAULT_LEVELDB_BLOCK_SIZE;
  options.write_buffer_size = DEFAULT_LEVELDB_WRITE_BUFFER_SIZE;
  options.max_open_files = DEFAULT_LEVELDB_MAX_OPEN_FILES;
  options.error_if_exists = mode == NEW;
  options.create_if_missing = mode != READ;
  leveldb::DB* db_temp;
  leveldb::Status s = leveldb::DB::Open(options, source, &db_temp);
  if (!s.ok()) {
    std::stringstream ss;
    ss << "Failed to open leveldb " << source << ". " << s.ToString();
    return Status(error::USER_ERROR, ss.str());
  }
  db_.reset(db_temp);
  return Status::OK(); 
}

Status LevelDB::Open(const string& source, Mode mode, int64_t db_cache_size) {
  source_ = source;
  mode_ = mode;
  leveldb::Cache* db_cache_temp = leveldb::NewLRUCache(db_cache_size * MBYTES);
  if (!db_cache_temp) {
    return Status(error::USER_ERROR, "Failed to new leveldb lru cache");
  }
  db_cache_.reset(db_cache_temp);
  leveldb::Options options;
  options.error_if_exists = mode == NEW;
  options.create_if_missing = mode != READ;
  options.block_cache = db_cache_.get();
  leveldb::DB* db_temp;
  leveldb::Status s = leveldb::DB::Open(options, source, &db_temp);
  if (!s.ok()) {
    std::stringstream ss;
    ss << "Failed to open leveldb " << source << ". " << s.ToString();
    return Status(error::USER_ERROR, ss.str());
  }
  db_.reset(db_temp);
  return Status::OK();
}

Status LevelDB::Close() { 
  db_.reset();
  db_cache_.reset();
  DB::Close();
  return Status::OK();
}

bool LevelDB::Valid() {
  return (nullptr != db_);
}

Status LevelDB::DestroyDB() {
  Close();
  leveldb::Options options;
  leveldb::Status s = leveldb::DestroyDB(source_, options);
  if (s.ok())
    return Status::OK();
  std::stringstream ss;
  ss << "Leveldb failed to DestroyDB source: " << source_ << ". " << s.ToString();
  return Status(error::USER_ERROR, ss.str());
}

Status LevelDB::Get(const string& key, string* value) {
  leveldb::Status s = db_->Get(leveldb::ReadOptions(), key, value);
  if (s.ok())
    return Status::OK();
  std::stringstream ss;
  ss << "Leveldb failed to Get key: " << key << ". " << s.ToString();
  return Status(error::USER_ERROR, ss.str());
}

Status LevelDB::Put(const string& key, const string& value) {
  leveldb::Status s = db_->Put(leveldb::WriteOptions(), key, value);
  if (s.ok())
    return Status::OK();
  std::stringstream ss;
  ss << "Leveldb failed to Put key: " << key << ". " << s.ToString();
  return Status(error::USER_ERROR, ss.str());
}

Status LevelDB::Delete(const string& key) {
  leveldb::Status s = db_->Delete(leveldb::WriteOptions(), key);
  if (s.ok())
    return Status::OK();
  std::stringstream ss;
  ss << "Leveldb failed to Delete key: " << key << ". " << s.ToString();
  return Status(error::USER_ERROR, ss.str());
}

} // namespace db
} // namespace bubblefs