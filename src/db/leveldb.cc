
#include <sstream>
#include "db/leveldb.h"

namespace bubblefs {
namespace db {

Status LevelDBCursor::GetStatus() {
  if (iter_->status().ok())
    return Status::OK();
  return Status(error::USER_ERROR, iter_->status().ToString());
}  
  
Status LevelDBCursor::StartSeek() {
  SeekToFirst();
  if (iter_->status().ok())
    return Status::OK();
  return Status(error::USER_ERROR, iter_->status().ToString());
}

Status LevelDBTransaction::Commit() {
  leveldb::Status status = db_->Write(leveldb::WriteOptions(), batch_.get());
  batch_.reset(new leveldb::WriteBatch());
  if (status.ok())
    return Status::OK();
  return Status(error::USER_ERROR, status.ToString());  
}

Status LevelDB::Open(const string& source, Mode mode) {
  mode_ = mode;
  leveldb::Options options;
  options.block_size = DEFAULT_LEVELDB_BLOCK_SIZE;
  options.write_buffer_size = DEFAULT_LEVELDB_WRITE_BUFFER_SIZE;
  options.max_open_files = DEFAULT_LEVELDB_MAX_OPEN_FILES;
  options.error_if_exists = mode == NEW;
  options.create_if_missing = mode != READ;
  leveldb::DB* db_temp;
  leveldb::Status status = leveldb::DB::Open(options, source, &db_temp);
  if (!status.ok()) {
    std::stringstream ss;
    ss << "Failed to open leveldb " << source << ". " << status.ToString();
    return Status(error::USER_ERROR, ss.str());
  }
  db_.reset(db_temp);
  FPRINTF_INFO("Opened leveldb %s\n", source.c_str());
  return Status::OK(); 
}

Status LevelDB::Open(const string& source, Mode mode, int64_t db_cache_size) {
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
  leveldb::Status status = leveldb::DB::Open(options, source, &db_temp);
  if (!status.ok()) {
    std::stringstream ss;
    ss << "Failed to open leveldb " << source << ". " << status.ToString();
    return Status(error::USER_ERROR, ss.str());
  }
  db_.reset(db_temp);
  FPRINTF_INFO("Opened leveldb %s\n", source.c_str());
  return Status::OK();
}

} // namespace db
} // namespace bubblefs