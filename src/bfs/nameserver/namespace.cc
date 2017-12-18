
#include "namespace.h"
#include <fcntl.h>
#include "db/leveldb.h"
#include "gflags/gflags.h"
#include "nameserver/sync.h"
#include "platform/base_error.h"
#include "platform/bdcommon_logging.h"
#include "platform/types.h"
#include "platform/timer.h"
#include "utils/bdcommon_str_util.h"
#include "utils/raw_coding.h"

DECLARE_string(namedb_path);
DECLARE_int64(namedb_cache_size);
DECLARE_int32(default_replica_num);
DECLARE_int32(block_id_allocation_size);
DECLARE_int32(snapshot_step);
DECLARE_bool(check_orphan);

const int64_t kRootEntryid = 1;

namespace bubblefs {
namespace bfs {

NameSpace::NameSpace(bool standalone)
    : version_(0), last_entry_id_(1), 
    block_id_upbound_(1), next_block_id_(1) {
  db_cache_  = leveldb::NewLRUCache(FLAGS_namedb_cache_size * MBYTES);
  leveldb::Options db_opts;
  db_opts.create_if_missing = true;
  db_opts.block_cache = db_cache_;
  leveldb::Status s = leveldb::DB::Open(db_opts, FLAGS_namedb_path, &db_);
  if (!s.ok()) {
    db_ = nullptr;
    BDCOMMON_LOG(ERROR, "Open leveldb fail: %s", s.ToString().c_str());
    EXIT_FAIL("Open leveldb fail\n");
  }
  if (standalone) {
    Activate(nullptr, nullptr);
  }
}
    
void NameSpace::Activate(std::function<void (const FileInfo&)> callback, 
                         nameserver::NameServerLog* log) {
  std::string version_key(8, 0);
  version_key.append("version");
  
}  
  
} // namespace bfs  
} // namespace bubblefs

