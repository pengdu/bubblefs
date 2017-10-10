
#include "namespace.h"
#include <fcntl.h>
#include "db/db.h"
#include "gflags/gflags.h"
#include "nameserver/sync.h"
#include "platform/bdcommon_logging.h"
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

  
  
} // namespace bfs  
} // namespace bubblefs

