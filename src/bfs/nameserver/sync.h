
#ifndef BUBBLEFS_NAMESERVER_SYNC_H_
#define BUBBLEFS_NAMESERVER_SYNC_H_

#include <functional>
#include <string>

#include "proto/status_code.pb.h"

namespace bubblefs {
namespace bfs {  
  
class RpcClient;

typedef std::function<void (const std::string& log)> LogCallback;
typedef std::function<void (int32_t, std::string*)> SnapshotCallback;
typedef std::function<void ()> EraseCallback;

struct SyncCallbacks {
  LogCallback log_callback;
  SnapshotCallback snapshot_callback;
  EraseCallback erase_callback;
  
  SyncCallbacks(LogCallback log_callback2, 
                SnapshotCallback snapshot_callback2,
                EraseCallback erase_callback2) : 
                log_callback(log_callback2),
                snapshot_callback(snapshot_callback2),
                erase_callback(erase_callback2) {}
};

class Sync {
 public:
   Sync() {}
   virtual ~Sync() {};
   // Description: Register 'callback' to Sync and redo log.
   // NOTICE: Sync does not work until Init is called.
   virtual void Init(SyncCallbacks callbacks) = 0;
   // Description: Return true if this server is Leader.
   // TODO: return 'leader_addr' which points to the current leader.
   virtual bool IsLeader(std::string* leader_addr = nullptr) = 0;
   // Description: Synchronous interface. Leader will replicate 'entry' to followers.
   // Return true upon success.
   // Follower will ignore this call and return true
   virtual bool Log(const std::string& entry, int timeout_ms = 10000) = 0;
   // Description: Asynchronous interface. Leader will replicate 'entry' to followers,
   // then call 'callback' with result(true if success, false is failed) to notify the user.
   // Follower will ignore this call and return true.
   virtual void Log(const std::string& entry, std::function<void (bool)> callback) = 0;
   // Turn a follower to leader.
   // Leader will ignore this call.
   virtual void SwitchToLeader() = 0;
   // Return ha status.
   virtual std::string GetStatus() = 0;
};
  
} // namespace bfs  
} // namespace bubblefs

#endif // BUBBLEFS_NAMESERVER_SYNC_H_