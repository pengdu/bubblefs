
#ifndef BUBBLEFS_CLIENT_NAMESERVER_CLIENT_H_
#define BUBBLEFS_CLIENT_NAMESERVER_CLIENT_H_

#include <string>
#include <vector>
#include "platform/bdcommon_logging.h"
#include "platform/mutexlock.h"
#include "proto/nameserver.pb.h"
#include "proto/status_code.pb.h"
#include "rpc/rpc_client.h"

namespace bubblefs {
namespace bfs {

class NameServerClient {
 public:
   NameServerClient(rpc::RpcClient* rpc_client, const std::string& nameserver_nodes);
   
   template <class Request, class Response, class Callback>
   bool SendRequest(void(nameserver::NameServer_Stub::*func)(
                    google::protobuf::RpcController*,
                    const Request*, Response*, Callback*),
                    const Request* request, Response* response,
                    rpc::RpcOptions& rpc_options) {
     int32_t rpc_timeout = rpc_options.rpc_timeout;
     int retry_times = rpc_options.retry_times;
     
     bool ret = false;
     for (uint32_t i = 0; i < stubs_.size(); ++i) {
       int ns_id = leader_id_;
       ret = rpc::SendRequest(stubs_[ns_id], func, request, response,
                              rpc_timeout, retry_times);
       if (ret && response->status() != kIsFollower) {
         Log(DEBUG, "Send rpc to %d %s return %s", 
             leader_id_, nameserver_nodes_[leader_id_].c_str(),
             StatusCode_Name(response->status()).c_str());
         return true;
       }
       MutexLock lock(&mu_);
       if (ns_id == leader_id_) {
         leader_id_ = (leader_id_ + 1) % stubs_.size();
       }
       Log(INFO, "Try next nameserver %d %s",
           leader_id_, nameserver_nodes_[leader_id_].c_str());
     }
   }
   
 private:
   rpc::RpcClient* rpc_client_;
   std::vector<std::string> nameserver_node_;
   std::vector<nameserver::NameServer_Stub*> stubs_;
   port::Mutex mu_;
   int leader_id_;
};  
  
} // namespace bfs  
} // namespace bubblefs

#endif // BUBBLEFS_CLIENT_NAMESERVER_CLIENT_H_