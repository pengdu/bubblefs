
#ifndef BUBBLEFS_RPC_BASE_RPC_CLIENT_H_
#define BUBBLEFS_RPC_BASE_RPC_CLIENT_H_

#include "google/protobuf/service.h"
#include "platform/macros.h"

namespace bubblefs {
namespace rpc {

struct RpcOptions {
  int32_t rpc_timeout;
  int retry_times;
  
  RpcOptions(int32_t rpc_timeout2, int retry_times2) 
             : rpc_timeout(rpc_timeout2), retry_times(retry_times2) {} 
};

class RpcClient {
 public:
   RpcClient() {}
    
   virtual ~RpcClient() {}
   
   virtual google::protobuf::RpcChannel* GetRpcChannel(const std::string server) = 0;
};

} // namespace rpc
} // namespace bubblefs

#endif // BUBBLEFS_RPC_BASE_RPC_CLIENT_H_