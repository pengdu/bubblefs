
#ifndef BUBBLEFS_RPC_RPC_CLIENT_H_
#define BUBBLEFS_RPC_RPC_CLIENT_H_

#include "google/protobuf/service.h"
#include "platform/macros.h"

namespace bubblefs {
namespace rpc {

struct RpcOptions {
  int32_t rpc_timeout;
  int retry_times;
};
  
class RpcClient {
 public:
   SofaPbrpcClient() {}
    
   virtual ~RpcClient() {}
    
   template <class T>
   virtual bool GetStub(const std::string server, T** stub) = 0;
    
   template <class Stub, class Request, class Response, class Callback>
   virtual bool SendRequest(Stub* stub, void(Stub::*func)(
                    google::protobuf::RpcController*,
                    const Request*, Response*, Callback*),
                    const Request* request, Response* response,
                    const RpcOptions& options) = 0;
    
   template <class Stub, class Request, class Response, class Callback>
   void AsyncRequest(Stub* stub, void(Stub::*func)(
                    google::protobuf::RpcController*,
                    const Request*, Response*, Callback*),
                    const Request* request, Response* response,
                    std::function<void (const Request*, Response*, bool, int)> callback,
                    const RpcOptions& options) = 0;
    
   template <class Request, class Response, class Callback>
   static void RpcCallback(google::protobuf::RpcController* rpc_controller,
                            const Request* request,
                            Response* response,
                            std::function<void (const Request*, Response*, bool, int)> callback) = 0;
                            
 protected:
  DISALLOW_COPY_AND_ASSIGN(RpcClient);
};

} // namespace rpc
} // namespace bubblefs

#endif // BUBBLEFS_RPC_RPC_CLIENT_H_