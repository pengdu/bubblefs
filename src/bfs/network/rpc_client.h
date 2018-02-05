
#ifndef BUBBLEFS_BFS_NETWORK_RPC_CLIENT_H_
#define BUBBLEFS_BFS_NETWORK_RPC_CLIENT_H_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include "platform/mutex.h"
#include "platform/bdcom_logging.h"
#include "utils/bdcom_threadpool.h"

#include "sofa/pbrpc/pbrpc.h"

namespace bubblefs {
namespace bfs {

class RpcClient {
 public:
  RpcClient() { }
  
  startup() {
    sofa::pbrpc::RpcClientOptions client_options;
    client_options.max_pending_buffer_size = 128; // 128MB
    rpc_client_.reset(new sofa::pbrpc::RpcClient(client_options));
  }
  
  // stub -> RpcChannel -> RpcClient(RpcClientStream)
  template <class T>
  bool getStub(const std::string server_address, T** stub) {
    MutexLock lock(&host_map_lock_);
    sofa::pbrpc::RpcChannel* channel = nullptr;
    auto it = host_map_.find(server_address);
    if (it != host_map_.end()) {
      channel = it->second;
    } else {
      sofa::pbrpc::RpcChannelOptions channel_options;
      channel = new sofa::pbrpc::RpcChannel(rpc_client_, server_address, channel_options);
      host_map_[server_address] = channel;
    }
    *stub = new T(channel); // stub needs channel to ctor
    return true;
  }
  
  template <class Stub, class Request, class Response, class Callback>
  bool sendRequest(Stub* stub, void(Stub::*func)(
                   google::protobuf::RpcController*,
                   const Request*, Response*, Callback*),
                   const Request* request, Response* response,
                   int32_t rpc_timeout, int retry_times) {
    sofa::pbrpc::RpcController controller;
    controller.SetTimeout(rpc_timeout * 1000L);
    for (int32_t retry = 0; retry < retry_times; ++retry) {
      // stub --RpcController call--> RpcChannel
      (stub->*func)(&controller, request, response, nullptr);
      if (controller.Failed()) {
        if (retry < retry_times - 1) {
          LOG(DEBUG, "RpcClient send failed, retry...");
          usleep(1000000); // 1s
        } else {
          LOG(WARNING, "RpcClient sendRequest to %s fail: %s\n",
              controller.RemoteAddress().c_str(), controller.ErrorText().c_str());
        }
      } else {
        return true;
      }
      controller.Reset();
    }
    return false;
  }
  
  template <class Stub, class Request, class Response, class Callback>
  void asyncSendRequest(Stub* stub, void(Stub::*func)(
                    google::protobuf::RpcController*,
                    const Request*, Response*, Callback*),
                    const Request* request, Response* response,
                    std::function<void (const Request*, Response*, bool, int)> callback,
                    int32_t rpc_timeout, int retry_times) {
    sofa::pbrpc::RpcController* controller = new sofa::pbrpc::RpcController(); // use async, new here
    controller->SetTimeout(rpc_timeout * 1000L);
    google::protobuf::Closure* done =
      sofa::pbrpc::NewClosure(&RpcClient::template rpcCallback<Request, Response, Callback>,
                              controller, request, response, callback);
    (stub->*func)(controller, request, response, done);
  }
  
  template <class Request, class Response, class Callback>
  static void rpcCallback(sofa::pbrpc::RpcController* rpc_controller,
                          const Request* request,
                          Response* response,
                          std::function<void (const Request*, Response*, bool, int)> callback) {
    bool failed = rpc_controller->Failed();
    int error = rpc_controller->ErrorCode();
    if (failed || error) {
      assert(failed && error);
      if (error != sofa::pbrpc::RPC_ERROR_SEND_BUFFER_FULL) {
        LOG(WARNING, "RpcClient rpcCallback: %s %s\n",
            rpc_controller->RemoteAddress().c_str(), rpc_controller->ErrorText().c_str());
      } else {
        ///TODO: Retry
      }
    }
    delete rpc_controller; // delete here
    callback(request, response, failed, error);
  }
  
 private:
  typedef std::map<std::string, sofa::pbrpc::RpcChannel*> HostMap;
   
  std::unique_ptr<sofa::pbrpc::RpcClient> rpc_client_;
  HostMap host_map_; // holds channel
  port::Mutex host_map_lock_;
}; 
  
} // namespace bfs  
} // namespace bubblefs

#endif // BUBBLEFS_BFS_NETWORK_RPC_CLIENT_H_