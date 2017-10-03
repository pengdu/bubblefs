// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// bfs/src/rpc/rpc_client.h

#ifndef  BUBBLEFS_RPC_SOFA_PBRPC_CLIENT_H_
#define  BUBBLEFS_RPC_SOFA_PBRPC_CLIENT_H_

#include <assert.h>
#include <functional>
#include "platform/mutex.h"
#include "platform/logging_simple.h"
#include "rpc/base_rpc_client.h"
#include "sofa/pbrpc/pbrpc.h"
#include "utils/threadpool_simple.h"

namespace bubblefs {
namespace rpc {

template <class Stub, class Request, class Response, class Callback>
static bool SendRequest(Stub* stub, void(Stub::*func)(
                        google::protobuf::RpcController*,
                        const Request*, Response*, Callback*),
                        const Request* request, Response* response,
                        const RpcOptions& rpc_options) {
  int32_t rpc_timeout = rpc_options.rpc_timeout;
  int retry_times = rpc_options.retry_times;
  sofa::pbrpc::RpcController controller;
  controller.SetTimeout(rpc_timeout * 1000L);
  for (int32_t retry = 0; retry < retry_times; ++retry) {
    (stub->*func)(&controller, request, response, NULL);
    if (controller.Failed()) {
      if (retry < retry_times - 1) {
        Log(DEBUG, "Send failed, retry ...");
        usleep(1000000);
      } else {
        Log(WARNING, "SendRequest to %s fail: %s\n",
            controller.RemoteAddress().c_str(), controller.ErrorText().c_str());
      }
    } else {
      return true;
    }
    controller.Reset();
  }
  return false;
}
   
template <class Request, class Response, class Callback>
static void RpcCallback(sofa::pbrpc::RpcController* rpc_controller,
                        const Request* request,
                        Response* response,
                        std::function<void (const Request*, Response*, bool, int)> callback) {
  bool failed = rpc_controller->Failed();
  int error = rpc_controller->ErrorCode();
  if (failed || error) {
    assert(failed && error);
    if (error != sofa::pbrpc::RPC_ERROR_SEND_BUFFER_FULL) {
      Log(WARNING, "RpcCallback: %s %s\n",
          rpc_controller->RemoteAddress().c_str(), rpc_controller->ErrorText().c_str());
    } else {
      ///TODO: Retry
    }
  }
  delete rpc_controller; // Note: dlete rpc_controller in the callback
  callback(request, response, failed, error);
} 
    
template <class Stub, class Request, class Response, class Callback>
static void AsyncRequest(Stub* stub, void(Stub::*func) (
                         google::protobuf::RpcController*,
                         const Request*, Response*, Callback*),
                         const Request* request, Response* response,
                         std::function<void (const Request*, Response*, bool, int)> callback,
                         const RpcOptions& rpc_options) {
  int32_t rpc_timeout = rpc_options.rpc_timeout;
  int retry_times = rpc_options.retry_times;
  sofa::pbrpc::RpcController* controller = new sofa::pbrpc::RpcController();
  controller->SetTimeout(rpc_timeout * 1000L);
  google::protobuf::Closure* done =
    sofa::pbrpc::NewClosure(&RpcCallback<Request, Response, Callback>,
                            controller, request, response, callback);
  (stub->*func)(controller, request, response, done);
} 
  
class SofaPbrpcClient : public RpcClient {
 public:
   SofaPbrpcClient() {
     sofa::pbrpc::RpcClientOptions options;
     options.max_pending_buffer_size = 128;
     rpc_client_ = new sofa::pbrpc::RpcClient(options);
   }
    
   virtual ~SofaPbrpcClient() {
     delete rpc_client_;
   }
   
   virtual google::protobuf::RpcChannel& GetRpcChannel(const std::string server) override;
   
   virtual std::string RemoteAddress(google::protobuf::RpcController*) const override;
   
   template <class T>
   bool GetStub(const std::string server, T** stub) {
     MutexLock lock(&host_map_lock_);
     sofa::pbrpc::RpcChannel* channel = nullptr;
     HostMap::iterator it = host_map_.find(server);
     if (it != host_map_.end()) {
       channel = it->second;
     } else {
       sofa::pbrpc::RpcChannelOptions channel_options;
       channel = new sofa::pbrpc::RpcChannel(rpc_client_, server, channel_options);
       host_map_[server] = channel;
     }
     *stub = new T(channel);
     return true;
   }
    
 private:
   sofa::pbrpc::RpcClient* rpc_client_;
   typedef std::map<std::string, sofa::pbrpc::RpcChannel*> HostMap;
   HostMap host_map_;
   port::Mutex host_map_lock_;
};

} // namespace rpc
} // namespace bubblefs

#endif  // BUBBLEFS_RPC_SOFA_PBRPC_CLIENT_H_