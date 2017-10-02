// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

#ifndef  BUBBLEFS_RPC_SOFA_PBRPC_CLIENT_H_
#define  BUBBLEFS_RPC_SOFA_PBRPC_CLIENT_H_

#include <assert.h>
#include <functional>
#include "platform/mutex.h"
#include "platform/logging_simple.h"
#include "sofa/pbrpc/pbrpc.h"
#include "utils/threadpool_simple.h"

namespace bubblefs {
namespace rpc {

class SofaPbrpcClient {
public:
    SofaPbrpcClient() {
        sofa::pbrpc::RpcClientOptions options;
        options.max_pending_buffer_size = 128;
        _rpc_client = new sofa::pbrpc::RpcClient(options);
    }
    
    ~SofaPbrpcClient() {
        delete _rpc_client;
    }
    
    template <class T>
    bool GetStub(const std::string server, T** stub) {
        MutexLock lock(&_host_map_lock);
        sofa::pbrpc::RpcChannel* channel = NULL;
        HostMap::iterator it = _host_map.find(server);
        if (it != _host_map.end()) {
            channel = it->second;
        } else {
            sofa::pbrpc::RpcChannelOptions channel_options;
            channel = new sofa::pbrpc::RpcChannel(_rpc_client, server, channel_options);
            _host_map[server] = channel;
        }
        *stub = new T(channel);
        return true;
    }
    
    template <class Stub, class Request, class Response, class Callback>
    bool SendRequest(Stub* stub, void(Stub::*func)(
                    google::protobuf::RpcController*,
                    const Request*, Response*, Callback*),
                    const Request* request, Response* response,
                    int32_t rpc_timeout, int retry_times) {
        sofa::pbrpc::RpcController controller;
        controller.SetTimeout(rpc_timeout * 1000L);
        for (int32_t retry = 0; retry < retry_times; ++retry) {
            (stub->*func)(&controller, request, response, NULL);
            if (controller.Failed()) {
                if (retry < retry_times - 1) {
                    LOG(DEBUG, "Send failed, retry ...");
                    usleep(1000000);
                } else {
                    LOG(WARNING, "SendRequest to %s fail: %s\n",
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
    void AsyncRequest(Stub* stub, void(Stub::*func)(
                    google::protobuf::RpcController*,
                    const Request*, Response*, Callback*),
                    const Request* request, Response* response,
                    std::function<void (const Request*, Response*, bool, int)> callback,
                    int32_t rpc_timeout, int retry_times) {
        sofa::pbrpc::RpcController* controller = new sofa::pbrpc::RpcController();
        controller->SetTimeout(rpc_timeout * 1000L);
        google::protobuf::Closure* done =
            sofa::pbrpc::NewClosure(&SofaPbrpcClient::template RpcCallback<Request, Response, Callback>,
                                          controller, request, response, callback);
        (stub->*func)(controller, request, response, done);
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
                LOG(WARNING, "RpcCallback: %s %s\n",
                    rpc_controller->RemoteAddress().c_str(), rpc_controller->ErrorText().c_str());
            } else {
                ///TODO: Retry
            }
        }
        delete rpc_controller;
        callback(request, response, failed, error);
    }
    
private:
    sofa::pbrpc::RpcClient* _rpc_client;
    typedef std::map<std::string, sofa::pbrpc::RpcChannel*> HostMap;
    HostMap _host_map;
    port::Mutex _host_map_lock;
};

} // namespace rpc
} // namespace bubblefs

#endif  // BUBBLEFS_RPC_SOFA_PBRPC_CLIENT_H_