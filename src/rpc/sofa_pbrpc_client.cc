
#include "platform/mutexlock.h"
#include "rpc/sofa_pbrpc_client.h"

std::string RemoteAddress(google::protobuf::RpcController* rpc_controller) {
  if (nullptr == rpc_controller)
    return "";
  sofa::pbrpc::RpcController* sofa_cntl
    = static_cast<sofa::pbrpc::RpcController*>(rpc_controller);
  return sofa_cntl->RemoteAddress();
}

int ErrorCode(google::protobuf::RpcController* rpc_controller) {
  if (nullptr == rpc_controller)
    return sofa::pbrpc::RPC_ERROR_FROM_USER;
  sofa::pbrpc::RpcController* sofa_cntl = 
    static_cast<sofa::pbrpc::RpcController*>(rpc_controller);
  return sofa_cntl->ErrorCode();
}

google::protobuf::RpcChannel* bubblefs::rpc::SofaPbrpcClient::GetRpcChannel(const std::string server) {
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
  return channel;
}
