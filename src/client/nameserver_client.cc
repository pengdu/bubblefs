
#include "client/nameserver_client.h"
#include "utils/bdcommon_str_util.h"
#include "rpc/rpc_client.h"

namespace bubblefs {
namespace bfs {

NameServerClient::NameServerClient(rpc::RpcClient* rpc_client, const string& nameserver_nodes) 
                                   : rpc_client_(rpc_client), leader_id_(0) {
  bdcommon::SplitString(nameserver_nodes, ",", &nameserver_nodes_);
  stubs_.resize(nameserver_nodes_.size());
  for (uint32_t i = 0; i < nameserver_nodes_.size(); ++i) {
    rpc::NewStub(rpc_client_, nameserver_nodes_[i], &stubs_[i]);
  }
}
  
} // namespace bfs  
} // namespace bubblefs