// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/include/pink_cli.h

#ifndef BUBBLEFS_UTILS_PINK_CLI_H_
#define BUBBLEFS_UTILS_PINK_CLI_H_

#include <string>
#include "utils/slash_status.h"

using bubblefs::myslash::Status;

namespace bubblefs {
namespace mypink {

// Usage like:
// PinkCli* cli = NewPbCli();
// Status s = cli->Connect(ip, port);
// myproto::Ping msg; msg.set_address("127.00000"); msg.set_port(2222);
// s = cli->Send((void *)&msg);
// myproto::PingRes req;
// s = cli->Recv((void *)&req);
// cli->Close();
// DeletePinkCli(&cli);  
  
class PinkCli {
 public:
  explicit PinkCli(const std::string& ip = "", const int port = 0);
  virtual ~PinkCli();

  Status Connect(const std::string& bind_ip = "");
  Status Connect(const std::string &peer_ip, const int peer_port,
      const std::string& bind_ip = "");
  // Compress and write the message
  virtual Status Send(void *msg) = 0;

  // Read, parse and store the reply
  virtual Status Recv(void *result = NULL) = 0;

  void Close();

  // TODO(baotiao): delete after redis_cli use RecvRaw
  int fd() const;

  bool Available() const;

  // default connect timeout is 1000ms
  int set_send_timeout(int send_timeout);
  int set_recv_timeout(int recv_timeout);
  void set_connect_timeout(int connect_timeout);

 protected:
  Status SendRaw(void* buf, size_t len);
  Status RecvRaw(void* buf, size_t* len);

 private:
  struct Rep;
  Rep* rep_;
  int set_tcp_nodelay();

  PinkCli(const PinkCli&);
  void operator=(const PinkCli&);
};

extern PinkCli *NewPbCli(
    const std::string& peer_ip = "",
    const int peer_port = 0);

extern PinkCli *NewRedisCli();

void DeletePinkCli(PinkCli** cli);
  
}  // namespace mypink
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PINK_CLI_H_