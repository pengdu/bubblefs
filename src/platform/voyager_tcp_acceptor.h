// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/tcp_acceptor.h

#ifndef BUBBLEFS_PLATFORM_VOYAGER_TCP_ACCEPTOR_H_
#define BUBBLEFS_PLATFORM_VOYAGER_TCP_ACCEPTOR_H_

#include <netdb.h>
#include <functional>
#include <utility>
#include "platform/voyager_server_socket.h"
#include "utils/voyager_dispatch.h"

namespace bubblefs {
namespace myvoyager {

class SockAddr;
class EventLoop;

class TcpAcceptor {
 public:
  typedef std::function<void(int fd, const struct sockaddr_storage& sa)>
      NewConnectionCallback;

  TcpAcceptor(EventLoop* eventloop, const SockAddr& addr, int backlog,
              bool reuseport);
  ~TcpAcceptor();

  void EnableListen();
  bool IsListenning() const { return listenning_; }

  void SetNewConnectionCallback(const NewConnectionCallback& cb) {
    conn_cb_ = cb;
  }
  void SetNewConnectionCallback(NewConnectionCallback&& cb) {
    conn_cb_ = std::move(cb);
  }

 private:
  void Accept();

  EventLoop* eventloop_;
  ServerSocket socket_;
  Dispatch dispatch_;
  int backlog_;
  int idlefd_;
  bool listenning_;
  NewConnectionCallback conn_cb_;

  // No copying alloweded
  TcpAcceptor(const TcpAcceptor&);
  void operator=(const TcpAcceptor&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_VOYAGER_TCP_ACCEPTOR_H_