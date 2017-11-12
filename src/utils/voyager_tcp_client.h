// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/tcp_client.h

#ifndef BUBBLEFS_UTILS_VOYAGER_TCP_CLIENT_H_
#define BUBBLEFS_UTILS_VOYAGER_TCP_CLIENT_H_

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include "platform/voyager_sockaddr.h"
#include "platform/voyager_tcp_connection.h"
#include "utils/voyager_eventloop.h"

namespace bubblefs {
namespace myvoyager {

class TcpConnector;

typedef std::shared_ptr<TcpConnector> TcpConnectorPtr;

class TcpClient {
 public:
  TcpClient(EventLoop* ev, const SockAddr& addr,
            const std::string& name = "VoyagerClient");
  ~TcpClient();

  void Connect(bool retry = true);
  void Close();

  EventLoop* GetLoop() const { return ev_; }
  TcpConnectionPtr GetTcpConnectionPtr() { return weak_ptr_.lock(); }

  void SetConnectionCallback(const ConnectionCallback& cb) {
    connection_cb_ = cb;
  }
  void SetConnectFailureCallback(const ConnectFailureCallback& cb);

  void SetCloseCallback(const CloseCallback& cb) { close_cb_ = cb; }
  void SetMessageCallback(const MessageCallback& cb) { message_cb_ = cb; }
  void SetWriteCompleteCallback(const WriteCompleteCallback& cb) {
    writecomplete_cb_ = cb;
  }

  void SetConnectionCallback(ConnectionCallback&& cb) {
    connection_cb_ = std::move(cb);
  }
  void SetConnectFailureCallback(ConnectFailureCallback&& cb);
  void SetCloseCallback(CloseCallback&& cb) { close_cb_ = std::move(cb); }
  void SetMessageCallback(MessageCallback&& cb) { message_cb_ = std::move(cb); }
  void SetWriteCompleteCallback(WriteCompleteCallback&& cb) {
    writecomplete_cb_ = std::move(cb);
  }

 private:
  void NewConnection(int fd);
  void ConnectFailure();

  static std::atomic<int> conn_id_;

  EventLoop* ev_;
  SockAddr addr_;
  std::string name_;
  TcpConnectorPtr connector_;
  std::atomic<bool> connect_;

  ConnectionCallback connection_cb_;
  CloseCallback close_cb_;
  MessageCallback message_cb_;
  WriteCompleteCallback writecomplete_cb_;

  std::weak_ptr<TcpConnection> weak_ptr_;

  // No copying allowed
  TcpClient(const TcpClient&);
  void operator=(const TcpClient&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_TCP_CLIENT_H_