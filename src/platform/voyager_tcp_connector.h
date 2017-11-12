// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef VOYAGER_CORE_TCP_CONNECTOR_H_
#define VOYAGER_CORE_TCP_CONNECTOR_H_

#include <netdb.h>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include "platform/voyager_client_socket.h"
#include "platform/voyager_sockaddr.h"
#include "utils/voyager_callback.h"
#include "utils/voyager_dispatch.h"
#include "utils/voyager_timeops.h"

#ifdef __linux__
#include "platform/voyager_newtimer.h"
#endif

namespace bubblefs {
namespace myvoyager {

class EventLoop;

class TcpConnector : public std::enable_shared_from_this<TcpConnector> {
 public:
  typedef std::function<void(int fd)> NewConnectionCallback;

  TcpConnector(EventLoop* ev, const SockAddr& addr);

  void SetNewConnectionCallback(const NewConnectionCallback& cb) {
    newconnection_cb_ = cb;
  }
  void SetConnectFailureCallback(const ConnectFailureCallback& cb) {
    connect_failure_cb_ = cb;
  }
  void SetNewConnectionCallback(NewConnectionCallback&& cb) {
    newconnection_cb_ = std::move(cb);
  }
  void SetConnectFailureCallback(ConnectFailureCallback&& cb) {
    connect_failure_cb_ = std::move(cb);
  }

  void Start(bool retry = true);
  void Stop();

 private:
  enum ConnectState { kDisConnected, kConnected, kConnecting };

  static const uint64_t kMaxRetryTime = 30000000;
  static const uint64_t kInitRetryTime = 2000000;

  void StartInLoop();
  void Connect();
  void Connecting();
  void Retry();

  void HandleEvent();

  void ResetDispatch();

  std::string StateToString() const;

  EventLoop* ev_;
  SockAddr addr_;
  ConnectState state_;
  bool connect_;
  bool retry_;
  uint64_t retry_time_;
  std::unique_ptr<Dispatch> dispatch_;
  std::unique_ptr<ClientSocket> socket_;
  NewConnectionCallback newconnection_cb_;
  ConnectFailureCallback connect_failure_cb_;

#ifdef __linux__
  std::unique_ptr<NewTimer> timer_;
#endif

  // No copying allowed
  TcpConnector(const TcpConnector&);
  void operator=(const TcpConnector&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_VOYAGER_TCP_CONNECTOR_H_