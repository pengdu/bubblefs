// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/server_socket.h

#ifndef BUBBLEFS_PLATFORM_VOYAGER_SERVER_SOCKET_H_
#define BUBBLEFS_PLATFORM_VOYAGER_SERVER_SOCKET_H_

#include "platform/voyager_base_socket.h"

namespace bubblefs {
namespace myvoyager {

class ServerSocket : public BaseSocket {
 public:
  ServerSocket(int domain, bool nonblocking)
      : BaseSocket(domain, nonblocking) {}

  explicit ServerSocket(int socketfd) : BaseSocket(socketfd) {}

  void Bind(const struct sockaddr* sa, socklen_t salen);
  void Listen(int backlog);
  int Accept(struct sockaddr* sa, socklen_t* salen);

 private:
  // No copying allowed
  ServerSocket(const ServerSocket&);
  void operator=(const ServerSocket&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_VOYAGER_SERVER_SOCKET_H_