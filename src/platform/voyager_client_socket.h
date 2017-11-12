// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/client_socket.h

#ifndef BUBBLEFS_PLATFORM_VOYAGER_CLIENT_SOCKET_H_
#define BUBBLEFS_PLATFORM_VOYAGER_CLIENT_SOCKET_H_

#include "platform/voyager_base_socket.h"

namespace bubblefs {
namespace myvoyager {

class ClientSocket : public BaseSocket {
 public:
  ClientSocket(int domain, bool nonblocking)
      : BaseSocket(domain, nonblocking) {}

  explicit ClientSocket(int socketfd) : BaseSocket(socketfd) {}

  int Connect(const struct sockaddr* sa, socklen_t salen) const;

 private:
  // No copying allowed
  ClientSocket(const ClientSocket&);
  void operator=(const ClientSocket&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_VOYAGER_CLIENT_SOCKET_H_