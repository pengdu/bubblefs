// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/client_socket.cc

#include "platform/voyager_client_socket.h"

namespace bubblefs {
namespace myvoyager {

int ClientSocket::Connect(const struct sockaddr* sa, socklen_t salen) const {
  return ::connect(fd_, sa, salen);
}

}  // namespace myvoyager
}  // namespace bubblefs