// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/src/pink_conn.cc

#include "utils/pink_conn.h"
#include <stdio.h>
#include <unistd.h>
#include "platform/pink_server_socket.h"
#include "platform/slash_xdebug.h"
#include "utils/pink_thread.h"

namespace bubblefs {
namespace mypink {

PinkConn::PinkConn(const int fd,
                   const std::string &ip_port,
                   ServerThread *thread)
    : fd_(fd),
      ip_port_(ip_port),
      is_reply_(false),
#ifdef TF_USE_SSL
      ssl_(nullptr),
#endif
      server_thread_(thread) {
  gettimeofday(&last_interaction_, nullptr);
}

PinkConn::~PinkConn() {
#ifdef TF_USE_SSL
  SSL_free(ssl_);
  ssl_ = nullptr;
#endif
}

bool PinkConn::SetNonblock() {
  flags_ = Setnonblocking(fd());
  if (flags_ == -1) {
    return false;
  }
  return true;
}

#ifdef TF_USE_SSL
bool PinkConn::CreateSSL(SSL_CTX* ssl_ctx) {
  ssl_ = SSL_new(ssl_ctx);
  if (!ssl_) {
    log_warn("SSL_new() failed");
    return false;
  }

  if (SSL_set_fd(ssl_, fd_) == 0) {
    log_warn("SSL_set_fd() failed");
    return false;
  }

  SSL_set_accept_state(ssl_);

  return true;
}
#endif

}  // namespace mypink
}  // namespace bubblefs