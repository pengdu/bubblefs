// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "platform/voyager_sockaddr.h"
#include "utils/voyager_eventloop.h"
#include "utils/voyager_tcp_server.h"

bubblefs::voyager::TcpServer* g_server = nullptr;

namespace bubblefs {
namespace voyager {

void DeleteServer() {
  if (g_server) {
    delete g_server;
  }
}

} // namespace voyager
} // namespace bubblefs

int main(int argc, char** argv) {
  bubblefs::voyager::EventLoop eventloop;
  bubblefs::voyager::SockAddr addr(5666);
  g_server = new bubblefs::voyager::TcpServer(&eventloop, addr, "Voyager", 4);
  g_server->Start();
  eventloop.RunAfter(5000000, []() { bubblefs::voyager::DeleteServer(); });
  eventloop.RunAfter(6000000, [&eventloop]() { eventloop.Exit(); });
  eventloop.Loop();
  return 0;
}