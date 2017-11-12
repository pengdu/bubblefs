// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "platform/voyager_sockaddr.h"
#include "utils/voyager_eventloop.h"
#include "utils/voyager_tcp_server.h"

bubblefs::myvoyager::TcpServer* g_server = nullptr;

namespace bubblefs {
namespace myvoyager {

void DeleteServer() {
  if (g_server) {
    delete g_server;
  }
}

} // namespace myvoyager
} // namespace bubblefs

int main(int argc, char** argv) {
  bubblefs::myvoyager::EventLoop eventloop;
  bubblefs::myvoyager::SockAddr addr(5666);
  g_server = new bubblefs::myvoyager::TcpServer(&eventloop, addr, "Voyager", 4);
  g_server->Start();
  eventloop.RunAfter(5000000, []() { bubblefs::myvoyager::DeleteServer(); });
  eventloop.RunAfter(6000000, [&eventloop]() { eventloop.Exit(); });
  eventloop.Loop();
  return 0;
}