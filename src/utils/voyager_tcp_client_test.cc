// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "platform/voyager_logging.h"
#include "platform/voyager_sockaddr.h"
#include "platform/voyager_tcp_connection.h"
#include "utils/voyager_tcp_client.h"
#include "utils/voyager_buffer.h"
#include "utils/voyager_callback.h"
#include "utils/voyager_eventloop.h"
#include "utils/stringpiece.h"
#include <boost/concept_check.hpp>

bubblefs::voyager::TcpClient* g_client = nullptr;

namespace bubblefs {
namespace voyager {

void OnMessage(const TcpConnectionPtr& p, Buffer* buf) {
  std::string s = buf->RetrieveAllAsString();
  VOYAGER_LOG(INFO) << s;
  if (s == "Nice!") {
    Slice message = "That's OK! I close!";
    p->SendMessage(message);
  } else if (s == "Bye!") {
    VOYAGER_LOG(INFO) << p->StateToString();
    p->ShutDown();
    VOYAGER_LOG(INFO) << p->StateToString();
  } else {
    Slice message = "Yes, I know!";
    p->SendMessage(message);
  }
}

void DeleteClient() { delete g_client; }

}  // namespace voyager
}  // namespace bubblefs

using namespace std::placeholders;

int main(int argc, char** argv) {
  bubblefs::voyager::EventLoop ev;
  bubblefs::voyager::SockAddr serveraddr("127.0.0.1", 5666);
  g_client = new bubblefs::voyager::TcpClient(&ev, serveraddr);
  g_client->SetMessageCallback(std::bind(bubblefs::voyager::OnMessage, _1, _2));
  g_client->Connect();
  ev.RunAfter(15000000, []() { bubblefs::voyager::DeleteClient(); });
  ev.RunAfter(20000000, [&ev]() { ev.Exit(); });
  ev.Loop();
  return 0;
}