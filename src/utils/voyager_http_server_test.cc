// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <stdio.h>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include "platform/voyager_logging.h"
#include "utils/voyager_eventloop.h"
#include "utils/voyager_http_request.h"
#include "utils/voyager_http_response.h"
#include "utils/voyager_http_server.h"

namespace bubblefs {
namespace myvoyager {

void HandleHttpRequest(HttpRequestPtr request, HttpResponse* response) {
  std::cout << request->RequestMessage().Peek() << std::endl;

  response->SetVersion(request->Version());
  response->SetStatusCode("200");
  response->SetReasonParse("OK");
  response->AddHeader("Content-Type", "text/html; charset=UTF-8");
  response->AddHeader("Content-Encoding", "UTF-8");
  response->AddHeader("Connection", "close");

  std::string s("Welcome to Voyager's WebServer!");
  char buf[32];
  snprintf(buf, sizeof(buf), "%zu", s.size());
  response->AddHeader("Content-Length", buf);
  response->SetBody(std::move(s));

  response->SetCloseState(true);
}

}  // namespace myvoyager
}  // namespace bubblefs

int main() {
  bubblefs::myvoyager::EventLoop ev;
  bubblefs::myvoyager::SockAddr addr(5666);
  bubblefs::myvoyager::HttpServer server(&ev, addr, "WebServer", 4);
  server.SetHttpCallback(std::bind(bubblefs::myvoyager::HandleHttpRequest,
                                   std::placeholders::_1,
                                   std::placeholders::_2));
  server.Start();
  ev.Loop();
  return 0;
}