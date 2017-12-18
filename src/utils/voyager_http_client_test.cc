// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <functional>
#include <iostream>
#include "utils/voyager_http_client.h"
#include "utils/status.h"
#include "utils/voyager_eventloop.h"

namespace bubblefs {
namespace myvoyager {

void HandleResponse(HttpResponsePtr response, const Status& s) {
  if (s.ok()) {
    std::cout << response->ResponseMessage().RetrieveAllAsString() << "\n";
  } else {
    std::cout << s.ToString() << "\n";
  }
}

}  // namespace myvoyager
}  // namespace bubblefs

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: httpclient <host> <path>" << std::endl;
    return 1;
  }

  const char* host = argv[1];
  const char* path = argv[2];

  bubblefs::myvoyager::EventLoop ev;

  bubblefs::myvoyager::HttpClient client(&ev);
  bubblefs::myvoyager::HttpRequestPtr request(new bubblefs::myvoyager::HttpRequest());
  request->SetMethod(bubblefs::myvoyager::HttpRequest::kGet);
  request->SetPath(path);
  request->SetVersion(bubblefs::myvoyager::HttpMessage::kHttp11);
  request->AddHeader("Host", host);
  request->AddHeader("Connection", "keep-alive");

  client.DoHttpRequest(request,
                       std::bind(bubblefs::myvoyager::HandleResponse, std::placeholders::_1,
                                 std::placeholders::_2));
  ev.Loop();
}