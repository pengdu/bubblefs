// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <functional>
#include <iostream>
#include "http/voyager_http_client.h"
#include "utils/status.h"
#include "utils/voyager_eventloop.h"

namespace bubblefs {
namespace voyager {

void HandleResponse(HttpResponsePtr response, const Status& s) {
  if (s.ok()) {
    std::cout << response->ResponseMessage().RetrieveAllAsString() << "\n";
  } else {
    std::cout << s.ToString() << "\n";
  }
}

}  // namespace voyager
}  // namespace bubblefs

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: httpclient <host> <path>" << std::endl;
    return 1;
  }

  const char* host = argv[1];
  const char* path = argv[2];

  bubblefs::voyager::EventLoop ev;

  bubblefs::voyager::HttpClient client(&ev);
  bubblefs::voyager::HttpRequestPtr request(new bubblefs::voyager::HttpRequest());
  request->SetMethod(bubblefs::voyager::HttpRequest::kGet);
  request->SetPath(path);
  request->SetVersion(bubblefs::voyager::HttpMessage::kHttp11);
  request->AddHeader("Host", host);
  request->AddHeader("Connection", "keep-alive");

  client.DoHttpRequest(request,
                       std::bind(bubblefs::voyager::HandleResponse, std::placeholders::_1,
                                 std::placeholders::_2));
  ev.Loop();
}