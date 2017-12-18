// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/http/http_client.h

#ifndef BUBBLEFS_UTILS_VOYAGER_HTTP_CLIENT_H_
#define BUBBLEFS_UTILS_VOYAGER_HTTP_CLIENT_H_

#include <deque>
#include <functional>
#include <memory>
#include "utils/voyager_http_request.h"
#include "utils/voyager_http_response.h"
#include "utils/status.h"
#include "utils/voyager_tcp_client.h"
#include "utils/voyager_timerlist.h"

namespace bubblefs {
namespace myvoyager {

// Noly write for test, which cann't use in produce environment!
class HttpClient {
 public:
  typedef std::function<void(HttpResponsePtr, const Status&)> RequestCallback;
  explicit HttpClient(EventLoop* ev, uint64_t timeout = 15);

  void DoHttpRequest(const HttpRequestPtr& request, const RequestCallback& cb);

 private:
  void DoHttpRequestInLoop(const HttpRequestPtr& request,
                           const RequestCallback& cb);
  void FirstRequest(const HttpRequestPtr& request);

  void HandleMessage(const TcpConnectionPtr& ptr, Buffer* buffer);
  void HandleClose(const TcpConnectionPtr& ptr);
  void HandleTimeout();

  EventLoop* eventloop_;
  uint64_t timeout_;
  TimerId timer_;
  std::weak_ptr<TcpConnection> gaurd_;
  std::unique_ptr<TcpClient> client_;
  typedef std::deque<RequestCallback> CallbackQueue;
  CallbackQueue queue_cb_;

  // No copying allowed
  HttpClient(const HttpClient&);
  void operator=(const HttpClient&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_HTTP_CLIENT_H_