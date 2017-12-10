// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/examples/https_server.cc

#include <signal.h>
#include <atomic>
#include <chrono>
#include <string>
#include "utils/pink_http_conn.h"
#include "utils/pink_thread.h"
#include "utils/pink_server_thread.h"
#include "utils/slash_hash.h"

namespace bubblefs {

class MyHTTPHandles : public mypink::HTTPHandles {
 public:
  std::string body_data;
  std::string body_md5;
  std::string zero_space;
  size_t write_pos = 0;
  std::chrono::system_clock::time_point start, end;
  std::chrono::duration<double, std::milli> diff;

  // Request handles
  virtual bool HandleRequest(const mypink::HTTPRequest* req) {
    req->Dump();
    body_data.clear();

    start = std::chrono::system_clock::now();

    // Continue receive body
    return false;
  }
  virtual void HandleBodyData(const char* data, size_t size) {
    std::cout << "ReqBodyPartHandle: " << size << std::endl;
    body_data.append(data, size);
  }

  // Response handles
  virtual void PrepareResponse(mypink::HTTPResponse* resp) {
    body_md5.assign(myslash::md5(body_data));

    resp->SetStatusCode(200);
    resp->SetContentLength(body_md5.size());
    write_pos = 0;
    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << "Use: " << diff.count() << " ms" << std::endl;
  }

  virtual int WriteResponseBody(char* buf, size_t max_size) {
    size_t size = std::min(max_size, body_md5.size() - write_pos);
    memcpy(buf, body_md5.data() + write_pos, size);
    write_pos += size;
    return size;
  }
};

class MyConnFactory : public mypink::ConnFactory {
 public:
  virtual mypink::PinkConn* NewPinkConn(int connfd, const std::string& ip_port,
                                mypink::ServerThread* thread,
                                void* worker_specific_data) const {
    auto my_handles = std::make_shared<MyHTTPHandles>();
    return new mypink::HTTPConn(connfd, ip_port, thread, my_handles,
                              worker_specific_data);
  }
};

static std::atomic<bool> running(false);

static void IntSigHandle(const int sig) {
  printf("Catch Signal %d, cleanup...\n", sig);
  running.store(false);
  printf("server Exit");
}

static void SignalSetup() {
  signal(SIGHUP, SIG_IGN);
  signal(SIGPIPE, SIG_IGN);
  signal(SIGINT, &IntSigHandle);
  signal(SIGQUIT, &IntSigHandle);
  signal(SIGTERM, &IntSigHandle);
}

int main(int argc, char* argv[]) {
  int port = 0;
  if (argc < 2) {
    printf("Usage: ./http_server port");
  } else {
    port = atoi(argv[1]);
  }

  SignalSetup();

  mypink::ConnFactory* my_conn_factory = new MyConnFactory();
  mypink::ServerThread *st = mypink::NewDispatchThread(port, 4, my_conn_factory, 1000);

  if (st->EnableSecurity("/complete_path_to/host.crt",
                         "/complete_path_to/host.key") != 0) {
    printf("EnableSecurity error happened!\n");
    exit(-1);
  }

  if (st->StartThread() != 0) {
    printf("StartThread error happened!\n");
    exit(-1);
  }
  running.store(true);
  while (running.load()) {
    sleep(1);
  }
  st->StopThread();

  delete st;
  delete my_conn_factory;

  return 0;
}

} // namespace bubblefs