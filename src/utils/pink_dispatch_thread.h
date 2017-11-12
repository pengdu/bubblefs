// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/src/dispatch_thread.h

#ifndef BUBBLEFS_UTILS_PINK_DISPATCH_THREAD_H_
#define BUBBLEFS_UTILS_PINK_DISPATCH_THREAD_H_

#include <map>
#include <queue>
#include <set>
#include <string>
#include <vector>
#include "utils/pink_server_thread.h"

namespace bubblefs {
namespace mypink {

// Usage like:
// class MyConn: public PbConn {
//  public:
//   MyConn(int fd, const std::string& ip_port, ServerThread *thread,
//          void* worker_specific_data) : PbConn(fd, ip_port, thread) { };
//   virtual ~MyConn() { };
//  protected:
//   virtual int DealMessage() {
//     ping_.ParseFromArray(rbuf_ + cur_pos_ - header_len_, header_len_);
//     ping_res_.Clear(); ping_res_.set_res(11234); ping_res_.set_mess("heiheidfdfdf");
//     res_ = &ping_res_;
//     set_is_reply(true);
//   };
//  private:
//   myproto::Ping ping_;
//   myproto::PingRes ping_res_;
// };  
//
// class MyConnFactory : public ConnFactory {
//  public:
//   virtual PinkConn *NewPinkConn(int connfd, const std::string &ip_port,
//                                 ServerThread *thread,
//                                 void* worker_specific_data) const {
//     return new MyConn(connfd, ip_port, thread, worker_specific_data);
//   }
// };
//
// SignalSetup();
// MyConnFactory conn_factory;
// ServerThread *st = NewDispatchThread(9211, 10, &conn_factory, 1000);
// st->StartThread();
// while (running.load()) { sleep(1); }
// st->StopThread();
// DeleteServerThread(&st);  
  
class PinkItem;
struct PinkFiredEvent;
class WorkerThread;

class DispatchThread : public ServerThread {
 public:
  DispatchThread(int port,
                 int work_num, ConnFactory* conn_factory,
                 int cron_interval,
                 int queue_limit,
                 const ServerHandle* handle);
  DispatchThread(const std::string &ip, int port,
                 int work_num, ConnFactory* conn_factory,
                 int cron_interval,
                 int queue_limit,
                 const ServerHandle* handle);
  DispatchThread(const std::set<std::string>& ips, int port,
                 int work_num, ConnFactory* conn_factory,
                 int cron_interval,
                 int queue_limit,
                 const ServerHandle* handle);

  virtual ~DispatchThread();

  virtual int StartThread() override;

  virtual int StopThread() override;

  virtual void set_keepalive_timeout(int timeout) override;

  virtual int conn_num() const override;

  virtual std::vector<ServerThread::ConnInfo> conns_info() const override;

  virtual PinkConn* MoveConnOut(int fd) override;

  virtual void KillAllConns() override;

  virtual bool KillConn(const std::string& ip_port) override;

 private:
  /*
   * Here we used auto poll to find the next work thread,
   * last_thread_ is the last work thread
   */
  int last_thread_;
  int work_num_;
  /*
   * This is the work threads
   */
  WorkerThread** worker_thread_;
  int queue_limit_;
  std::map<WorkerThread*, void*> localdata_;

  void HandleNewConn(const int connfd, const std::string& ip_port) override;
  void HandleConnEvent(PinkFiredEvent *pfe) override {
    EXPR_UNUSED(pfe);
  }

  // No copying allowed
  DispatchThread(const DispatchThread&);
  void operator=(const DispatchThread&);
};

}  // namespace mypink
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PINK_DISPATCH_THREAD_H_