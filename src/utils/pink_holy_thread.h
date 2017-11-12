// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/src/holy_thread.h

#ifndef BUBBLEFS_UTILS_PINK_HOLY_THREAD_H_
#define BUBBLEFS_UTILS_PINK_HOLY_THREAD_H_

#include <atomic>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "platform/mutexlock.h"
#include "utils/pink_conn.h"
#include "utils/pink_server_thread.h"

namespace bubblefs {
namespace mypink {
  
// Usage like:
// class MyConn: public PbConn {
//  public:
//   MyConn(int fd, const std::string& ip_port, ServerThread *thread,
//          void* worker_specific_data)
//        : PbConn(fd, ip_port, thread),
//          thread_(thread),
//          private_data_(static_cast<int*>(worker_specific_data)) { };
//   virtual ~MyConn() { };
//   ServerThread* thread() {
//     return thread_;
//   }
//  protected:
//   virtual int DealMessage() {
//     ping_.ParseFromArray(rbuf_ + cur_pos_ - header_len_, header_len_);
//     int* data = static_cast<int*>(private_data_);
//     ping_res_.Clear(); ping_res_.set_res(11234); ping_res_.set_mess("heiheidfdfdf");
//     res_ = &ping_res_;
//     set_is_reply(true);
//   };
//  private:
//   ServerThread *thread_;
//   int* private_data_;
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
// class MyServerHandle : public ServerHandle {
// public:
//  virtual void CronHandle() const override {
//    printf ("Cron operation\n");
//  }
//  using ServerHandle::AccessHandle;
//  virtual bool AccessHandle(std::string& ip) const override {
//    printf ("Access operation, receive:%s\n", ip.c_str());
//    return true;
//  }
//  virtual int CreateWorkerSpecificData(void** data) const {
//    int *num = new int(1234);
//    *data = static_cast<void*>(num);
//    return 0;
//  }
//  virtual int DeleteWorkerSpecificData(void* data) const {
//    delete static_cast<int*>(data);
//    return 0;
//  }
//};
//
// SignalSetup();
// int my_port = (argc > 1) ? atoi(argv[1]) : 8221;
// MyConnFactory conn_factory;
// MyServerHandle handle;
// ServerThread *st = NewHolyThread(my_port, &conn_factory, 1000, &handle);
// st->StartThread();
// while (running.load()) { sleep(1); }
// st->StopThread();
// DeleteServerThread(&st);    
  
class PinkConn;

class HolyThread: public ServerThread {
 public:
  // This type thread thread will listen and work self list redis thread
  HolyThread(int port, ConnFactory* conn_factory,
             int cron_interval = 0, const ServerHandle* handle = nullptr);
  HolyThread(const std::string& bind_ip, int port,
             ConnFactory* conn_factory,
             int cron_interval = 0, const ServerHandle* handle = nullptr);
  HolyThread(const std::set<std::string>& bind_ips, int port,
             ConnFactory* conn_factory,
             int cron_interval = 0, const ServerHandle* handle = nullptr);
  virtual ~HolyThread();

  virtual int StartThread() override;

  virtual int StopThread() override;

  virtual void set_keepalive_timeout(int timeout) override {
    keepalive_timeout_ = timeout;
  }

  virtual int conn_num() const override;

  virtual std::vector<ServerThread::ConnInfo> conns_info() const override;

  virtual PinkConn* MoveConnOut(int fd) override;

  virtual void KillAllConns() override;

  virtual bool KillConn(const std::string& ip_port) override;

 private:
  mutable port::RWMutex rwlock_; /* For external statistics */
  std::map<int, PinkConn*> conns_;

  ConnFactory *conn_factory_;
  void* private_data_;

  std::atomic<int> keepalive_timeout_;  // keepalive second

  void DoCronTask() override;

  port::Mutex killer_mutex_;
  std::set<std::string> deleting_conn_ipport_;

  void HandleNewConn(int connfd, const std::string &ip_port) override;
  void HandleConnEvent(PinkFiredEvent *pfe) override;

  void CloseFd(PinkConn* conn);
  void Cleanup();
};  // class HolyThread

}  // namespace mypink
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PINK_HOLY_THREAD_H_