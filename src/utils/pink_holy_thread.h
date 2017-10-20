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
namespace pink {
  
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


}  // namespace pink
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PINK_HOLY_THREAD_H_