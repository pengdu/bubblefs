// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/src/worker_thread.cc

#include <vector>
#include "platform/pink_epoll.h"
#include "utils/pink_conn.h"
#include "utils/pink_item.h"
#include "utils/pink_worker_thread.h"

namespace bubblefs {
namespace pink {

WorkerThread::WorkerThread(ConnFactory *conn_factory,
                           ServerThread* server_thread,
                           int cron_interval)
      : server_thread_(server_thread),
        conn_factory_(conn_factory),
        cron_interval_(cron_interval),
        keepalive_timeout_(kDefaultKeepAliveTime) {
  /*
   * install the protobuf handler here
   */
  pink_epoll_ = new PinkEpoll();
  int fds[2];
  if (pipe(fds)) {
    exit(-1);
  }
  notify_receive_fd_ = fds[0];
  notify_send_fd_ = fds[1];
  pink_epoll_->PinkAddEvent(notify_receive_fd_, EPOLLIN | EPOLLERR | EPOLLHUP);
}

WorkerThread::~WorkerThread() {
  delete(pink_epoll_);
}

int WorkerThread::conn_num() const {
  ReadLock l(&rwlock_);
  return conns_.size();
}

std::vector<ServerThread::ConnInfo> WorkerThread::conns_info() const {
  std::vector<ServerThread::ConnInfo> result;
  ReadLock l(&rwlock_);
  for (auto& conn : conns_) {
    result.push_back({
                      conn.first,
                      conn.second->ip_port(),
                      conn.second->last_interaction()
                     });
  }
  return result;
}

PinkConn* WorkerThread::MoveConnOut(int fd) {
  WriteLock l(&rwlock_);
  PinkConn* conn = nullptr;
  auto iter = conns_.find(fd);
  if (iter != conns_.end()) {
    int fd = iter->first;
    conn = iter->second;
    pink_epoll_->PinkDelEvent(fd);
    conns_.erase(iter);
  }
  return conn;
}

void *WorkerThread::ThreadMain() {
  int nfds;
  PinkFiredEvent *pfe = NULL;
  char bb[1];
  PinkItem ti;
  PinkConn *in_conn = NULL;

  struct timeval when;
  gettimeofday(&when, NULL);
  struct timeval now = when;

  when.tv_sec += (cron_interval_ / 1000);
  when.tv_usec += ((cron_interval_ % 1000) * 1000);
  int timeout = cron_interval_;
  if (timeout <= 0) {
    timeout = PINK_CRON_INTERVAL;
  }

  while (!should_stop()) {
    if (cron_interval_ > 0) {
      gettimeofday(&now, NULL);
      if (when.tv_sec > now.tv_sec ||
          (when.tv_sec == now.tv_sec && when.tv_usec > now.tv_usec)) {
        timeout = (when.tv_sec - now.tv_sec) * 1000 +
          (when.tv_usec - now.tv_usec) / 1000;
      } else {
        DoCronTask();
        when.tv_sec = now.tv_sec + (cron_interval_ / 1000);
        when.tv_usec = now.tv_usec + ((cron_interval_ % 1000) * 1000);
        timeout = cron_interval_;
      }
    }

    nfds = pink_epoll_->PinkPoll(timeout);

    for (int i = 0; i < nfds; i++) {
      pfe = (pink_epoll_->firedevent()) + i;
      if (pfe->fd == notify_receive_fd_) {
        if (pfe->mask & EPOLLIN) {
          read(notify_receive_fd_, bb, 1);
          {
            MutexLock l(&mutex_);
            ti = conn_queue_.front();
            conn_queue_.pop();
          }
          PinkConn *tc = conn_factory_->NewPinkConn(
              ti.fd(), ti.ip_port(),
              server_thread_, private_data_);
          if (!tc || !tc->SetNonblock()) {
            delete tc;
            continue;
          }

#ifdef TF_USE_SSL
          // Create SSL failed
          if (server_thread_->security() &&
              !tc->CreateSSL(server_thread_->ssl_ctx())) {
            CloseFd(tc);
            delete tc;
            continue;
          }
#endif

          {
          WriteLock l(&rwlock_);
          conns_[ti.fd()] = tc;
          }
          pink_epoll_->PinkAddEvent(ti.fd(), EPOLLIN);
        } else {
          continue;
        }
      } else {
        in_conn = NULL;
        int should_close = 0;
        if (pfe == NULL) {
          continue;
        }
        std::map<int, PinkConn *>::iterator iter = conns_.find(pfe->fd);
        if (iter == conns_.end()) {
          pink_epoll_->PinkDelEvent(pfe->fd);
          continue;
        }

        in_conn = iter->second;

        if (pfe->mask & EPOLLOUT && in_conn->is_reply()) {
          pink_epoll_->PinkModEvent(pfe->fd, 0, EPOLLIN);  // Remove EPOLLOUT
          WriteStatus write_status = in_conn->SendReply();
          in_conn->set_last_interaction(now);
          if (write_status == kWriteAll) {
            in_conn->set_is_reply(false);
          } else if (write_status == kWriteHalf) {
            pink_epoll_->PinkModEvent(pfe->fd, EPOLLIN, EPOLLOUT);
            continue; //  send all write buffer,
                      //  in case of next GetRequest()
                      //  pollute the write buffer
          } else if (write_status == kWriteError) {
            should_close = 1;
          }
        }

        if (!should_close && pfe->mask & EPOLLIN) {
          ReadStatus getRes = in_conn->GetRequest();
          in_conn->set_last_interaction(now);
          if (getRes != kReadAll && getRes != kReadHalf) {
            // kReadError kReadClose kFullError kParseError kDealError
            should_close = 1;
          } else if (in_conn->is_reply()) {
            WriteStatus write_status = in_conn->SendReply();
            if (write_status == kWriteAll) {
              in_conn->set_is_reply(false);
            } else if (write_status == kWriteHalf) {
              pink_epoll_->PinkModEvent(pfe->fd, EPOLLIN, EPOLLOUT);
            } else if (write_status == kWriteError) {
              should_close = 1;
            }
          } else {
            continue;
          }
        }
        if ((pfe->mask & EPOLLERR) || (pfe->mask & EPOLLHUP) || should_close) {
          {
            WriteLock l(&rwlock_);
            pink_epoll_->PinkDelEvent(pfe->fd);
            CloseFd(in_conn);
            delete(in_conn);
            in_conn = NULL;

            conns_.erase(pfe->fd);
          }
        }
      }  // connection event
    }  // for (int i = 0; i < nfds; i++)
  }  // while (!should_stop())

  Cleanup();
  return NULL;
}

void WorkerThread::DoCronTask() {
  struct timeval now;
  gettimeofday(&now, NULL);
  WriteLock l(&rwlock_);

  // Check whether close all connection
  MutexLock kl(&killer_mutex_);
  if (deleting_conn_ipport_.count(kKillAllConnsTask)) {
    for (auto& conn : conns_) {
      CloseFd(conn.second);
      delete conn.second;
    }
    conns_.clear();
    deleting_conn_ipport_.clear();
    return;
  }

  std::map<int, PinkConn*>::iterator iter = conns_.begin();
  while (iter != conns_.end()) {
    // Check connection should be closed
    if (deleting_conn_ipport_.count(iter->second->ip_port())) {
      CloseFd(iter->second);
      deleting_conn_ipport_.erase(iter->second->ip_port());
      delete iter->second;
      iter = conns_.erase(iter);
      continue;
    }

    // Check keepalive timeout connection
    if (keepalive_timeout_ > 0 &&
        (now.tv_sec - iter->second->last_interaction().tv_sec >
         keepalive_timeout_)) {
      CloseFd(iter->second);
      server_thread_->handle_->FdTimeoutHandle(
          iter->first, iter->second->ip_port());
      delete iter->second;
      iter = conns_.erase(iter);
      continue;
    }
    ++iter;
  }
}

bool WorkerThread::TryKillConn(const std::string& ip_port) {
  bool find = false;
  if (ip_port != kKillAllConnsTask) {
    ReadLock l(&rwlock_);
    for (auto& iter : conns_) {
      if (iter.second->ip_port() == ip_port) {
        find = true;
        break;
      }
    }
  }
  if (find || ip_port == kKillAllConnsTask) {
    MutexLock l(&killer_mutex_);
    deleting_conn_ipport_.insert(ip_port);
    return true;
  }
  return false;
}

void WorkerThread::CloseFd(PinkConn* conn) {
  close(conn->fd());
  server_thread_->handle_->FdClosedHandle(conn->fd(), conn->ip_port());
}

void WorkerThread::Cleanup() {
  WriteLock l(&rwlock_);
  for (auto& iter : conns_) {
    CloseFd(iter.second);
    delete iter.second;
  }
  conns_.clear();
}

} // namespace pink
} // namespace bubblefs 