// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/dispatch.h

#ifndef BUBBLEFS_UTILS_VOYAGER_DISPATCH_H_
#define BUBBLEFS_UTILS_VOYAGER_DISPATCH_H_

#include <sys/poll.h>
#include <assert.h>
#include <functional>
#include <memory>
#include <utility>

namespace bubblefs {
namespace voyager {

class EventLoop;

class Dispatch {
 public:
  typedef std::function<void()> EventCallback;

  enum ModifyEvent {
    kNoModify = 0,
    kAddRead = 1,
    kAddWrite = 2,
    kDeleteRead = 3,
    kDeleteWrite = 4,
    kEnableWrite = 5,
    kDisableWrite = 6,
    kDeleteAll = 7,
  };

  Dispatch(EventLoop* eventloop, int fd);
  ~Dispatch();

  int Fd() const { return fd_; }
  int Events() const { return events_; }
  void SetRevents(int rv) { revents_ = rv; }
  void SetIndex(int index) { index_ = index; }
  int Index() const { return index_; }

  void HandleEvent();

  void SetReadCallback(const EventCallback& cb) { read_cb_ = cb; }
  void SetWriteCallback(const EventCallback& cb) { write_cb_ = cb; }
  void SetCloseCallback(const EventCallback& cb) { close_cb_ = cb; }
  void SetErrorCallback(const EventCallback& cb) { error_cb_ = cb; }

  void SetReadCallback(EventCallback&& cb) { read_cb_ = std::move(cb); }
  void SetWriteCallback(EventCallback&& cb) { write_cb_ = std::move(cb); }
  void SetCloseCallback(EventCallback&& cb) { close_cb_ = std::move(cb); }
  void SetErrorCallback(EventCallback&& cb) { error_cb_ = std::move(cb); }

  void EnableRead();
  void EnableWrite();

  void DisableRead();
  void DisableWrite();
  void DisableAll();

  bool IsNoneEvent() const { return events_ == kNoneEvent; }
  bool IsReading() const { return events_ & kReadEvent; }
  bool IsWriting() const { return events_ & kWriteEvent; }

  void RemoveEvents();

  EventLoop* OwnerEventLoop() const { return eventloop_; }

  void Tie(const std::shared_ptr<void>& obj);

  int Modify() const { return modify_; }

 private:
  void UpdateEvents();
  void HandleEventWithGuard();

  static const int kNoneEvent = 0;
  static const int kReadEvent = POLLIN | POLLPRI;
  static const int kWriteEvent = POLLOUT;

  EventLoop* eventloop_;
  const int fd_;
  int events_;
  int revents_;
  int index_;
  ModifyEvent modify_;
  bool add_write_;

  std::weak_ptr<void> tie_;
  bool tied_;
  bool event_handling_;

  EventCallback read_cb_;
  EventCallback write_cb_;
  EventCallback close_cb_;
  EventCallback error_cb_;

  // No copying allowed
  Dispatch(const Dispatch&);
  void operator=(const Dispatch&);
};

}  // namespace voyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_DISPATCH_H_