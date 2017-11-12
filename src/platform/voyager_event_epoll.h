// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/event_epoll.h

#ifndef BUBBLEFS_PLATFORM_VOYAGER_EVENT_EPOLL_H_
#define BUBBLEFS_PLATFORM_VOYAGER_EVENT_EPOLL_H_

#include <sys/epoll.h>
#include <vector>
#include "platform/voyager_event_poller.h"

namespace bubblefs {
namespace myvoyager {

class EventEpoll : public EventPoller {
 public:
  explicit EventEpoll(EventLoop* ev);
  virtual ~EventEpoll();

  virtual void Poll(int timeout, std::vector<Dispatch*>* dispatches);
  virtual void RemoveDispatch(Dispatch* dispatch);
  virtual void UpdateDispatch(Dispatch* dispatch);

 private:
  static const size_t kInitEpollFdSize = 16;

  void EpollCTL(int op, Dispatch* dispatch);

  int epollfd_;
  std::vector<struct epoll_event> epollfds_;
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_VOYAGER_EVENT_EPOLL_H_