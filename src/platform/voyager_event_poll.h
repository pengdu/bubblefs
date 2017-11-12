// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/event_poll.h

#ifndef BUBBLEFS_PLATFORM_VOYAGER_EVENT_POLL_H_
#define BUBBLEFS_PLATFORM_VOYAGER_EVENT_POLL_H_

#include <vector>
#include "platform/voyager_event_poller.h"

namespace bubblefs {
namespace myvoyager {

class EventPoll : public EventPoller {
 public:
  explicit EventPoll(EventLoop* ev);
  virtual ~EventPoll();

  virtual void Poll(int timeout, std::vector<Dispatch*>* dispatches);
  virtual void RemoveDispatch(Dispatch* dispatch);
  virtual void UpdateDispatch(Dispatch* dispatch);

 private:
  std::vector<struct pollfd> pollfds_;
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_VOYAGER_EVENT_POLL_H_