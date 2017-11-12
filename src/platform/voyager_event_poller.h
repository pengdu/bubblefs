// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/event_poller.h

#ifndef BUBBLEFS_PLATFORM_VOYAGER_EVENT_POLLER_H_
#define BUBBLEFS_PLATFORM_VOYAGER_EVENT_POLLER_H_

#include <unordered_map>
#include <vector>
#include "utils/voyager_dispatch.h"
#include "utils/voyager_eventloop.h"

namespace bubblefs {
namespace myvoyager {

class EventPoller {
 public:
  explicit EventPoller(EventLoop* ev);
  virtual ~EventPoller();

  virtual void Poll(int timeout, std::vector<Dispatch*>* dispatches) = 0;
  virtual void RemoveDispatch(Dispatch* dispatch) = 0;
  virtual void UpdateDispatch(Dispatch* dispatch) = 0;
  virtual bool HasDispatch(Dispatch* dispatch) const;

 protected:
  typedef std::unordered_map<int, Dispatch*> DispatchMap;

  DispatchMap dispatch_map_;
  EventLoop* eventloop_;

  // No copying allowed
  EventPoller(const EventPoller&);
  void operator=(const EventPoller&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_VOYAGER_EVENT_POLLER_H_