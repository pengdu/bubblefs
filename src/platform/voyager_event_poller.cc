// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/event_poller.cc

#include "platform/voyager_event_poller.h"

namespace bubblefs {
namespace myvoyager {

EventPoller::EventPoller(EventLoop* ev) : eventloop_(ev) {}

EventPoller::~EventPoller() {}

bool EventPoller::HasDispatch(Dispatch* dispatch) const {
  eventloop_->AssertInMyLoop();
  for (DispatchMap::const_iterator it = dispatch_map_.begin();
       it != dispatch_map_.end(); ++it) {
    if (it->second == dispatch) {
      return true;
    }
  }
  return false;
}

}  // namespace myvoyager
}  // namespace bubblefs