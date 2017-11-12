// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/event_select.h

#ifndef BUBBLEFS_PLATFORM_VOYAGER_EVENT_SELECT_H_
#define BUBBLEFS_PLATFORM_VOYAGER_EVENT_SELECT_H_

#include <sys/select.h>
#include <sys/types.h>
#include <map>
#include <vector>
#include "platform/voyager_event_poller.h"

namespace bubblefs {
namespace myvoyager {

class EventSelect : public EventPoller {
 public:
  explicit EventSelect(EventLoop* ev);
  virtual ~EventSelect();
  virtual void Poll(int timeout, std::vector<Dispatch*>* dispatches);
  virtual void RemoveDispatch(Dispatch* dispatch);
  virtual void UpdateDispatch(Dispatch* dispatch);

 private:
  int nfds_;
  fd_set readfds_;
  fd_set writefds_;
  fd_set exceptfds_;
  std::map<int, Dispatch*> worker_map_;
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_VOYAGER_EVENT_SELECT_H_