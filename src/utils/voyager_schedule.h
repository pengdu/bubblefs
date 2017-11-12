// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef BUBBLEFS_UTILS_VOYAGER_SCHEDULE_H_
#define BUBBLEFS_UTILS_VOYAGER_SCHEDULE_H_

#include <memory>
#include <vector>
#include "utils/voyager_bg_eventloop.h"

namespace bubblefs {
namespace myvoyager {

class Schedule {
 public:
  Schedule(EventLoop* ev, int size);

  void Start();

  EventLoop* AssignLoop();

  bool Started() const { return started_; }

  const std::vector<EventLoop*>* AllLoops() const;

 private:
  EventLoop* baseloop_;
  size_t size_;
  bool started_;
  std::vector<EventLoop*> loops_;
  std::vector<std::unique_ptr<BGEventLoop> > bg_loops_;

  // No copying alloweded
  Schedule(const Schedule&);
  void operator=(const Schedule&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_SCHEDULE_H_