// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/schedule.cc

#include "utils/voyager_schedule.h"
#include <assert.h>
#include <utility>

namespace bubblefs {
namespace myvoyager {

Schedule::Schedule(EventLoop* ev, int size)
    : baseloop_(ev), size_(size), started_(false) {}

void Schedule::Start() {
  assert(!started_);
  started_ = true;
  for (size_t i = 0; i < size_; ++i) {
    BGEventLoop* loop = new BGEventLoop(baseloop_->GetPollType());
    loops_.push_back(loop->Loop());
    bg_loops_.push_back(std::unique_ptr<BGEventLoop>(loop));
  }
  if (size_ == 0) {
    loops_.push_back(baseloop_);
  }
}

const std::vector<EventLoop*>* Schedule::AllLoops() const {
  assert(started_);
  return &loops_;
}

EventLoop* Schedule::AssignLoop() {
  baseloop_->AssertInMyLoop();
  assert(started_);
  assert(!loops_.empty());
  EventLoop* loop = loops_[0];
  int min = loop->ConnectionSize();

  for (size_t i = 1; i < loops_.size(); ++i) {
    int temp = loops_[i]->ConnectionSize();
    if (temp < min) {
      min = temp;
      loop = loops_[i];
    }
  }

  assert(loop != nullptr);
  return loop;
}

}  // namespace myvoyager
}  // namespace bubblefs