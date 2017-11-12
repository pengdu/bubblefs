// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/core/bg_eventloop.h

#ifndef BUBBLEFS_UTILS_VOYAGER_BG_EVENTLOOP_H_
#define BUBBLEFS_UTILS_VOYAGER_BG_EVENTLOOP_H_

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include "utils/voyager_eventloop.h"

namespace bubblefs {
namespace myvoyager {

class BGEventLoop {
 public:
  explicit BGEventLoop(PollType type = kEpoll);
  ~BGEventLoop();

  EventLoop* Loop();

 private:
  void ThreadFunc();

  PollType type_;
  EventLoop* eventloop_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::unique_ptr<std::thread> thread_;

  // No copying allowed
  BGEventLoop(const BGEventLoop&);
  void operator=(const BGEventLoop&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_BG_EVENTLOOP_H_