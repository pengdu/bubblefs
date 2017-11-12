// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/src/pink_item.h

#ifndef BUBBLEFS_UTILS_PINK_ITEM_H_
#define BUBBLEFS_UTILS_PINK_ITEM_H_

#include <string>
#include "utils/pink_define.h"

namespace bubblefs {
namespace mypink {

class PinkItem {
 public:
  PinkItem() { }
  PinkItem(const int fd, const std::string &ip_port)
      : fd_(fd),
        ip_port_(ip_port) {
  }

  int fd() const {
    return fd_;
  }
  std::string ip_port() const {
    return ip_port_;
  }

 private:
  int fd_;
  std::string ip_port_;
};

}  // namespace mypink
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PINK_ITEM_H_