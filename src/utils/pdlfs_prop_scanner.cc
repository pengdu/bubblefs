// Copyright (c) 2014 The IndexFS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// indexfs/common/scanner.cc

#include "utils/pdlfs_prop_scanner.h"

namespace bubblefs {
namespace mypdlfs {

PropScanner::~PropScanner() {
  if (fs_.is_open()) {
    fs_.close();
  }
}

bool PropScanner::NextKeyValue(std::string* key, std::string* value) {
  size_t idx;
  std::getline(fs_, buf_);
  if ((idx = buf_.find('=')) == std::string::npos) {
    return false;
  }
  key->assign(buf_.substr(0, idx));
  value->assign(buf_.substr(idx + 1));
  return true;
}

bool PropScanner::NextServerAddress(std::string* ip, std::string* port) {
  size_t idx;
  std::getline(fs_, buf_);
  if (buf_.empty()) {
    return false;
  }
  if ((idx = buf_.find(':')) == std::string::npos) {
    ip->assign(buf_);
    port->clear();
  } else {
    ip->assign(buf_.substr(0, idx));
    port->assign(buf_.substr(idx + 1));
  }
  return true;
}

} // namespace mypdlfs
} // namespace bubblefs