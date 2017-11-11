// Copyright (c) 2014 The IndexFS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// indexfs/common/scanner.h

#ifndef BUBBLEFS_UTILS_PDLFS_PROP_SCANNER_H_
#define BUBBLEFS_UTILS_PDLFS_PROP_SCANNER_H_

#include <string>
#include <fstream>

namespace bubblefs {
namespace mypdlfs {  

class PropScanner {

 public:

  virtual ~PropScanner();

  PropScanner(const char* file_name) {
    fs_.open(file_name, std::ifstream::in);
  }

  PropScanner(const std::string &file_name) {
    fs_.open(file_name.c_str(), std::ifstream::in);
  }

  bool IsOpen() {
    return fs_.is_open();
  }

  bool HasNextLine() {
    return fs_.is_open() && fs_.good();
  }

  bool NextKeyValue(std::string* key, std::string* value);

  bool NextServerAddress(std::string* ip, std::string* port);

 private:

  std::string buf_;
  std::ifstream fs_;

  // No copy allowed
  PropScanner(const PropScanner&);
  PropScanner& operator=(const PropScanner&);
};

} // namespace mypdlfs
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PDLFS_PROP_SCANNER_H_