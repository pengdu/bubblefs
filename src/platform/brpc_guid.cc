// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/guid_posix.cc

#include "platform/brpc_guid.h"
#include "utils/brpc_randuint64.h"
#include "utils/stringprintf.h"

namespace bubblefs {
namespace mybrpc {
namespace random {  
  
std::string GenerateGUID() {
  uint64_t sixteen_bytes[2] = { RandUint64(), RandUint64() };
  return RandomDataToGUIDString(sixteen_bytes);
}

bool IsValidGUID(const std::string& guid) {
  const size_t kGUIDLength = 36U;
  if (guid.length() != kGUIDLength)
    return false;

  const std::string hexchars = "0123456789ABCDEF";
  for (uint32_t i = 0; i < guid.length(); ++i) {
    char current = guid[i];
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (current != '-')
        return false;
    } else {
      if (hexchars.find(current) == std::string::npos)
        return false;
    }
  }

  return true;
}  

// TODO(cmasone): Once we're comfortable this works, migrate Windows code to
// use this as well.
std::string RandomDataToGUIDString(const uint64_t bytes[2]) {
  return strings::Printf("%08X-%04X-%04X-%04X-%012llX",
                         static_cast<unsigned int>(bytes[0] >> 32),
                         static_cast<unsigned int>((bytes[0] >> 16) & 0x0000ffff),
                         static_cast<unsigned int>(bytes[0] & 0x0000ffff),
                         static_cast<unsigned int>(bytes[1] >> 48),
                         bytes[1] & 0x0000ffffffffffffULL);
}

}  // namespace random
}  // namespace mybrpc
}  // namespace bubblefs