//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

// rocksdb/util/kv_map.h

#ifndef BUBBLEFS_UTILS_KV_MAP_H_
#define BUBBLEFS_UTILS_KV_MAP_H_

#include <map>
#include <string>
#include "utils/comparator.h"
#include "utils/coding.h"
#include "utils/murmurhash.h"
#include "utils/stringpiece.h"

namespace bubblefs {
namespace gtl {

struct LessOfComparator {
  explicit LessOfComparator(const Comparator* c = BytewiseComparator())
      : cmp(c) {}

  bool operator()(const std::string& a, const std::string& b) const {
    return cmp->Compare(StringPiece(a), StringPiece(b)) < 0;
  }
  bool operator()(const StringPiece& a, const StringPiece& b) const {
    return cmp->Compare(a, b) < 0;
  }

  const Comparator* cmp;
};

typedef std::map<std::string, std::string, LessOfComparator> KVMap;

} // namespace gtl
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_KV_MAP_H_