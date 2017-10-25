// Copyright (c) 2017 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// conhash/conhash.h

#ifndef BUBBLEFS_UTILS_SIMPLE_CONHASH_H_
#define BUBBLEFS_UTILS_SIMPLE_CONHASH_H_

#include <stdint.h>
#include <map>
#include <string>

namespace bubblefs {
namespace simple {
  
class ConHashNode {
 public:
  std::string identify;
  int replicas;
  ConHashNode() : identify(), replicas(0) {}
  ConHashNode(const std::string& s, int i) : identify(s), replicas(i) {}
};

class ConHash {
 public:
  ConHash() {}

  void AddNode(const ConHashNode& node);
  void RemoveNode(const ConHashNode& node);
  bool Lookup(const std::string& object, ConHashNode* node);

 private:
  uint32_t Hash(const std::string& identify, int i);

  std::map<uint32_t, ConHashNode> vnodes_;

  // No copying allowed
  ConHash(const ConHash&);
  void operator=(const ConHash&);
};

} // namespace simple
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SIMPLE_CONHASH_H_