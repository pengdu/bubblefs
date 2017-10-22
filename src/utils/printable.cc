//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// printable.cpp
//
// Identification: src/common/printable.cpp
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/common/printable.cpp

#include "utils/printable.h"
#include <sstream>

namespace bubblefs {

// Get a string representation for debugging
std::ostream &operator<<(std::ostream &os, const Printable &printable) {
  os << printable.GetInfo();
  return os;
};

}  // namespace bubblefs