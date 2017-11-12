//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// printable.h
//
// Identification: src/include/common/printable.h
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/include/common/printable.h

#ifndef BUBBLEFS_UTILS_PELOTON_PRINTABLE_H
#define BUBBLEFS_UTILS_PELOTON_PRINTABLE_H

#include <iosfwd>
#include <string>

namespace bubblefs {
namespace mypeloton {
//===--------------------------------------------------------------------===//
// Printable Object
//===--------------------------------------------------------------------===//

class Printable {
 public:
  virtual ~Printable() { };

  /** @brief Get the info about the object. */
  virtual const std::string GetInfo() const = 0;

  // Get a string representation for debugging
  friend std::ostream &operator<<(std::ostream &os, const Printable &printable);
};

}  // namespace mypeloton
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_PRINTABLE_H