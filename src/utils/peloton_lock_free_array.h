//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// lock_free_array.h
//
// Identification: src/include/container/lock_free_array.h
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#ifndef BUBBLEFS_UTILS_PELOTON_LOCK_FREE_ARRAY_H_
#define BUBBLEFS_UTILS_PELOTON_LOCK_FREE_ARRAY_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <array>
#include <atomic>
#include <memory>

namespace bubblefs {
namespace mypeloton {

constexpr size_t LOCK_FREE_ARRAY_MAX_SIZE = 1024 * 1024;

// LOCK_FREE_ARRAY_TEMPLATE_ARGUMENTS
#define LOCK_FREE_ARRAY_TEMPLATE_ARGUMENTS template <typename ValueType>

// LOCK_FREE_ARRAY_TYPE
#define LOCK_FREE_ARRAY_TYPE LockFreeArray<ValueType>

template <typename ValueType>
class LockFreeArray {
 public:

  LockFreeArray();
  ~LockFreeArray();

  // Update a item
  bool Update(const std::size_t &offset, ValueType value);

  // Append an item
  bool Append(ValueType value);

  // Get a item
  ValueType Find(const std::size_t &offset) const;

  // Get a valid item
  ValueType FindValid(const std::size_t &offset, const ValueType& invalid_value) const;

  // Delete key from the lock_free_array
  bool Erase(const std::size_t &offset, const ValueType& invalid_value);

  // Returns item count in the lock_free_array
  size_t GetSize() const;

  // Checks if the lock_free_array is empty
  bool IsEmpty() const;

  // Clear all elements and reset them to default value
  void Clear(const ValueType& invalid_value);

  // Exists ?
  bool Contains(const ValueType& value);

 private:

  // lock free array type
  typedef std::array<ValueType, LOCK_FREE_ARRAY_MAX_SIZE> lock_free_array_t;

  std::atomic<std::size_t> lock_free_array_offset {0};

  // lock free array
  std::unique_ptr<lock_free_array_t> lock_free_array;
};

}  // namespace mypeloton
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_LOCK_FREE_ARRAY_H_