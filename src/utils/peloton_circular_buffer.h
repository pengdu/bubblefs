//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// circular_buffer.h
//
// Identification: src/include/container/circular_buffer.h
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/include/container/circular_buffer.h

#ifndef BUBBLEFS_UTILS_PELOTON_CIRCULAR_BUFFER_H_
#define BUBBLEFS_UTILS_PELOTON_CIRCULAR_BUFFER_H_

#include "boost/circular_buffer.hpp"

namespace bubblefs {
namespace mypeloton {

template <typename ValueType>
class CircularBuffer {
 public:
  typedef boost::circular_buffer<ValueType> circular_buffer_t;

  // Push a new item
  void PushBack(ValueType value);

  // Set the container capaciry
  void SetCapaciry(size_t new_capacity);

  // Returns item count in the circular_buffer
  size_t GetSize() const;

  // Checks if the circular_buffer is empty
  bool IsEmpty() const;

  // Clear all elements
  void Clear();

  // Begin iterator
  typename circular_buffer_t::iterator begin() {
    return circular_buffer_.begin();
  }

  // End iterator
  typename circular_buffer_t::iterator end() { return circular_buffer_.end(); }

 private:
  // circular buffer type
  circular_buffer_t circular_buffer_;
};

}  // namespace mypeloton
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_CIRCULAR_BUFFER_H_