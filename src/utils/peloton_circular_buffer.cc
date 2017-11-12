//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// circular_buffer.cpp
//
// Identification: src/container/circular_buffer.cpp
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/container/circular_buffer.cpp

#include "utils/peloton_circular_buffer.h"

namespace bubblefs {
namespace mypeloton {

// Push a new item
template <typename ValueType>
void CircularBuffer<ValueType>::PushBack(ValueType value) {
  circular_buffer_.push_back(value);
}

// Set the container capaciry
template <typename ValueType>
void CircularBuffer<ValueType>::SetCapaciry(size_t new_capacity) {
  circular_buffer_.set_capacity(new_capacity);
}

// Returns item count in the circular_buffer
template <typename ValueType>
size_t CircularBuffer<ValueType>::GetSize() const { return circular_buffer_.size(); }

// Checks if the circular_buffer is empty
template <typename ValueType>
bool CircularBuffer<ValueType>::IsEmpty() const { return circular_buffer_.empty(); }

// Clear all elements
template <typename ValueType>
void CircularBuffer<ValueType>::Clear() { circular_buffer_.clear(); }

// Explicit template instantiation
template class CircularBuffer<double>;

}  // namespace mypeloton
}  // namespace bubblefs