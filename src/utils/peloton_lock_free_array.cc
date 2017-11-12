//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// lock_free_lock_free_array.cpp
//
// Identification: src/container/lock_free_lock_free_array.cpp
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/container/lock_free_array.cpp

#include "utils/peloton_lock_free_array.h"
#include <assert.h>
#include <memory>

namespace bubblefs {
namespace mypeloton {

template <typename ValueType>
LockFreeArray<ValueType>::LockFreeArray(){
  lock_free_array.reset(new lock_free_array_t());
}

template <typename ValueType>
LockFreeArray<ValueType>::~LockFreeArray(){
}

template <typename ValueType>
bool LockFreeArray<ValueType>::Update(const std::size_t &offset, ValueType value){
  assert(offset <= LOCK_FREE_ARRAY_MAX_SIZE);
  //LOG_TRACE("Update at %lu", lock_free_array_offset.load());
  lock_free_array->at(offset) =  value;
  return true;
}

template <typename ValueType>
bool LockFreeArray<ValueType>::Append(ValueType value){
  //LOG_TRACE("Appended at %lu", lock_free_array_offset.load());
  lock_free_array->at(lock_free_array_offset++) = value;
  return true;
}

template <typename ValueType>
bool LockFreeArray<ValueType>::Erase(const std::size_t &offset, const ValueType& invalid_value){
  assert(offset <= LOCK_FREE_ARRAY_MAX_SIZE);
  //LOG_TRACE("Erase at %lu", offset);
  lock_free_array->at(offset) =  invalid_value;
  return true;
}

template <typename ValueType>
ValueType LockFreeArray<ValueType>::Find(const std::size_t &offset) const{
  assert(offset <= LOCK_FREE_ARRAY_MAX_SIZE);
  //LOG_TRACE("Find at %lu", offset);
  auto value = lock_free_array->at(offset);
  return value;
}

template <typename ValueType>
ValueType LockFreeArray<ValueType>::FindValid(const std::size_t &offset,
                                          const ValueType& invalid_value) const {
  assert(offset <= LOCK_FREE_ARRAY_MAX_SIZE);
  //LOG_TRACE("Find Valid at %lu", offset);

  std::size_t valid_array_itr = 0;
  std::size_t array_itr;

  for(array_itr = 0;
      array_itr < lock_free_array_offset;
      array_itr++){
    auto value = lock_free_array->at(array_itr);
    if (value != invalid_value) {
      // Check offset
      if(valid_array_itr == offset) {
        return value;
      }

      // Update valid value count
      valid_array_itr++;
    }
  }

  return invalid_value;
}

template <typename ValueType>
size_t LockFreeArray<ValueType>::GetSize() const{
  return lock_free_array_offset;
}

template <typename ValueType>
bool LockFreeArray<ValueType>::IsEmpty() const{
  return lock_free_array->empty();
}

template <typename ValueType>
void LockFreeArray<ValueType>::Clear(const ValueType& invalid_value) {

  // Set invalid value for all elements and reset lock_free_array_offset
  for(std::size_t array_itr = 0;
      array_itr < lock_free_array_offset;
      array_itr++){
    lock_free_array->at(array_itr) = invalid_value;
  }

  // Reset sentinel
  lock_free_array_offset = 0;

}

template <typename ValueType>
bool LockFreeArray<ValueType>::Contains(const ValueType& value) {

  bool exists = false;

  for(std::size_t array_itr = 0;
      array_itr < lock_free_array_offset;
      array_itr++){
    auto array_value = lock_free_array->at(array_itr);
    // Check array value
    if(array_value == value) {
      exists = true;
      break;
    }
  }

  return exists;
}

}  // namespace mypeloton
}  // namespace bubblefs