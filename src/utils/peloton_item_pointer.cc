//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// item_pointer.cpp
//
// Identification: src/common/item_pointer.cpp
//
// Copyright (c) 2015-2017, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/common/item_pointer.cpp

#include "utils/peloton_item_pointer.h"
#include "platform/macros.h"

namespace bubblefs {
namespace mypeloton {

ItemPointer INVALID_ITEMPOINTER;

bool AtomicUpdateItemPointer(ItemPointer* src_ptr, const ItemPointer& value) {
  assert(sizeof(ItemPointer) == sizeof(int64_t));
  int64_t* cast_src_ptr = reinterpret_cast<int64_t*>((void*)src_ptr);
  int64_t* cast_value_ptr = reinterpret_cast<int64_t*>((void*)&value);
  return __sync_bool_compare_and_swap(cast_src_ptr, *cast_src_ptr,
                                      *cast_value_ptr);
}

}  // namespace mypeloton
}  // namespace bubblefs