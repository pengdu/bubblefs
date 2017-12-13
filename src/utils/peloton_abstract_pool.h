//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// abstract_pool.h
//
// Identification: src/include/type/abstract_pool.h
//
// Copyright (c) 2015-17, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/include/type/abstract_pool.h

#ifndef BUBBLEFS_UTILS_PELOTON_ABSTRACT_POOL_H_
#define BUBBLEFS_UTILS_PELOTON_ABSTRACT_POOL_H_

#include <cstdlib>

namespace bubblefs {
namespace mypeloton {
namespace type {

// Interface of a memory pool that can quickly allocate chunks of memory
class AbstractPool {
public:

  // Empty virtual destructor for proper cleanup
  virtual ~AbstractPool(){}

  // Allocate a contiguous block of memory of the given size. If the allocation
  // is successful a non-null pointer is returned. If the allocation fails, a
  // null pointer will be returned.
  // TODO: Provide good error codes for failure cases.
  virtual void *Allocate(size_t size) = 0;

  // Returns the provided chunk of memory back into the pool
  virtual void Free(void *ptr) = 0;

};

}  // namespace type
}  // namespace mypeloton
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_ABSTRACT_POOL_H_