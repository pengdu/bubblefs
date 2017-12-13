//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// iterator.h
//
// Identification: src/include/common/iterator.h
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/include/common/iterator.h

#ifndef BUBBLEFS_UTILS_PELOTON_ITERATOR_H_
#define BUBBLEFS_UTILS_PELOTON_ITERATOR_H_

namespace bubblefs {
namespace mypeloton {

//===--------------------------------------------------------------------===//
// Iterator Interface
//===--------------------------------------------------------------------===//

template <class T>
class Iterator {
 public:
  virtual bool Next(T &out) = 0;

  virtual bool HasNext() = 0;

  virtual ~Iterator() {}
};

}  // namespace mypeloton
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_ITERATOR_H_