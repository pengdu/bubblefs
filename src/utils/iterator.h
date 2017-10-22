/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tensorflow/tensorflow/core/lib/io/iterator.h

// An iterator yields a sequence of key/value pairs from a source.
// The following class defines the interface.  Multiple implementations
// are provided by this library.  In particular, iterators are provided
// to access the contents of a Table or a DB.
//
// Multiple threads can invoke const methods on an Iterator without
// external synchronization, but if any of the threads may call a
// non-const method, all threads accessing the same Iterator must use
// external synchronization.

#ifndef BUBBLEFS_UTILS_ITERATOR_H_
#define BUBBLEFS_UTILS_ITERATOR_H_

#include "utils/status.h"
#include "utils/stringpiece.h"

namespace bubblefs {
  
//===--------------------------------------------------------------------===//
// Iterator Interface
//===--------------------------------------------------------------------===//
// peloton/src/include/common/iterator.h
template <class T>
class Iterator {
 public:
  virtual bool Next(T &out) = 0;

  virtual bool HasNext() = 0;

  virtual ~Iterator() {}
};  

namespace table {

class Iterator {
 public:
  Iterator();
  virtual ~Iterator();

  // An iterator is either positioned at a key/value pair, or
  // not valid.  This method returns true iff the iterator is valid.
  virtual bool Valid() const = 0;

  // Position at the first key in the source.  The iterator is Valid()
  // after this call iff the source is not empty.
  virtual void SeekToFirst() = 0;

  // Position at the first key in the source that is at or past target.
  // The iterator is Valid() after this call iff the source contains
  // an entry that comes at or past target.
  virtual void Seek(const StringPiece& target) = 0;

  // Moves to the next entry in the source.  After this call, Valid() is
  // true iff the iterator was not positioned at the last entry in the source.
  // REQUIRES: Valid()
  virtual void Next() = 0;

  // Return the key for the current entry.  The underlying storage for
  // the returned slice is valid only until the next modification of
  // the iterator.
  // REQUIRES: Valid()
  virtual StringPiece key() const = 0;

  // Return the value for the current entry.  The underlying storage for
  // the returned slice is valid only until the next modification of
  // the iterator.
  // REQUIRES: Valid()
  virtual StringPiece value() const = 0;

  // If an error has occurred, return it.  Else return an ok status.
  virtual Status status() const = 0;

  // Clients are allowed to register function/arg1/arg2 triples that
  // will be invoked when this iterator is destroyed.
  //
  // Note that unlike all of the preceding methods, this method is
  // not abstract and therefore clients should not override it.
  typedef void (*CleanupFunction)(void* arg1, void* arg2);
  void RegisterCleanup(CleanupFunction function, void* arg1, void* arg2);

 private:
  struct Cleanup {
    CleanupFunction function;
    void* arg1;
    void* arg2;
    Cleanup* next;
  };
  Cleanup cleanup_;

  // No copying allowed
  DISALLOW_COPY_AND_ASSIGN(Iterator);
};

// Return an empty iterator (yields nothing).
extern Iterator* NewEmptyIterator();

// Return an empty iterator with the specified status.
extern Iterator* NewErrorIterator(const Status& status);

}  // namespace table

}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_ITERATOR_H_