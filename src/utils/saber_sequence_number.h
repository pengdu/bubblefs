// Copyright (c) 2017 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// saber/saber/util/sequence_number.h

#ifndef BUBBLEFS_UTILS_SABER_SEQUENCE_NUMBER_H_
#define BUBBLEFS_UTILS_SABER_SEQUENCE_NUMBER_H_

#include "platform/mutexlock.h"

namespace bubblefs {
namespace mysaber {

template <typename T>
class SequenceNumber {
 public:
  SequenceNumber(T max) : max_(max), num_(0) {}
  T GetNext() {
    MutexLock lock(&mutex_);
    if (num_ >= max_) {
      num_ = 0;
    }
    return num_++;
  }

 private:
  port::Mutex mutex_;
  T max_;
  T num_;

  // No copying allowed
  SequenceNumber(const SequenceNumber&);
  void operator=(const SequenceNumber&);
};

}  // namespace mysaber
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SABER_SEQUENCE_NUMBER_H_