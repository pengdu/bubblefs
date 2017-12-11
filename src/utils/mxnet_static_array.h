/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// incubator-mxnet/src/common/static_array.h

/*!
 * \file static_array.h
 */

#ifndef BUBBLEFS_UTILS_MXNET_STATIC_ARRAY_H_
#define BUBBLEFS_UTILS_MXNET_STATIC_ARRAY_H_

#include "platform/macros.h"

namespace bubblefs {
namespace mymxnet {
namespace common {

/*! \brief
 * Static array. This code is borrowed from struct Shape<ndim>,
 * except that users can specify the type of the elements of
 * the statically allocated array.
 * The object instance of the struct is copyable between CPU and GPU.
 * \tparam T element type of the array, must be copyable between CPU and GPU
 * \tparam num number of elements in the array
 */
template<typename T, int num>
struct StaticArray {
  static const int kNum = num;

  T array_[kNum];

  /*! \brief default constructor, do nothing */
  INLINE StaticArray(void) {}

  /*! \brief constructor, fill in the array with the input value */
  INLINE StaticArray(const T& val) {
    #pragma unroll
    for (int i = 0; i < num; ++i) {
      this->array_[i] = val;
    }
  }

  /*! \brief constuctor */
  INLINE StaticArray(const StaticArray<T, num>& sa) {
    #pragma unroll
    for (int i = 0; i < num; ++i) {
      this->array_[i] = sa[i];
    }
  }

  INLINE T& operator[](const index_t idx) {
    return array_[idx];
  }

  INLINE const T& operator[](const index_t idx) const {
    return array_[idx];
  }
};  // StaticArray

}  // namespace common
}  // namespace mymxnet
}  // namespace bubblefs 

#endif  // BUBBLEFS_UTILS_MXNET_STATIC_ARRAY_H_