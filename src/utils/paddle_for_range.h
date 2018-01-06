/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Paddle/paddle/platform/for_range.h
// Paddle/paddle/platform/transform.h

#pragma once

#include <algorithm>
#include <type_traits>
#include "utils/paddle_device_context.h"

#ifdef __NVCC__
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include "paddle/platform/details/device_ptr_cast.h"
#endif

namespace bubblefs {
namespace mypaddle {
namespace platform {

template <typename DeviceContext>
struct ForRange {
  ForRange(const DeviceContext& dev_ctx, size_t limit);

  template <typename Function>
  void operator()(Function func) const;
};

template <>
struct ForRange<CPUDeviceContext> {
  ForRange(const CPUDeviceContext& dev_ctx, size_t limit) : limit_(limit) {}

  template <typename Function>
  void operator()(Function func) const {
    for (size_t i = 0; i < limit_; ++i) {
      func(i);
    }
  }

  size_t limit_;
};

#ifdef __NVCC__
template <typename Function>
__global__ static void ForRangeElemwiseOpGridIsOne(Function func) {
  size_t idx = static_cast<size_t>(threadIdx.x);
  func(idx);
}

template <typename Function>
__global__ static void ForRangeElemwiseOp(Function func, int limit) {
  size_t idx = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < limit) {
    func(idx);
  }
}

template <>
struct ForRange<CUDADeviceContext> {
  ForRange(const CUDADeviceContext& dev_ctx, size_t limit)
      : dev_ctx_(dev_ctx), limit_(static_cast<int>(limit)) {}

  template <typename Function>
  inline void operator()(Function func) const {
    constexpr int num_threads = 1024;
    int block_size = limit_ <= num_threads ? limit_ : num_threads;
    int grid_size = (limit_ + num_threads - 1) / num_threads;

    if (grid_size == 1) {
      ForRangeElemwiseOpGridIsOne<<<1, block_size, 0, dev_ctx_.stream()>>>(
          func);
    } else {
      ForRangeElemwiseOp<<<grid_size, block_size, 0, dev_ctx_.stream()>>>(
          func, limit_);
    }
  }

  const CUDADeviceContext& dev_ctx_;
  int limit_;
};

#endif


// Transform on host or device. It provides the same API in std library.
template <typename DeviceContext>
struct Transform {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const DeviceContext& context, InputIter first, InputIter last,
                  OutputIter result, UnaryOperation op);

  template <typename InputIter1, typename InputIter2, typename OutputIter,
            typename BinaryOperation>
  void operator()(const DeviceContext& context, InputIter1 first1,
                  InputIter1 last1, InputIter2 first2, OutputIter result,
                  BinaryOperation op);
};

template <>
struct Transform<platform::CPUDeviceContext> {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const platform::CPUDeviceContext& context, InputIter first,
                  InputIter last, OutputIter result, UnaryOperation op) {
    std::transform(first, last, result, op);
  }

  template <typename InputIter1, typename InputIter2, typename OutputIter,
            typename BinaryOperation>
  void operator()(const platform::CPUDeviceContext& context, InputIter1 first1,
                  InputIter1 last1, InputIter2 first2, OutputIter result,
                  BinaryOperation op) {
    std::transform(first1, last1, first2, result, op);
  }
};

#ifdef __NVCC__
template <>
struct Transform<platform::CUDADeviceContext> {
  template <typename InputIter, typename OutputIter, typename UnaryOperation>
  void operator()(const platform::CUDADeviceContext& context, InputIter first,
                  InputIter last, OutputIter result, UnaryOperation op) {
    auto place = context.GetPlace();
    PADDLE_ENFORCE(is_gpu_place(place), "It must use GPU place.");
    thrust::transform(thrust::cuda::par.on(context.stream()),
                      details::DevPtrCast(first), details::DevPtrCast(last),
                      details::DevPtrCast(result), op);
  }

  template <typename InputIter1, typename InputIter2, typename OutputIter,
            typename BinaryOperation>
  void operator()(const platform::CUDADeviceContext& context, InputIter1 first1,
                  InputIter1 last1, InputIter2 first2, OutputIter result,
                  BinaryOperation op) {
    auto place = context.GetPlace();
    PADDLE_ENFORCE(is_gpu_place(place), "It must use GPU place.");
    thrust::transform(thrust::cuda::par.on(context.stream()),
                      details::DevPtrCast(first1), details::DevPtrCast(last1),
                      details::DevPtrCast(first2), details::DevPtrCast(result),
                      op);
  }
};
#endif

}  // namespace platform
}  // namespace mypaddle
}  // namespace bubblefs