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

// Paddle/paddle/platform/hostdevice.h
// Paddle/paddle/platform/cuda_helper.h
// Paddle/paddle/platform/cuda_profiler.h

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda.h>
#include <cuda_profiler_api.h>

namespace bubblefs {
namespace mypaddle {
namespace platform {

#ifdef __CUDACC__
#define HOSTDEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOSTDEVICE
#define DEVICE
#define HOST
#endif

#define CUDA_ATOMIC_WRAPPER(op, T) \
  __device__ __forceinline__ T CudaAtomic##op(T* address, const T val)

#define USE_CUDA_ATOMIC(op, T) \
  CUDA_ATOMIC_WRAPPER(op, T) { return atomic##op(address, val); }

// Default thread count per block(or block size).
// TODO(typhoonzero): need to benchmark against setting this value
//                    to 1024.
constexpr int PADDLE_CUDA_NUM_THREADS = 512;

// For atomicAdd.
USE_CUDA_ATOMIC(Add, float);
USE_CUDA_ATOMIC(Add, int);
USE_CUDA_ATOMIC(Add, unsigned int);
USE_CUDA_ATOMIC(Add, unsigned long long int);

CUDA_ATOMIC_WRAPPER(Add, int64_t) {
  static_assert(sizeof(int64_t) == sizeof(long long int),
                "long long should be int64");
  return CudaAtomicAdd(reinterpret_cast<unsigned long long int*>(address),
                       static_cast<unsigned long long int>(val));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
USE_CUDA_ATOMIC(Add, double);
#else
CUDA_ATOMIC_WRAPPER(Add, double) {
  unsigned long long int* address_as_ull =
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

void CudaProfilerInit(std::string output_file, std::string output_mode,
                      std::string config_file) {
  PADDLE_ENFORCE(output_mode == "kvp" || output_mode == "csv");
  cudaOutputMode_t mode = output_mode == "csv" ? cudaCSV : cudaKeyValuePair;
  PADDLE_ENFORCE(
      cudaProfilerInitialize(config_file.c_str(), output_file.c_str(), mode));
}

void CudaProfilerStart() { PADDLE_ENFORCE(cudaProfilerStart()); }

void CudaProfilerStop() { PADDLE_ENFORCE(cudaProfilerStop()); }

}  // namespace platform
}  // namespace mpaddle
}  // namespace bubblefs