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

// Paddle/paddle/platform/dynload/cudnn.h
// Paddle/paddle/platform/dynload/cudnn.cc
// Paddle/paddle/platform/dynload/cublas.h
// Paddle/paddle/platform/cudnn_helper.h

#pragma once

#include <vector>
#include <dlfcn.h>
#include <mutex>

#include <cudnn.h>
#include <cublas_v2.h>

namespace paddle {
namespace platform {

extern std::once_flag cudnn_dso_flag;
extern void* cudnn_dso_handle;
extern bool HasCUDNN();

#ifdef PADDLE_USE_DSO

extern void EnforceCUDNNLoaded(const char* fn_name);
#define DECLARE_DYNAMIC_LOAD_CUDNN_WRAP(__name)                    \
  struct DynLoad__##__name {                                       \
    template <typename... Args>                                    \
    auto operator()(Args... args) -> decltype(__name(args...)) {   \
      using cudnn_func = decltype(__name(args...)) (*)(Args...);   \
      std::call_once(cudnn_dso_flag,                               \
                     paddle::platform::dynload::GetCudnnDsoHandle, \
                     &cudnn_dso_handle);                           \
      EnforceCUDNNLoaded(#__name);                                 \
      void* p_##__name = dlsym(cudnn_dso_handle, #__name);         \
      return reinterpret_cast<cudnn_func>(p_##__name)(args...);    \
    }                                                              \
  };                                                               \
  extern struct DynLoad__##__name __name

#else

#define DECLARE_DYNAMIC_LOAD_CUDNN_WRAP(__name)                  \
  struct DynLoad__##__name {                                     \
    template <typename... Args>                                  \
    auto operator()(Args... args) -> decltype(__name(args...)) { \
      return __name(args...);                                    \
    }                                                            \
  };                                                             \
  extern DynLoad__##__name __name

#endif

/**
 * include all needed cudnn functions in HPPL
 * different cudnn version has different interfaces
 **/
#define CUDNN_DNN_ROUTINE_EACH(__macro)             \
  __macro(cudnnSetTensor4dDescriptor);              \
  __macro(cudnnSetTensor4dDescriptorEx);            \
  __macro(cudnnSetTensorNdDescriptor);              \
  __macro(cudnnGetTensorNdDescriptor);              \
  __macro(cudnnGetConvolutionNdForwardOutputDim);   \
  __macro(cudnnGetConvolutionForwardAlgorithm);     \
  __macro(cudnnCreateTensorDescriptor);             \
  __macro(cudnnDestroyTensorDescriptor);            \
  __macro(cudnnCreateFilterDescriptor);             \
  __macro(cudnnSetFilter4dDescriptor);              \
  __macro(cudnnSetFilterNdDescriptor);              \
  __macro(cudnnGetFilterNdDescriptor);              \
  __macro(cudnnSetPooling2dDescriptor);             \
  __macro(cudnnSetPoolingNdDescriptor);             \
  __macro(cudnnGetPoolingNdDescriptor);             \
  __macro(cudnnDestroyFilterDescriptor);            \
  __macro(cudnnCreateConvolutionDescriptor);        \
  __macro(cudnnCreatePoolingDescriptor);            \
  __macro(cudnnDestroyPoolingDescriptor);           \
  __macro(cudnnSetConvolution2dDescriptor);         \
  __macro(cudnnDestroyConvolutionDescriptor);       \
  __macro(cudnnSetConvolutionNdDescriptor);         \
  __macro(cudnnGetConvolutionNdDescriptor);         \
  __macro(cudnnDeriveBNTensorDescriptor);           \
  __macro(cudnnCreate);                             \
  __macro(cudnnDestroy);                            \
  __macro(cudnnSetStream);                          \
  __macro(cudnnActivationForward);                  \
  __macro(cudnnConvolutionForward);                 \
  __macro(cudnnConvolutionBackwardBias);            \
  __macro(cudnnGetConvolutionForwardWorkspaceSize); \
  __macro(cudnnTransformTensor);                    \
  __macro(cudnnPoolingForward);                     \
  __macro(cudnnPoolingBackward);                    \
  __macro(cudnnSoftmaxBackward);                    \
  __macro(cudnnSoftmaxForward);                     \
  __macro(cudnnGetVersion);                         \
  __macro(cudnnGetErrorString);
CUDNN_DNN_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)

#define CUDNN_DNN_ROUTINE_EACH_R2(__macro) \
  __macro(cudnnAddTensor);                 \
  __macro(cudnnConvolutionBackwardData);   \
  __macro(cudnnConvolutionBackwardFilter);
CUDNN_DNN_ROUTINE_EACH_R2(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)

// APIs available after R3:
#if CUDNN_VERSION >= 3000
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R3(__macro)           \
  __macro(cudnnGetConvolutionBackwardFilterWorkspaceSize); \
  __macro(cudnnGetConvolutionBackwardDataAlgorithm);       \
  __macro(cudnnGetConvolutionBackwardFilterAlgorithm);     \
  __macro(cudnnGetConvolutionBackwardDataWorkspaceSize);
CUDNN_DNN_ROUTINE_EACH_AFTER_R3(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

// APIs available after R4:
#if CUDNN_VERSION >= 4007
#define CUDNN_DNN_ROUTINE_EACH_AFTER_R4(__macro)    \
  __macro(cudnnBatchNormalizationForwardTraining);  \
  __macro(cudnnBatchNormalizationForwardInference); \
  __macro(cudnnBatchNormalizationBackward);
CUDNN_DNN_ROUTINE_EACH_AFTER_R4(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

// APIs in R5
#if CUDNN_VERSION >= 5000
#define CUDNN_DNN_ROUTINE_EACH_R5(__macro)  \
  __macro(cudnnCreateActivationDescriptor); \
  __macro(cudnnSetActivationDescriptor);    \
  __macro(cudnnGetActivationDescriptor);    \
  __macro(cudnnDestroyActivationDescriptor);
CUDNN_DNN_ROUTINE_EACH_R5(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif

#if CUDNN_VERSION >= 7001
#define CUDNN_DNN_ROUTINE_EACH_R7(__macro) \
  __macro(cudnnSetConvolutionGroupCount);
CUDNN_DNN_ROUTINE_EACH_R7(DECLARE_DYNAMIC_LOAD_CUDNN_WRAP)
#endif  

extern std::once_flag cublas_dso_flag;
extern void *cublas_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#ifdef PADDLE_USE_DSO
#define DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP(__name)                    \
  struct DynLoad__##__name {                                        \
    template <typename... Args>                                     \
    inline cublasStatus_t operator()(Args... args) {                \
      typedef cublasStatus_t (*cublasFunc)(Args...);                \
      std::call_once(cublas_dso_flag,                               \
                     paddle::platform::dynload::GetCublasDsoHandle, \
                     &cublas_dso_handle);                           \
      void *p_##__name = dlsym(cublas_dso_handle, #__name);         \
      return reinterpret_cast<cublasFunc>(p_##__name)(args...);     \
    }                                                               \
  };                                                                \
  extern DynLoad__##__name __name
#else
#define DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP(__name)     \
  struct DynLoad__##__name {                         \
    template <typename... Args>                      \
    inline cublasStatus_t operator()(Args... args) { \
      return __name(args...);                        \
    }                                                \
  };                                                 \
  extern DynLoad__##__name __name
#endif

#define DECLARE_DYNAMIC_LOAD_CUBLAS_V2_WRAP(__name) \
  DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP(__name)

#define CUBLAS_BLAS_ROUTINE_EACH(__macro) \
  __macro(cublasSaxpy_v2);                \
  __macro(cublasDaxpy_v2);                \
  __macro(cublasSgemv_v2);                \
  __macro(cublasDgemv_v2);                \
  __macro(cublasSgemm_v2);                \
  __macro(cublasDgemm_v2);                \
  __macro(cublasSgeam_v2);                \
  __macro(cublasDgeam_v2);                \
  __macro(cublasCreate_v2);               \
  __macro(cublasDestroy_v2);              \
  __macro(cublasSetStream_v2);            \
  __macro(cublasSetPointerMode_v2);       \
  __macro(cublasGetPointerMode_v2);       \
  __macro(cublasSgemmBatched);            \
  __macro(cublasDgemmBatched);            \
  __macro(cublasCgemmBatched);            \
  __macro(cublasZgemmBatched);            \
  __macro(cublasSgemmStridedBatched);     \
  __macro(cublasDgemmStridedBatched);     \
  __macro(cublasCgemmStridedBatched);     \
  __macro(cublasZgemmStridedBatched);     \
  __macro(cublasSgetrfBatched);           \
  __macro(cublasSgetriBatched);           \
  __macro(cublasDgetrfBatched);           \
  __macro(cublasDgetriBatched)

CUBLAS_BLAS_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP);

#undef DECLARE_DYNAMIC_LOAD_CUBLAS_WRAP
  
inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
    default:
      return "Unknown cudnn error number";
  }
}

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= ((major)*1000 + (minor)*100 + (patch)))

#define CUDNN_ENFORCE(condition)                                  \
  do {                                                            \
    cudnnStatus_t status = condition;                             \
    if (status != CUDNN_STATUS_SUCCESS) {                         \
      VLOG(1) << ::paddle::platform::cudnnGetErrorString(status); \
      PADDLE_THROW("cuDNN call failed");                          \
    }                                                             \
  } while (false)

enum class DataLayout {  // Not use
  kNHWC,
  kNCHW,
  kNCDHW,
  kNCHW_VECT_C,
};

enum class PoolingMode {
  kMaximum,
  kAverage,
};

template <typename T>
class CudnnDataType;

template <>
class CudnnDataType<float> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  typedef const float ScalingParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

template <>
class CudnnDataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  typedef const double ScalingParamType;
  static ScalingParamType* kOne() {
    static ScalingParamType v = 1.0;
    return &v;
  }
  static ScalingParamType* kZero() {
    static ScalingParamType v = 0.0;
    return &v;
  }
};

inline cudnnTensorFormat_t GetCudnnTensorFormat(
    const DataLayout& order) {  // Not use
  switch (order) {
    case DataLayout::kNHWC:
      return CUDNN_TENSOR_NHWC;
    case DataLayout::kNCHW:
      return CUDNN_TENSOR_NCHW;
    case DataLayout::kNCDHW:
      return CUDNN_TENSOR_NCHW;  // NOTE: cudnn treat NdTensor as the same
    default:
      PADDLE_THROW("Unknown cudnn equivalent for order");
  }
  return CUDNN_TENSOR_NCHW;
}

class ScopedTensorDescriptor {
 public:
  ScopedTensorDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnCreateTensorDescriptor(&desc_));
  }
  ~ScopedTensorDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnDestroyTensorDescriptor(desc_));
  }

  inline cudnnTensorDescriptor_t descriptor(const cudnnTensorFormat_t format,
                                            const cudnnDataType_t type,
                                            const std::vector<int>& dims,
                                            const int groups = 1) {
    // the format is not used now, will add later
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    // Update tensor descriptor dims setting if groups > 1
    // NOTE: Assume using NCHW or NCDHW order
    std::vector<int> dims_with_group(dims.begin(), dims.end());  // copy
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }
    PADDLE_ENFORCE(dynload::cudnnSetTensorNdDescriptor(
        desc_, type, dims_with_group.size(), dims_with_group.data(),
        strides.data()));
    return desc_;
  }

  template <typename T>
  inline cudnnTensorDescriptor_t descriptor(const DataLayout& order,
                                            const std::vector<int>& dims,
                                            const int groups = 1) {
    return descriptor(GetCudnnTensorFormat(order), CudnnDataType<T>::type, dims,
                      groups);
  }

 private:
  cudnnTensorDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedTensorDescriptor);
};

class ScopedFilterDescriptor {
 public:
  ScopedFilterDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnCreateFilterDescriptor(&desc_));
  }
  ~ScopedFilterDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnDestroyFilterDescriptor(desc_));
  }

  inline cudnnFilterDescriptor_t descriptor(const cudnnTensorFormat_t format,
                                            const cudnnDataType_t type,
                                            const std::vector<int>& kernel,
                                            const int groups = 1) {
    // filter layout: MCHW(MCDHW), where M is the number of
    // output image channels, C is the number of input image channels,
    // D is the depth of the filter, H is the height of the filter, and W is the
    // width of the filter.
    std::vector<int> kernel_with_group(kernel.begin(), kernel.end());
    if (groups > 1) {
      kernel_with_group[0] /= groups;
      // NOTE: input filter(C) of the filter is already asserted to be C/groups.
    }
    PADDLE_ENFORCE(dynload::cudnnSetFilterNdDescriptor(
        desc_, type, format, kernel_with_group.size(),
        kernel_with_group.data()));
    return desc_;
  }

  template <typename T>
  inline cudnnFilterDescriptor_t descriptor(const DataLayout& order,
                                            const std::vector<int>& kernel,
                                            const int groups = 1) {
    return descriptor(GetCudnnTensorFormat(order), CudnnDataType<T>::type,
                      kernel, groups);
  }

 private:
  cudnnFilterDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedFilterDescriptor);
};

class ScopedConvolutionDescriptor {
 public:
  ScopedConvolutionDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnCreateConvolutionDescriptor(&desc_));
  }
  ~ScopedConvolutionDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnDestroyConvolutionDescriptor(desc_));
  }

  inline cudnnConvolutionDescriptor_t descriptor(
      cudnnDataType_t type, const std::vector<int>& pads,
      const std::vector<int>& strides, const std::vector<int>& dilations) {
    PADDLE_ENFORCE_EQ(pads.size(), strides.size());
    PADDLE_ENFORCE_EQ(pads.size(), dilations.size());

#if !CUDNN_VERSION_MIN(6, 0, 0)
    // cudnn v5 does not support dilation conv, the argument is called upscale
    // instead of dilations and it is must be one.
    for (size_t i = 0; i < dilations.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          dilations[i], 1,
          "Dilations conv is not supported in this cuDNN version(%d.%d.%d).",
          CUDNN_VERSION / 1000, CUDNN_VERSION % 1000 / 100,
          CUDNN_VERSION % 100);
    }
#endif

    PADDLE_ENFORCE(dynload::cudnnSetConvolutionNdDescriptor(
        desc_, pads.size(), pads.data(), strides.data(), dilations.data(),
        CUDNN_CROSS_CORRELATION, type));
    return desc_;
  }

  template <typename T>
  inline cudnnConvolutionDescriptor_t descriptor(
      const std::vector<int>& pads, const std::vector<int>& strides,
      const std::vector<int>& dilations) {
    return descriptor(CudnnDataType<T>::type, pads, strides, dilations);
  }

 private:
  cudnnConvolutionDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedConvolutionDescriptor);
};

class ScopedPoolingDescriptor {
 public:
  ScopedPoolingDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnCreatePoolingDescriptor(&desc_));
  }
  ~ScopedPoolingDescriptor() {
    PADDLE_ENFORCE(dynload::cudnnDestroyPoolingDescriptor(desc_));
  }

  inline cudnnPoolingDescriptor_t descriptor(const PoolingMode& mode,
                                             const std::vector<int>& kernel,
                                             const std::vector<int>& pads,
                                             const std::vector<int>& strides) {
    PADDLE_ENFORCE_EQ(kernel.size(), pads.size());
    PADDLE_ENFORCE_EQ(kernel.size(), strides.size());
    PADDLE_ENFORCE(dynload::cudnnSetPoolingNdDescriptor(
        desc_, (mode == PoolingMode::kMaximum
                    ? CUDNN_POOLING_MAX
                    : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING),
        CUDNN_PROPAGATE_NAN,  // Always propagate nans.
        kernel.size(), kernel.data(), pads.data(), strides.data()));
    return desc_;
  }

 private:
  cudnnPoolingDescriptor_t desc_;
  DISABLE_COPY_AND_ASSIGN(ScopedPoolingDescriptor);
};

}  // namespace platform
}  // namespace paddle
