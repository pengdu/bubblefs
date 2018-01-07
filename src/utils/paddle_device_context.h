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

// Paddle/paddle/platform/place.h
// Paddle/paddle/platform/device_context.h

#pragma once

#include <memory>
#include <unordered_map>
#include <iostream>

namespace bubblefs {
namespace mypaddle {
namespace platform {

struct CPUPlace {
  // WORKAROUND: for some reason, omitting this constructor
  // causes errors with boost 1.59 and OSX
  CPUPlace() {}

  // needed for variant equality comparison
  inline bool operator==(const CPUPlace &) const { return true; }
  inline bool operator!=(const CPUPlace &) const { return false; }
};

struct CUDAPlace {
  CUDAPlace() : CUDAPlace(0) {}
  explicit CUDAPlace(int d) : device(d) {}

  inline int GetDeviceId() const { return device; }
  // needed for variant equality comparison
  inline bool operator==(const CUDAPlace &o) const {
    return device == o.device;
  }
  inline bool operator!=(const CUDAPlace &o) const { return !(*this == o); }

  int device;
};

struct IsCUDAPlace : public boost::static_visitor<bool> {
  bool operator()(const CPUPlace &) const { return false; }
  bool operator()(const CUDAPlace &gpu) const { return true; }
};

typedef boost::variant<CUDAPlace, CPUPlace> Place;

void set_place(const Place &);
const Place &get_place();

const CUDAPlace default_gpu();
const CPUPlace default_cpu();

bool is_gpu_place(const Place &);
bool is_cpu_place(const Place &);
bool places_are_same_class(const Place &, const Place &);

std::ostream &operator<<(std::ostream &, const Place &);

template <typename Visitor>
struct PlaceVisitorWrapper
    : public boost::static_visitor<typename Visitor::result_type> {
  const Visitor &visitor_;
  explicit PlaceVisitorWrapper(const Visitor &visitor) : visitor_(visitor) {}

  typename Visitor::result_type operator()(const CPUPlace &cpu) const {
    return visitor_(cpu);
  }

  typename Visitor::result_type operator()(const CUDAPlace &cuda) const {
#ifdef PADDLE_WITH_CUDA
    return visitor_(cuda);
#else
    PADDLE_THROW("Paddle is not compiled with CUDA. Cannot visit cuda device");
    return typename Visitor::result_type();
#endif
  }
};

template <typename Visitor>
typename Visitor::result_type VisitPlace(const Place &place,
                                         const Visitor &visitor) {
  return boost::apply_visitor(PlaceVisitorWrapper<Visitor>(visitor), place);
}

class DeviceContext {
 public:
  virtual ~DeviceContext() {}
  virtual Place GetPlace() const = 0;

  virtual void Wait() const {}
};

class CPUDeviceContext : public DeviceContext {
 public:
  CPUDeviceContext();
  explicit CPUDeviceContext(CPUPlace place);

  Eigen::DefaultDevice* eigen_device() const;

  Place GetPlace() const override;

 private:
  CPUPlace place_;
  std::unique_ptr<Eigen::DefaultDevice> eigen_device_;
};

template <typename Place>
struct DefaultDeviceContextType;

template <>
struct DefaultDeviceContextType<platform::CPUPlace> {
  using TYPE = CPUDeviceContext;
};

#ifdef PADDLE_WITH_CUDA

class EigenCudaStreamDevice;

class CUDADeviceContext : public DeviceContext {
 public:
  explicit CUDADeviceContext(CUDAPlace place);
  virtual ~CUDADeviceContext();

  /*! \brief  Wait for all operations completion in the stream. */
  void Wait() const override;

  /*! \brief  Return place in the device context. */
  Place GetPlace() const override;

  /*! \brief  Return eigen device in the device context. */
  Eigen::GpuDevice* eigen_device() const;

  /*! \brief  Return cublas handle in the device context. */
  cublasHandle_t cublas_handle() const;

  /*! \brief  Return cudnn  handle in the device context. */
  cudnnHandle_t cudnn_handle() const;

  /*! \brief  Return cuda stream in the device context. */
  cudaStream_t stream() const;

 private:
  CUDAPlace place_;

  std::unique_ptr<Eigen::GpuDevice> eigen_device_;
  std::unique_ptr<EigenCudaStreamDevice> eigen_stream_;

  cudaStream_t stream_;
  cudnnHandle_t cudnn_handle_;
  cublasHandle_t cublas_handle_;
};

template <>
struct DefaultDeviceContextType<platform::CUDAPlace> {
  using TYPE = CUDADeviceContext;
};

#endif

#ifdef PADDLE_WITH_MKLDNN
class MKLDNNDeviceContext : public CPUDeviceContext {
 public:
  explicit MKLDNNDeviceContext(CPUPlace place);

  /* \brief  Add new element: memory, primitive or primitive desc */
  template <typename T>
  void AddElement(const std::string& op_key, const T& value);

  /* \brief  Get existed element: memory, primitive or primitive desc */
  template <typename T>
  const T& GetElement(const std::string& op_key) const;

  /* \brief  Get element pool: memory, primitive or primitive desc pool */
  template <typename T>
  const std::unordered_map<const std::string, const T, std::hash<std::string>>&
  GetElementPool() const;

  /* \brief  Get the active engine */
  const MKLDNNEngine& engine() const { return *engine_; }

  /* \brief  Submit primitive to pipeline */
  void Submit(const MKLDNNPrimitivePtr& p) { pipeline_.push_back(*p); }

  /*! \brief  Execute all submitted primitives in pipeline */
  void Execute(bool block = true);

 protected:
  /*! \brief  Reset the stream to prepare next exectue */
  void ResetStream();

 private:
  std::unordered_map<const std::string, const MKLDNNMemoryPtr,
                     std::hash<std::string>>
      memory_pool_;
  std::unordered_map<const std::string, const MKLDNNPrimitivePtr,
                     std::hash<std::string>>
      primitive_pool_;
  std::unordered_map<const std::string, const MKLDNNPrimitiveDescPtr,
                     std::hash<std::string>>
      primitive_desc_pool_;
  std::vector<MKLDNNPrimitive> pipeline_;
  MKLDNNStreamPtr stream_;
  MKLDNNEnginePtr engine_;
  bool ready_;
};
#endif

/*! \brief device context pool singleton */
class DeviceContextPool {
 public:
  explicit DeviceContextPool(const std::vector<platform::Place>& places);

  static DeviceContextPool& Instance() {
    PADDLE_ENFORCE_NOT_NULL(pool, "Need to Create DeviceContextPool first!");
    return *pool;
  }

  /*! \brief  Create should only called by Init function */
  static DeviceContextPool& Init(const std::vector<platform::Place>& places) {
    if (pool == nullptr) {
      pool = new DeviceContextPool(places);
    }
    return *pool;
  }

  /*! \brief  Return handle of single device context. */
  const platform::DeviceContext* Get(const platform::Place& place);

  template <typename Place>
  const typename DefaultDeviceContextType<Place>::TYPE* GetByPlace(
      const Place& place) {
    return reinterpret_cast<
        const typename DefaultDeviceContextType<Place>::TYPE*>(Get(place));
  }

 private:
  static DeviceContextPool* pool;
  constexpr static int LEFT_SHIFT = 8;
  struct Hash {
    std::hash<int> hash_;
    size_t operator()(const platform::Place& place) const {
      int pre_hash = place.which() << LEFT_SHIFT;
      if (platform::is_gpu_place(place)) {
        pre_hash += boost::get<platform::CUDAPlace>(place).GetDeviceId();
      }
      return hash_(pre_hash);
    }
  };
  std::unordered_map<const platform::Place, const platform::DeviceContext*,
                     Hash>
      device_contexts_;
  DISABLE_COPY_AND_ASSIGN(DeviceContextPool);
};


DeviceContextPool* DeviceContextPool::pool = nullptr;

const platform::DeviceContext* DeviceContextPool::Get(
    const platform::Place& place) {
  auto it = device_contexts_.find(place);
  if (it == device_contexts_.end()) {
    PADDLE_THROW(
        "'Place' is not supported, Please re-compile with WITH_GPU "
        "option");
  }
  return it->second;
}

DeviceContextPool::DeviceContextPool(
    const std::vector<platform::Place>& places) {
  PADDLE_ENFORCE_GT(places.size(), 0);
  for (size_t i = 0; i < places.size(); i++) {
    if (platform::is_cpu_place(places[i])) {
      device_contexts_.emplace(places[i],
                               new platform::CPUDeviceContext(
                                   boost::get<platform::CPUPlace>(places[i])));
    } else if (platform::is_gpu_place(places[i])) {
#ifdef PADDLE_WITH_CUDA
      device_contexts_.emplace(places[i],
                               new platform::CUDADeviceContext(
                                   boost::get<platform::CUDAPlace>(places[i])));
#else
      PADDLE_THROW(
          "'CUDAPlace' is not supported, Please re-compile with WITH_GPU "
          "option");
#endif
    }
  }
}

CPUDeviceContext::CPUDeviceContext() {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

CPUDeviceContext::CPUDeviceContext(CPUPlace place) : place_(place) {
  eigen_device_.reset(new Eigen::DefaultDevice());
}

Eigen::DefaultDevice* CPUDeviceContext::eigen_device() const {
  return eigen_device_.get();
}

Place CPUDeviceContext::GetPlace() const { return place_; }

#ifdef PADDLE_WITH_CUDA

class EigenCudaStreamDevice : public Eigen::StreamInterface {
 public:
  EigenCudaStreamDevice() : scratch_(nullptr), semaphore_(nullptr) {
    Eigen::initializeDeviceProp();
  }
  ~EigenCudaStreamDevice() override {}

  void Reinitialize(const cudaStream_t* cuda_stream, CUDAPlace place) {
    stream_ = cuda_stream;
    place_ = place;
    device_prop_ = &Eigen::m_deviceProperties[place.device];
  }

  const cudaStream_t& stream() const override { return *stream_; }

  const cudaDeviceProp& deviceProperties() const override {
    return *device_prop_;
  }

  void* allocate(size_t num_bytes) const override {
    return paddle::memory::Alloc(place_, num_bytes);
  }

  void deallocate(void* buffer) const override {
    paddle::memory::Free(place_, buffer);
  }

  void* scratchpad() const override {
    if (scratch_ == NULL) {
      scratch_ = allocate(Eigen::kCudaScratchSize + sizeof(unsigned int));
    }
    return scratch_;
  }

  unsigned int* semaphore() const override {
    if (semaphore_ == NULL) {
      char* scratch =
          static_cast<char*>(scratchpad()) + Eigen::kCudaScratchSize;
      semaphore_ = reinterpret_cast<unsigned int*>(scratch);
      PADDLE_ENFORCE(
          cudaMemsetAsync(semaphore_, 0, sizeof(unsigned int), *stream_));
    }
    return semaphore_;
  }

 private:
  CUDAPlace place_;
  const cudaStream_t* stream_;         // not owned;
  const cudaDeviceProp* device_prop_;  // not owned;
  mutable void* scratch_;
  mutable unsigned int* semaphore_;
};

CUDADeviceContext::CUDADeviceContext(CUDAPlace place) : place_(place) {
  SetDeviceId(place_.device);
  PADDLE_ENFORCE(cudaStreamCreate(&stream_));
  eigen_stream_.reset(new EigenCudaStreamDevice());
  eigen_stream_->Reinitialize(&stream_, place);
  eigen_device_.reset(new Eigen::GpuDevice(eigen_stream_.get()));
  PADDLE_ENFORCE(dynload::cublasCreate(&cublas_handle_));
  PADDLE_ENFORCE(dynload::cublasSetStream(cublas_handle_, stream_));
  if (dynload::HasCUDNN()) {
    PADDLE_ENFORCE(dynload::cudnnCreate(&cudnn_handle_));
    PADDLE_ENFORCE(dynload::cudnnSetStream(cudnn_handle_, stream_));
  } else {
    cudnn_handle_ = nullptr;
  }
}

CUDADeviceContext::~CUDADeviceContext() {
  SetDeviceId(place_.device);
  Wait();
  PADDLE_ENFORCE(dynload::cublasDestroy(cublas_handle_));
  if (cudnn_handle_ != nullptr) {
    PADDLE_ENFORCE(dynload::cudnnDestroy(cudnn_handle_));
  }
  eigen_stream_.reset();
  eigen_device_.reset();
  PADDLE_ENFORCE(cudaStreamDestroy(stream_));
}

Place CUDADeviceContext::GetPlace() const { return place_; }

void CUDADeviceContext::Wait() const {
  PADDLE_ENFORCE(cudaStreamSynchronize(stream_));
  PADDLE_ENFORCE(cudaGetLastError());
}

Eigen::GpuDevice* CUDADeviceContext::eigen_device() const {
  return eigen_device_.get();
}

cublasHandle_t CUDADeviceContext::cublas_handle() const {
  return cublas_handle_;
}

cudnnHandle_t CUDADeviceContext::cudnn_handle() const { return cudnn_handle_; }

cudaStream_t CUDADeviceContext::stream() const { return stream_; }

#endif

#ifdef PADDLE_WITH_MKLDNN
MKLDNNDeviceContext::MKLDNNDeviceContext(CPUPlace place)
    : CPUDeviceContext(place), ready_(false) {
  stream_.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
  engine_.reset(new mkldnn::engine(mkldnn::engine::cpu, 0));
}

template <typename T>
void MKLDNNDeviceContext::AddElement(const std::string& op_key,
                                     const T& value) {
  if (GetElement<T>(op_key)) {
    return;
  }
  GetElementPool<T>().emplace(op_key, std::move(value));
}

template <typename T>
const T& MKLDNNDeviceContext::GetElement(const std::string& op_key) const {
  auto it = GetElementPool<T>().find(op_key);
  return it == GetElementPool<T>().end() ? nullptr : it->second;
}

template <>
const std::unordered_map<const std::string, const MKLDNNMemoryPtr,
                         std::hash<std::string>>&
MKLDNNDeviceContext::GetElementPool<MKLDNNMemoryPtr>() const {
  return memory_pool_;
}

template <>
const std::unordered_map<const std::string, const MKLDNNPrimitivePtr,
                         std::hash<std::string>>&
MKLDNNDeviceContext::GetElementPool<MKLDNNPrimitivePtr>() const {
  return primitive_pool_;
}

template <>
const std::unordered_map<const std::string, const MKLDNNPrimitiveDescPtr,
                         std::hash<std::string>>&
MKLDNNDeviceContext::GetElementPool<MKLDNNPrimitiveDescPtr>() const {
  return primitive_desc_pool_;
}

void MKLDNNDeviceContext::Execute(bool block) {
  if (pipeline_.empty()) {
    return;
  }
  ResetStream();
  stream_->submit(pipeline_).wait(block);
  ready_ = false;
  pipeline_.clear();
}

void MKLDNNDeviceContext::ResetStream() {
  if (ready_) {
    return;
  }
  // TODO(TJ): change me when mkldnn have specific method to reset this state
  stream_.reset(new mkldnn::stream(mkldnn::stream::kind::eager));
  ready_ = true;
}

#endif

}  // namespace platform
}  // namespace mypaddle
}  // namespace bubblefs