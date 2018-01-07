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

// Paddle/paddle/framework/data_layout.h
// Paddle/paddle/framework/tensor_impl.h
// Paddle/paddle/framework/lod_tensor.h

#pragma once

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iterator>
#include <memory>
#include <typeindex>
#include <vector>

#include "utils/paddle_ddim.h"
#include "utils/paddle_memory.h"
#include "utils/paddle_device_context.h"

#ifdef PADDLE_WITH_CUDA
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#endif

namespace bubblefs {
namespace mypaddle {
namespace framework {

class LoDTensor;

enum class DataLayout {
  kNHWC = 0,
  kNCHW = 1,
  kAnyLayout = 2,
};

class Tensor {
 public:
  template <typename T, size_t D, int MajorType, typename IndexType>
  friend struct EigenTensor;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenMatrix;

  template <typename T, int MajorType, typename IndexType>
  friend struct EigenVector;

 public:
  Tensor() : offset_(0) {}

  /*! Return a pointer to mutable memory block. */
  template <typename T>
  inline T* data();

  /*! Return a pointer to constant memory block. */
  template <typename T>
  inline const T* data() const;

  inline void switch_place(platform::Place new_place);

  /**
   * @brief   Return a pointer to mutable memory block.
   * @note    If not exist, then allocation.
   */
  template <typename T>
  inline T* mutable_data(platform::Place place);

  inline void* mutable_data(platform::Place place, std::type_index type);

  inline void* mutable_data(platform::Place place);

  /**
   * @brief     Return a pointer to mutable memory block.
   *
   * @param[in] dims    The dimensions of the memory block.
   * @param[in] place   The place of the memory block.
   *
   * @note      If not exist, then allocation.
   */
  template <typename T>
  inline T* mutable_data(DDim dims, platform::Place place);

  /*! Return the dimensions of the memory block. */
  inline const DDim& dims() const;

  /*! Return the numel of the memory block. */
  inline int64_t numel() const;

  /*! Resize the dimensions of the memory block. */
  inline Tensor& Resize(const DDim& dims);

  /*! The internal of two tensors share the same memory block. */
  inline Tensor& ShareDataWith(const Tensor& src);

  /**
   * @brief  Return a sub-tensor of the given tensor.
   *
   * @param[in] begin_idx   The index of the start row(inclusive) to slice.
   *                        The index number begins from 0.
   * @param[in] end_idx     The index of the end row(exclusive) to slice.
   *                        The index number begins from 0.
   */
  inline Tensor Slice(int begin_idx, int end_idx) const;

  platform::Place place() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_, "Tensor not initialized yet when Tensor::place() is called.");
    return holder_->place();
  }

  std::type_index type() const {
    PADDLE_ENFORCE_NOT_NULL(
        holder_, "Tensor not initialized yet when Tensor::type() is called.");
    return holder_->type();
  }

  size_t memory_size() const;

  inline void check_memory_size() const;

  inline DataLayout layout() const { return layout_; }

  inline void set_layout(const DataLayout layout) { layout_ = layout; }

 private:
  friend class LoDTensor;

  /**
   * @note    Placeholder hides type T, so it doesn't appear as a template
   *          parameter of Variable.
   */
  struct Placeholder {
    virtual ~Placeholder() = default;
    virtual void* ptr() const = 0;
    virtual size_t size() const = 0;
    virtual std::type_index type() const = 0;
    virtual platform::Place place() const = 0;
    virtual void set_type(std::type_index type) = 0;
  };

  template <typename Place>
  struct PlaceholderImpl : public Placeholder {
    PlaceholderImpl(Place place, size_t size, std::type_index type)
        : ptr_(static_cast<uint8_t*>(memory::Alloc(place, size)),
               memory::PODDeleter<uint8_t, Place>(place)),
          place_(place),
          size_(size),
          type_(type) {
      PADDLE_ENFORCE_NOT_NULL(ptr_, "Insufficient %s memory to allocation.",
                              (is_cpu_place(place_) ? "CPU" : "GPU"));
    }

    virtual size_t size() const { return size_; }
    virtual platform::Place place() const { return place_; }
    virtual void* ptr() const { return static_cast<void*>(ptr_.get()); }
    virtual std::type_index type() const { return type_; }
    virtual void set_type(std::type_index type) { type_ = type; }

    /*! the pointer of memory block. */
    std::unique_ptr<uint8_t, memory::PODDeleter<uint8_t, Place>> ptr_;

    /*! the place of memory block. */
    platform::Place place_;

    /*! the size of memory block. */
    size_t size_;

    /* the current type of memory */
    std::type_index type_;
  };

  /*! holds the memory block if allocated. */
  std::shared_ptr<Placeholder> holder_;

  /**
   * @brief points to elements dimensions.
   *
   * @note dims_ do not indicate the memory block size.
   */

  DDim dims_;

  /**
   * @brief the layout of memory block, default is NHWC.
   *
   * @note the memory allocation order, describe how weight/data is stored
   *       For example, in 4-D Tensor(rank=4), there are three commonly
   *       used layout. They are
   *            NCHW, NHWC, CHWN.
   *       N,C,H,W for respectively the batch size, the number of
   *       feature maps, the height.
   */

  DataLayout layout_ = DataLayout::kNHWC;

  /**
   * @brief   A PlaceHolder may be shared by more than one tensor.
   *
   * @note    Some of them may be slices of the others. So the offset_
   *          is introduced here to indicate the byte offset between
   *          PlaceHolder::ptr_ and where the tensor data really begins.
   */
  size_t offset_;
};

inline void Tensor::switch_place(platform::Place new_place) {
  if (holder_->place() == new_place) {
    return;
  }

  // TODO(tonyyang-svail): do memcpy here.
  PADDLE_THROW("Not Implemented");
}

template <typename... T>
struct SizeOfTypeFunctor;

template <typename T>
struct SizeOfTypeFunctor<T> {
  size_t operator()(std::type_index type) const {
    if (typeid(T).hash_code() == type.hash_code()) {
      return sizeof(T);
    } else {
      return 0UL;
    }
  }
};

template <>
struct SizeOfTypeFunctor<> {
  size_t operator()(std::type_index type) const { return 0UL; }
};

template <typename HEAD, typename... TAIL>
struct SizeOfTypeFunctor<HEAD, TAIL...> {
  size_t operator()(std::type_index type) const {
    SizeOfTypeFunctor<HEAD> head;
    size_t head_size = head(type);
    if (head_size != 0) {
      return head_size;
    }
    SizeOfTypeFunctor<TAIL...> tail;
    return tail(type);
  }
};

static inline size_t SizeOfType(std::type_index type) {
  SizeOfTypeFunctor<int, float, double, int16_t, int64_t, bool> functor;
  size_t size = functor(type);
  PADDLE_ENFORCE(size != 0UL, "Cannot get size of type %s", type.name());
  return size;
}

inline void Tensor::check_memory_size() const {
  PADDLE_ENFORCE_NOT_NULL(
      holder_, "Tensor holds no memory. Call Tensor::mutable_data first.");
  PADDLE_ENFORCE_GE(
      holder_->size(), memory_size() + offset_,
      "Tensor's dims_ is out of bound. Call Tensor::mutable_data "
      "first to re-allocate memory.\n"
      "or maybe the required data-type mismatches the data already stored.");
}

inline size_t Tensor::memory_size() const {
  return holder_ == nullptr ? 0UL : numel() * SizeOfType(type());
}

template <typename T>
inline const T* Tensor::data() const {
  check_memory_size();
  PADDLE_ENFORCE(std::is_same<T, void>::value ||
                     holder_->type().hash_code() == typeid(T).hash_code(),
                 "Tensor holds the wrong type, it holds %s",
                 this->holder_->type().name());

  return reinterpret_cast<const T*>(
      reinterpret_cast<uintptr_t>(holder_->ptr()) + offset_);
}

template <typename T>
inline T* Tensor::data() {
  check_memory_size();
  PADDLE_ENFORCE(std::is_same<T, void>::value ||
                     holder_->type().hash_code() == typeid(T).hash_code(),
                 "Tensor holds the wrong type, it holds %s",
                 this->holder_->type().name());
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                              offset_);
}

template <typename T>
inline T* Tensor::mutable_data(DDim dims, platform::Place place) {
  static_assert(std::is_pod<T>::value, "T must be POD");
  Resize(dims);
  return mutable_data<T>(place);
}

template <typename T>
inline T* Tensor::mutable_data(platform::Place place) {
  static_assert(std::is_pod<T>::value, "T must be POD");
  return reinterpret_cast<T*>(mutable_data(place, typeid(T)));
}

inline void* Tensor::mutable_data(platform::Place place, std::type_index type) {
  if (holder_ != nullptr) {
    holder_->set_type(type);
  }
  PADDLE_ENFORCE_GT(
      numel(), 0,
      "When calling this method, the Tensor's numel must be larger than zero. "
      "Please check Tensor::Resize has been called first.");
  int64_t size = numel() * SizeOfType(type);
  /* some versions of boost::variant don't have operator!= */
  if (holder_ == nullptr || !(holder_->place() == place) ||
      holder_->size() < size + offset_) {
    if (platform::is_cpu_place(place)) {
      holder_.reset(new PlaceholderImpl<platform::CPUPlace>(
          boost::get<platform::CPUPlace>(place), size, type));
    } else if (platform::is_gpu_place(place)) {
#ifndef PADDLE_WITH_CUDA
      PADDLE_THROW("'CUDAPlace' is not supported in CPU only device.");
    }
#else
      holder_.reset(new PlaceholderImpl<platform::CUDAPlace>(
          boost::get<platform::CUDAPlace>(place), size, type));
    }
#endif
    offset_ = 0;
  }
  return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(holder_->ptr()) +
                                 offset_);
}

inline void* Tensor::mutable_data(platform::Place place) {
  PADDLE_ENFORCE(this->holder_ != nullptr,
                 "Cannot invoke mutable data if current hold nothing");
  return mutable_data(place, holder_->type());
}

inline Tensor& Tensor::ShareDataWith(const Tensor& src) {
  src.check_memory_size();
  *this = src;
  return *this;
}

inline Tensor Tensor::Slice(int begin_idx, int end_idx) const {
  check_memory_size();
  PADDLE_ENFORCE_GE(begin_idx, 0,
                    "The start row index must be greater than 0.");
  PADDLE_ENFORCE_LE(end_idx, dims_[0], "The end row index is out of bound.");
  PADDLE_ENFORCE_LT(
      begin_idx, end_idx,
      "The start row index must be lesser than the end row index.");

  if (dims_[0] == 1) {
    return *this;
  } else {
    size_t base = numel() / dims_[0];
    Tensor dst;
    dst.holder_ = holder_;
    dst.set_layout(layout_);
    DDim dst_dims = dims_;
    dst_dims[0] = end_idx - begin_idx;
    dst.Resize(dst_dims);
    dst.offset_ = offset_ + begin_idx * base * SizeOfType(type());
    return dst;
  }
}

inline Tensor& Tensor::Resize(const DDim& dims) {
  dims_ = dims;
  return *this;
}

inline const DDim& Tensor::dims() const { return dims_; }

inline int64_t Tensor::numel() const { return product(dims_); }

inline Tensor ReshapeToMatrix(const Tensor& src, int num_col_dims) {
  Tensor res;
  res.ShareDataWith(src);
  res.Resize(flatten_to_2d(src.dims(), num_col_dims));
  return res;
}

#ifndef PADDLE_WITH_CUDA
template <typename T>
using Vector = std::vector<T>;
#else
template <typename T>
using Vector = thrust::host_vector<
    T, thrust::system::cuda::experimental::pinned_allocator<T>>;
#endif

/*
 * LoD is short for Level of Details.
 *
 * - in a level, each element indicates relative offset of the lower level
 * - the first element should be 0 and that indicates that this sequence start
 * from 0
 * - each sequence's begin and end(no-inclusive) is level[id, id+1]
 *
 * For example:
 *    3-level LoD stores
 *
 *    0 2 3
 *    0 2 4 7
 *    0 2 5 7 10 12 15 20
 */
using LoD = std::vector<Vector<size_t>>;

std::ostream& operator<<(std::ostream& os, const LoD& lod);
std::ostream& operator<<(std::ostream& os, const LoDTensor& t);

/*
 * Slice levels from a LoD.
 * NOTE the lowest level should always be the absolute offsets of the underlying
 * tensor instances. So if higher layers are sliced without the lowest level,
 * the lower level of the sliced LoD will be transformed to the absolute offset.
 */
LoD SliceLevels(const LoD& in, size_t level_begin, size_t level_end);

LoD SliceInLevel(const LoD& in, size_t level, size_t elem_begin,
                 size_t elem_end);
/*
 * Transform an LoD from relative offsets to absolute offsets.
 */
LoD ToAbsOffset(const LoD& in);

bool operator==(const LoD& a, const LoD& b);

/*
 * LoDTensor (Level of details Tensor)
 * see https://en.wikipedia.org/wiki/Level_of_details for reference.
 */
class LoDTensor : public Tensor {
 public:
  LoDTensor() {}

  explicit LoDTensor(const LoD& lod) : lod_(lod) {}

  void set_lod(const LoD& lod) { lod_ = lod; }

  const LoD& lod() const { return lod_; }

  LoD* mutable_lod() { return &lod_; }

  /*
   * Get the start offset and end offset of an  element from LoD.
   */
  std::pair<size_t, size_t> lod_element(size_t level, size_t elem) const {
    PADDLE_ENFORCE_LT(level, NumLevels());
    PADDLE_ENFORCE_LT(elem, NumElements(level));
    return std::make_pair((lod_)[level][elem], (lod_)[level][elem + 1]);
  }

  /*
   * Number of LoDTensor's levels, each level has units of data, for example,
   * in the sentence's view, article, paragraph, sentence are 3 levels.
   */
  size_t NumLevels() const { return lod_.size(); }
  /*
   * Number of elements in a level.
   */
  size_t NumElements(size_t level = 0) const {
    PADDLE_ENFORCE_LT(level, NumLevels());
    // the last offset is the end of last element
    return (lod_)[level].size() - 1;
  }

  /*
   * Number of lower-level elements.
   * For example, a 2-level lod-tensor
   *
   * 0-th level   |   |
   * 1-th level   ||  |||
   *
   * NumElements(0, 0) get 2
   * NumElements(0, 1) get 3
   */
  size_t NumElements(size_t level, size_t idx) const;

  /*
   * Get the number of instances in the underlying tensor in the `idx`-th
   * element.
   */
  size_t NumInstancesInElement(size_t level, size_t idx) const;

  /*
   * Shrink levels[level_begin:level_end]
   */
  void ShrinkLevels(size_t level_begin, size_t level_end);

  /*
   * Shrink elements of a level, [elem_begin: elem_end]
   * @note: low performance in slice lod_.
   */
  void ShrinkInLevel(size_t level, size_t elem_begin, size_t elem_end);

  std::vector<LoDTensor> SplitLoDTensor(
      const std::vector<platform::Place> places) const;

  void MergeLoDTensor(const std::vector<const LoDTensor*>& lod_tensors,
                      platform::Place place);

 private:
  LoD lod_;
};

/*
 * Expand the `source` to fit the LoD of `lod`. For example, a `source`
 * LoDTensor is
 *  - LoD: [0, 2]
 *  - tensor: [a0, a1]
 * a `lod` is
 *  - LoD: [0 3 5]
 * returns a new LoDTensor
 *  - [a0 a0 a0 a1 a1]
 */
template <typename T>
LoDTensor LodExpand(const LoDTensor& source, const LoD& lod, size_t level,
                    const platform::Place& place) {
  LoD abs_lod = ToAbsOffset(lod);
  const auto& lod_level = lod[level];
  size_t num_instances = source.dims()[0];

  // new tensor
  LoDTensor tensor;
  tensor.set_lod(lod);
  auto dims = source.dims();
  dims[0] = lod_level.back();
  tensor.Resize(dims);
  tensor.mutable_data<T>(place);

  PADDLE_ENFORCE_EQ(num_instances, lod_level.size() - 1);
  for (size_t ins = 0; ins < num_instances; ins++) {
    for (size_t elem = lod_level[ins]; elem < lod_level[ins + 1]; elem++) {
      auto slice = tensor.Slice(elem, elem + 1);
      CopyFrom(source.Slice(ins, ins + 1), platform::CPUPlace(),
               platform::CPUDeviceContext(), &slice);
    }
  }
  return tensor;
}

// Get the absolute offset of a lod[start_level][start_idx:end_idx] and
// relative length of details for every levels(i.e., [start_level: ]).
//
// For example,
//   lod = [[0, 3, 4, 8], [0, 9, 10, 11, 13, 17, 19, 22, 24]]
//   start_level = 0
//   start_idx = 1
//   end_idx = 3
//
// Returns:
//  LoD = [[1, 4], [2, 4, 2, 3, 2]]
//  pair<size_t, size_t> = {11, 24}
std::pair<LoD, std::pair<size_t, size_t>> GetSubLoDAndAbsoluteOffset(
    const LoD& lod, size_t start_idx, size_t end_idx, size_t start_level);

void AppendLoD(LoD* lod, const LoD& lod_length);

/*
 * Serialize/Desiralize LoDTensor to std::ostream
 * You can pass ofstream or ostringstream to serilize to file
 * or to a in memory string. GPU tensor will be copied to CPU.
 */
void SerializeToStream(std::ostream& os, const LoDTensor& tensor,
                       const platform::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is, LoDTensor* tensor,
                           const platform::DeviceContext& dev_ctx);

}  // namespace framework
}  // namespace mypaddle
}  // namespace bubblefs