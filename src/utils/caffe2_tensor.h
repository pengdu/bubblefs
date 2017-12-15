/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/core/tensor.h

#ifndef BUBBLEFS_UTILS_CAFFE2_TENSOR_H_
#define BUBBLEFS_UTILS_CAFFE2_TENSOR_H_

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <type_traits>
#include <vector>

#include "platform/base_error.h"
#include "platform/types.h"
#include "utils/caffe2_context.h"
#include "utils/caffe2_typeid.h"

namespace bubblefs {
namespace mycaffe2 {
  
// A global boolean variable to control whether we free memory when a Tensor
// is shrinked to a smaller size. As a result, a Tensor is always going to
// keep the memory allocated for its maximum capacity reshaped to so far.
extern bool FLAGS_caffe2_keep_on_shrink;

// Since we can have high variance in blob memory allocated across different
// inputs in the same run, we will shrink the blob only if the memory gain
// is larger than this flag in bytes.
extern int64 FLAGS_caffe2_max_keep_on_shrink_memory;   
  
/**
 * A utility function to convert vector<int> to vector<TIndex>.
 */
inline std::vector<TIndex> ToVectorTIndex(const std::vector<int>& src) {
  return std::vector<TIndex>(src.begin(), src.end());
}

/**
 * Return product of all dimensions starting from K
 */
inline TIndex size_from_dim_(int k, std::vector<TIndex> dims) {
  TIndex r = 1;
  for (int i = k; i < dims.size(); ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims up to
inline TIndex size_to_dim_(int k, std::vector<TIndex> dims) {
  PANIC_ENFORCE(k <= dims.size(), "k > dims.size()");
  TIndex r = 1;
  for (int i = 0; i < k; ++i) {
    r *= dims[i];
  }
  return r;
}

// Product of all dims between k and l (not including dims[k] and dims[l])
inline TIndex size_between_dim_(int k, int l, vector<TIndex> dims) {
  PANIC_ENFORCE(l < dims.size(), "l >= dims.size()");
  TIndex r = 1;
  if (k < l) {
    for (int i = k + 1; i < l; ++i) {
      r *= dims[i];
    }
  } else {
    for (int i = l + 1; i < k; ++i) {
      r *= dims[i];
    }
  }
  return r;
}

inline int canonical_axis_index_(int axis_index, int ndims) {
  PANIC_ENFORCE_GE(axis_index, -ndims);
  PANIC_ENFORCE_LT(axis_index, ndims);
  if (axis_index < 0) {
    return axis_index + ndims;
  }
  return axis_index;
}

/**
 * @brief Tensor is the basic class in Caffe2 that stores a contiguous memory
 * with its shape information.
 *
 * The Tensor class is essentially a wrapper around a device-specific memory
 * (the device is specified by the Context template argument), and deals with
 * the allocation and de-allocation of such memory. We make a simplified
 * assumption that the memory is always contiguous.
 */
template <class Context>
class Tensor {
 public:
  /**
   * Initializes an empty tensor.
   */
  Tensor() {}

  /**
   * @brief Creates a tensor of the given dimension.
   *
   * Note that the actual data allocation is not going to be carried out until
   * the first time mutable_data() is called.
   */
  explicit Tensor(const std::vector<TIndex>& dims) { Resize(dims); }
  explicit Tensor(const std::vector<int>& dims) { Resize(dims); }

  /**
   * @brief Creates a tensor from a source tensor, copying over the content.
   *
   * Note that the source tensor can be from a different device context. The
   * second argument provides a device context object (either Context or
   * SrcContext) that will be responsible for copying the underlying data.
   * If you do not wish to pass in a Context object, an equivalent constructor
   * function exists that will create an implicit context object for copy, but
   * be noted that this will cause a potential performance hit.
   */
  template <class SrcContext, class ContextForCopy>
  Tensor(const Tensor<SrcContext>& src, ContextForCopy* context) {
    CopyFrom(src, context);
  }

  /**
   * @brief Creates a tensor from a source tensor, copying over the content.
   *
   * Note that this may have a potential performance hit, since a temporary
   * context object will be created for the memory copy. Prefer explicitly
   * providing a context for copy if you can.
   */
  template <class SrcContext>
  Tensor(const Tensor<SrcContext>& src) {
    CopyFrom(src);
  }

  /**
   * @brief Creates a tensor, and fills its contents with the given values.
   */
  template <typename T>
  Tensor(const std::vector<TIndex>& dims, const std::vector<T>& values, Context* context)
      : meta_(TypeMeta::Make<T>()) {
    Resize(dims);
    PANIC_ENFORCE_EQ(values.size(), size_);
    context->template Copy<T, CPUContext, Context>(size_, values.data(), mutable_data<T>());
  }

  /**
   * @brief Creates a scalar tensor, and fills its content with the given value.
   */
  template <typename T,
            typename = typename std::enable_if<std::is_scalar<T>::value>::type>
  Tensor(const T& value, Context* context) {
    Resize(std::vector<TIndex>{});
    context->template Copy<T, CPUContext, Context>(size_, &value, mutable_data<T>());
  }

  /**
   * @brief Copies the data from a source tensor, with a contex provided to
   * carry out the underlying memcpy operation.
   */
  template <class SrcContext, class ContextForCopy>
  void CopyFrom(const Tensor<SrcContext>& src, ContextForCopy* context) {
    if ((void*)&src == (void*)this) {
      return;
    }
    meta_ = src.meta();
    Resize(src.dims());
    if (size() > 0) {
      if (meta_.copy()) {
        meta_.copy()(src.raw_data(), raw_mutable_data(), size());
      } else {
        context->template CopyBytes<SrcContext, Context>(
            nbytes(), src.raw_data(), raw_mutable_data());
      }
    }
  }

  /**
   * @brief Copies the data from a source tensor.
   *
   * Note that this may have a potential performance hit, since a temporary
   * context object will be created for the memory copy. Prefer explicitly
   * providing a context for copy if you can.
   */
  template <class SrcContext>
  inline void CopyFrom(const Tensor<SrcContext>& src) {
    SrcContext tmp_context;
    CopyFrom(src, &tmp_context);
  }

  virtual ~Tensor() noexcept {}

  /**
   * @brief Extends the outer-most dimension of this tensor by num elements,
   * preserving the existing data.
   *
   * The underlying data may be reallocated in order to accommodate the new
   * elements, in which case this tensors' capacity is grown at a factor of
   * growthPct. This ensures that Extend runs on an amortized O(1) time
   * complexity.
   */
  template <class ContextForCopy>
  void Extend(TIndex num, float growthPct, ContextForCopy* context) {
    PANIC_ENFORCE_GE(dims_.size(), 1);
    auto newDims = dims_;
    newDims[0] += num;
    if (!data_) {
      Resize(newDims);
      return;
    }
    auto newSize = std::accumulate(
        newDims.begin(),
        newDims.end(),
        static_cast<TIndex>(1),
        std::multiplies<TIndex>());
    if (newSize * meta_.itemsize() <= capacity_) {
      dims_ = newDims;
      size_ = newSize;
      return;
    }
    auto newCapacity = dims_;
    newCapacity[0] = std::max<size_t>(
        newDims[0], std::ceil(dims_[0] * (growthPct + 100) / 100));
    Reserve(newCapacity, context);
    dims_ = newDims;
    size_ = newSize;
  }

  template <class T, class ContextForCopy>
  void Reserve(const std::vector<T>& newCapacity, ContextForCopy* context) {
    auto newSize = std::accumulate(
        newCapacity.begin(),
        newCapacity.end(),
        static_cast<TIndex>(1),
        std::multiplies<TIndex>());
    if (newSize * meta_.itemsize() <= capacity_) {
      return;
    }
    auto oldData = std::move(data_);
    auto oldSize = size_;
    auto oldDims = dims_;
    Resize(newCapacity);
    auto* newData = raw_mutable_data(meta_);
    context->template CopyItems<ContextForCopy, ContextForCopy>(
        meta_, oldSize, oldData.get(), newData);
    dims_ = oldDims;
    size_ = oldSize;
    reserved_ = true;
  }

  /**
   * @brief Shrinks the outer-most dimension to given size, keeping the data.
   *
   * This method guarantees that no re-allocations are carried out, which means
   * that the extra capacity after the end of the shurnk tensor is maintained.
   */
  void Shrink(TIndex outer_dim) {
    PANIC_ENFORCE(dims_.size() >= 1, "Tensor must be at least 1D");
    PANIC_ENFORCE(
        outer_dim <= dims_[0],
        "New outer dimension must be smaller than current.");
    dims_[0] = outer_dim;
    size_ = std::accumulate(
        dims_.begin(),
        dims_.end(),
        static_cast<TIndex>(1),
        std::multiplies<TIndex>());
  }

  /**
   * @brief Resizes a tensor.
   *
   * Resize takes in a vector of ints specifying the dimensions of the tensor.
   * You can pass in an empty vector to specify that it is a scalar (i.e.
   * containing one single item).
   *
   * The underlying storage may be deleted after calling Resize: if the new
   * shape leads to a different number of items in the tensor, the old memory
   * is deleted and new memory will be allocated next time you call
   * mutable_data(). However, if the shape is different but the total number of
   * items is the same, the underlying storage is kept.
   */
  template <typename... Ts>
  void Resize(Ts... dim_source) {
    bool size_changed = SetDims(dim_source...);
    if (size_changed) {
      // If needed, we will free the data. the next mutable_data() call
      // will create the data storage.
      int64_t new_size = size_ * meta_.itemsize();
      bool reset_tensor = false;
      if (reserved_) {
        // If tensor is reserved then don't claim its memeory unless capacity_
        // is smaller than new size
        reset_tensor = capacity_ < new_size;
      } else {
        reset_tensor = capacity_ < new_size || !FLAGS_caffe2_keep_on_shrink ||
            capacity_ - new_size > FLAGS_caffe2_max_keep_on_shrink_memory;
      }

      if (reset_tensor) {
        FreeMemory();
      }
    }
  }

  /**
   * Resize the tensor like the source tensor. Note that this is just a
   * sugar wrapper that essentially calls Resize(src_tensor.dims()).
   */
  template <class OtherContext>
  inline void ResizeLike(const Tensor<OtherContext>& src_tensor) {
    // Note: need casting for different context types.
    if (static_cast<void*>(this) != static_cast<const void*>(&src_tensor)) {
      Resize(src_tensor.dims());
    }
  }

  /**
   * Resizes the tensor without touching underlying storage.
   * This requires the total size of the tensor to remains constant.
   */
  inline void Reshape(const std::vector<TIndex>& dims) {
    TIndex new_size = 1;
    for (auto d : dims) {
      PANIC_ENFORCE_GE(d, 0);
      new_size *= d;
    }
    PANIC_ENFORCE(
        new_size == size_,
        "New size and old size are not equal. You cannot use Reshape, "
        "but should use Resize."
        // TODO(jiayq): remove the following warning after pending diffs
        // stabilize.
        " The old caffe2 mixes Reshape and Resize but this behavior has "
        "been changed. If you find this error, most likely you will need "
        "to change corresponding code from Reshape to Resize.");
    dims_ = dims;
  }

  inline void Reshape(const std::vector<int>& dims) {
    Reshape(ToVectorTIndex(dims));
  }

  /**
   * Release whatever memory the tensor was holding but keep size and type
   * information. Subsequent call to mutable_data will trigger new memory
   * allocation.
   */
  inline void FreeMemory() {
    data_.reset();
    capacity_ = 0;
    // If reserved is true and we changed tensor memory then it is fine
    // to switch it to false, if Resize is called from Reserve and it triggers
    // FreeMemory() then reserved_ will be set to true at end of Reserve()
    reserved_ = false;
  }

  /**
   * A utility function to print the debug string for the tensor. Note that this
   * is very slow since it involves quite some string operations, so do not use
   * it in your performance-critical code.
   */
  string DebugString() const {
    std::stringstream ss;
    ss << "A Tensor of item size " << itemsize() << " and type "
       << meta_.name() << " and dimension (";
    for (int d : dims_) {
      ss << d << ",";
    }
    ss << ").";
    return ss.str();
  }

  void swap(Tensor<Context>& other) {
    std::swap(dims_, other.dims_);
    std::swap(size_, other.size_);
    std::swap(meta_, other.meta_);
    std::swap(data_, other.data_);
    std::swap(shares_data_, other.shares_data_);
    std::swap(capacity_, other.capacity_);
    std::swap(reserved_, other.reserved_);
  }

  /**
   * @brief Shares the data with another tensor.
   *
   * To share data between two tensors, the sizes of the two tensors must be
   * equal already. The reason we do not implicitly do a Resize to make the two
   * tensors have the same shape is that we want to allow tensors of different
   * shapes but the same number of items to still be able to share data. This
   * allows one to e.g. have a n-dimensional Tensor and a flattened version
   * sharing the same underlying storage.
   *
   * The source tensor should already have its data allocated.
   */
  void ShareData(const Tensor& src) {
    meta_ = src.meta();
    PANIC_ENFORCE_EQ(src.size_, size_); // "Size mismatch - did you call reshape before sharing the data?");
    // It is possible that the source tensor hasn't called mutable_data() yet,
    // in which case ShareData() doesn't make much sense since we don't really
    // know what to share yet.
    PANIC_ENFORCE(
        src.data_.get() || src.size_ == 0,
        "Source tensor has no content and has size > 0");
    // Finally, do sharing.
    data_ = src.data_;
    capacity_ = src.capacity_;
    shares_data_ = true;
  }

  /**
   * @brief Shares the data with an externally managed pointer.
   *
   * This is similar to ShareData() but the source is a pointer with an advanced
   * deleter option. In default, no deletion takes place, and one needs to make
   * sure that the external memory is deallocated only after the tensor finishes
   * using it. If a Deleter object is passed in, when this tensor is reallocated
   * or freed, the deleter function is going to be called.
   */
  template <typename T, typename Deleter = MemoryDeleter>
  void ShareExternalPointer(T* src, size_t capacity = 0, Deleter d = nullptr) {
    ShareExternalPointer(src, TypeMeta::Make<T>(), capacity, d);
  }

  template <typename Deleter = MemoryDeleter>
  void ShareExternalPointer(
      void* src,
      const TypeMeta& meta,
      size_t capacity = 0,
      Deleter d = nullptr) {
    meta_ = meta;
    PANIC_ENFORCE(
        meta_.id(),
        "To share with a raw external pointer you need to have meta "
        "already set.");
    PANIC_ENFORCE(
        size_ >= 0,
        "To share data with a raw pointer, you need to set shape first.");
    // Check if the deleter is a MemoryDeleter and is a simple nullptr.
    if (std::is_same<MemoryDeleter, Deleter>::value &&
        reinterpret_cast<MemoryDeleter*>(&d)[0] == nullptr) {
      // Use aliasing constructor trick to avoid calling the destructor.
      data_ = std::shared_ptr<void>(std::shared_ptr<void>(), src);
    } else {
      data_.reset(src, d);
    }
    // Sets capacity. If not specified, we will implicitly assume that
    // the capacity is the current size.
    if (capacity) {
      capacity_ = capacity;
    } else {
      capacity_ = nbytes();
    }
    shares_data_ = true;
  }

  bool shares_data() const {
    return shares_data_;
  }

  /**
   * Returns a const raw void* pointer of the underlying storage. mutable_data()
   * or raw_mutable_data() must have been called prior to this function call.
   */
  inline const void* raw_data() const {
    PANIC_ENFORCE(data_.get() || size_ == 0, "data is empty");
    return data_.get();
  }

  /**
   * Returns a typed pointer of the underlying storage. mutable_data() or
   * raw_mutable_data() must have been called prior to this function call, and
   * the data type must be of the correct type. If you want to get a void*
   * pointer instead, use raw_data().
   */
  template <typename T>
  inline const T* data() const {
    PANIC_ENFORCE(
        data_.get() || size_ == 0,
        "The tensor is of non-zero shape, but its data is not allocated yet. "
        "Caffe2 uses a lazy allocation, so you will need to call "
        "mutable_data() or raw_mutable_data() to actually allocate memory.");
    PANIC_ENFORCE(
        IsType<T>(),
        "Tensor type mismatch, caller expects elements to be %s while tensor contains %s",
        TypeMeta::Name<T>(),
        meta_.name());
    return static_cast<T*>(data_.get());
  }

  /**
   * Returns a mutable raw pointer of the underlying storage. Since we will need
   * to know the type of the data for allocation, a TypeMeta object is passed in
   * to specify the necessary information. This is conceptually equivalent of
   * calling mutable_data<T>() where the TypeMeta parameter meta is derived from
   * the type T. This function differs from mutable_data<T>() in the sense that
   * the type T can be specified during runtime via the TypeMeta object.
   *
   * If the existing data does not match the desired type, it will be deleted
   * and a new storage will be created.
   */
  inline void* raw_mutable_data(const TypeMeta& meta) {
    // For 0-size tensors it's fine to return any pointer (including nullptr)
    if (meta_ == meta && (data_.get() || size_ == 0)) {
      return data_.get();
    } else {
      bool had_special_dtor = meta_.dtor() != nullptr;
      meta_ = meta;
      PANIC_ENFORCE(
          size_ >= 0,
          "Tensor is not initialized. You probably need to call Resize() "
          "before calling mutable_data()");

      // We can reuse the existing buffer if the current data does not have
      // a special destructor and the new data doesn't have a special
      // constructor.
      if (size_ == 0 ||
          (meta.ctor() == nullptr && !had_special_dtor &&
           capacity_ >= size_ * meta_.itemsize())) {
        return data_.get();
      }
      if (meta.ctor()) {
        // For types that need placement new, we will call it, as well as
        // making sure that when the data is freed, it calls the right
        // destruction procedure.
        auto size = size_;
        auto dtor = meta_.dtor();
        auto ptr_and_deleter = Context::New(size_ * meta_.itemsize());
        auto deleter = ptr_and_deleter.second;
        data_.reset(
            ptr_and_deleter.first, [size, dtor, deleter](void* ptr) -> void {
              dtor(ptr, size);
              deleter(ptr);
            });
        meta_.ctor()(data_.get(), size_);
      } else {
        // For fundamental type, new and delete is easier.
        auto ptr_and_deleter = Context::New(size_ * meta_.itemsize());
        data_.reset(ptr_and_deleter.first, ptr_and_deleter.second);
      }
      capacity_ = size_ * meta_.itemsize();
      return data_.get();
    }
  }

  /**
   * Returns a mutable raw pointer of the underlying storage. This can only be
   * used when you know for sure that the underlying storage of the tensor is
   * already created via an earlier raw_mutable_data(meta) call or a
   * mutable_data<T>() call.
   *
   * If the existing data does not match the desired type, it will be deleted
   * and a new storage will be created.
   */
  inline void* raw_mutable_data() {
    PANIC_ENFORCE(
        meta_.id() != 0,
        "Calling raw_mutable_data() without meta, but the current meta is "
        "of unknown type.");
    return raw_mutable_data(meta_);
  }

  /**
   * Returns a typed pointer of the underlying storage.
   *
   * For fundamental types, we reuse possible existing storage if there
   * is sufficient capacity.
   */
   template <typename T>
    inline T* mutable_data() {
      if ((size_ == 0 || data_.get()) && IsType<T>()) {
        return static_cast<T*>(data_.get());
      }
      return static_cast<T*>(raw_mutable_data(TypeMeta::Make<T>()));
    }


  /**
   * Returns the number of dimensions of the data.
   */
  inline int ndim() const { return dims_.size(); }
  /**
   * Returns the size (i.e. the number of items) of the tensor.
   */
  inline TIndex size() const { return size_; }
  /**
   * Return the number of bytes each item takes in the tensor.
   */
  inline size_t itemsize() const { return meta_.itemsize(); }
  /**
   * Returns the total number of bytes of the storage.
   *
   * This is equivalent to calling size() * itemsize().
   */
  inline size_t nbytes() const { return size_ * meta_.itemsize(); }

  inline size_t capacity_nbytes() const {
    return capacity_;
  }
  /**
   * Returns the dimensions of the tensor as a vector.
   */
  inline const std::vector<TIndex>& dims() const { return dims_; }

  inline TIndex size_from_dim(int k) const {
    return size_from_dim_(k, dims_);
  }

  inline TIndex size_to_dim(int k) const {
    return size_to_dim_(k, dims_);
  }

  inline TIndex size_between_dim(int k, int l) const {
    return size_between_dim_(k, l, dims_);
  }

  /**
  * Returns the 'canonical' version of a (usually)  user-specified axis,
  * allowing for negative indexing (e.g., -1 for the last axis).
  *
  * @param axis_index the axis index.
  *        If 0 <= index < ndim(), return index.
  *        If -ndim <= index <= -1, return (ndim() - (-index)),
  *        e.g., the last axis index (ndim() - 1) if index == -1,
  *        the second to last if index == -2, etc.
  *        Dies on out of range index.
  */
  inline int canonical_axis_index(int axis_index) const {
    return canonical_axis_index_(axis_index, ndim());
  }

  /**
   * Checks if the tensor content is of the given data type.
   */
  template <typename T>
  inline bool IsType() const { return meta_.Match<T>(); }
  /**
   * Returns the TypeMeta object associated with the current data type.
   */
  inline const TypeMeta& meta() const { return meta_; }

  /**
   * Returns the i-th dimension of the tensor in int.
   *
   * This function returns an int value instead of TIndex, which depending on
   * the typedef could be int64. If you want int64 dim values, make sure you
   * call dim() instead.
   */
  inline int dim32(const int i) const {
    #ifndef NDEBUG
    PANIC_ENFORCE_LT(i, dims_.size()); // "Exceeding ndim limit");
    PANIC_ENFORCE_GE(i, 0); // "Cannot have negative dimension index");
    #endif
    PANIC_ENFORCE_LT(dims_[i], std::numeric_limits<int>::max());
    return static_cast<int>(dims_[i]);
  }

  /**
   * Returns the i-th dimension of the tensor. Note that the passed in index
   * must be between 0 (inclusive) and the number of dimensions, otherwise
   * this function will produce a fatal message.
   */
  inline TIndex dim(const int i) const {
    #ifndef NDEBUG
    PANIC_ENFORCE_LT(i, dims_.size()); // "Exceeding ndim limit");
    PANIC_ENFORCE_GE(i, 0); // "Cannot have negative dimension index");
    #endif
    return dims_[i];
  }

 protected:
  std::vector<TIndex> dims_;
  TIndex size_ = -1;
  TypeMeta meta_;
  std::shared_ptr<void> data_;
  bool shares_data_ = false;
  size_t capacity_ = 0;
  bool reserved_ = false;
  // In case of chunk load we store how much data was already loaded

 private:
  template <
      typename T,
      typename = typename std::enable_if<std::is_integral<T>::value>::type>
  bool SetDims(const std::vector<T>& src) {
    auto old_size = size_;
    dims_.resize(src.size());
    TIndex new_size = 1;
    for (unsigned int i = 0; i < src.size(); ++i) {
      new_size *= src[i];
      dims_[i] = src[i];
    }
    size_ = new_size;
    return size_ != old_size;
  }

  bool SetDims() {
    auto old_size = size_;
    dims_.resize(0);
    size_ = 1;
    return size_ != old_size;
  }

  // TODO(jiayq): maybe rewrite the following functions with initializer list.
  // NVCC does not play well with initializer lists last time, but worth
  // another shot.
  bool SetDims(const TIndex d0) {
    auto old_size = size_;
    dims_.resize(1);
    dims_[0] = d0;
    size_ = d0;
    return size_ != old_size;
  }

  bool SetDims(const TIndex d0, const TIndex d1) {
    auto old_size = size_;
    dims_.resize(2);
    dims_[0] = d0;
    dims_[1] = d1;
    size_ = d0 * d1;
    return size_ != old_size;
  }

  bool SetDims(const TIndex d0, const TIndex d1, const TIndex d2) {
    auto old_size = size_;
    dims_.resize(3);
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    size_ = d0 * d1 * d2;
    return size_ != old_size;
  }

  bool
  SetDims(const TIndex d0, const TIndex d1, const TIndex d2, const TIndex d3) {
    auto old_size = size_;
    dims_.resize(4);
    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    dims_[3] = d3;
    size_ = d0 * d1 * d2 * d3;
    return size_ != old_size;
  }

  // Note(jiayq): possibly a rule-of-three violation, but we explicitly
  // discourage the use of = for Tensors.
  Tensor& operator=(const Tensor& src) = delete;
};

// For simplicity, we will typedef Tensor<CPUContext> to TensorCPU.
typedef Tensor<CPUContext> TensorCPU;

constexpr int k_limit_default_ = 1000;

// Type call registry
typedef TypeMeta (*TypeCall)(const void*);
TypeCall GetTypeCallFunction(CaffeTypeId id);
void RegisterTypeCallFunction(CaffeTypeId id, TypeCall c);

template <class Context>
TypeMeta GetTensorType(const void* c) {
  const Tensor<Context>* tc = static_cast<const Tensor<Context>*>(c);
  return tc->meta();
}

// Shape call registry
typedef std::vector<TIndex> (*TensorInfoCall)(
    const void*,
    bool* shares_data,
    size_t* capacity,
    DeviceOption* device);
TensorInfoCall GetTensorInfoFunction(CaffeTypeId id);
void RegisterTensorInfoFunction(CaffeTypeId id, TensorInfoCall c);

template <class Context>
std::vector<TIndex> GetTensorInfo(
    const void* c,
    bool* shares_data,
    size_t* capacity,
    DeviceOption* device) {
  const Tensor<Context>* tc = static_cast<const Tensor<Context>*>(c);
  *shares_data = tc->shares_data();
  *capacity = tc->capacity_nbytes();
  device->device_type = CPU;
  device->cuda_gpu_id = 0;
  return tc->dims();
}

class TensorPrinter {
 public:
  explicit TensorPrinter(
      const std::string& tensor_name = "",
      const std::string& file_name = "",
      int limit = k_limit_default_);
  ~TensorPrinter();

  template <class T>
  void Print(const Tensor<CPUContext>& tensor);

  template <class Context>
  void PrintMeta(const Tensor<Context>& tensor);

  string MetaStr(const Tensor<CPUContext>& tensor);

 private:
  bool to_file_;
  int limit_;
  std::unique_ptr<std::ofstream> log_file_;
  std::string tensor_name_;
};

template <class T>
void TensorPrinter::Print(const Tensor<CPUContext>& tensor) {
  std::stringstream values_stream;
  // One most likely doesn't want to print int64-number of items for visual
  // inspection, so we cast down to int here.
  int total_count = std::min(tensor.size(), TIndex(limit_));
  const T* tensor_data = tensor.template data<T>();
  for (int i = 0; i < total_count - 1; ++i) {
    values_stream << tensor_data[i] << ",";
  }
  // We do not add a comma after the last item.
  values_stream << tensor_data[total_count - 1];
  if (to_file_) {
    (*log_file_) << MetaStr(tensor) << values_stream.str() << std::endl;
  } else {
    // Log to console.
    //LOG(INFO) << MetaStr(tensor) << values_stream.str();
  }
}

template <class Context>
void TensorPrinter::PrintMeta(const Tensor<Context>& tensor) {
  if (to_file_) {
    (*log_file_) << MetaStr(tensor) << std::endl;
  } else {
    //LOG(INFO) << MetaStr(tensor);
  }
}

TensorPrinter::TensorPrinter(
    const std::string& tensor_name,
    const std::string& file_name,
    int limit)
    : to_file_(!file_name.empty()),
      limit_(limit ? limit : k_limit_default_),
      tensor_name_(tensor_name) {
  if (to_file_) {
    // We will output to file instead of printing on screen.
    // We will write each individual tensor to its individual file.
    log_file_.reset(new std::ofstream(
        file_name, std::ofstream::out | std::ofstream::trunc));
    //CAFFE_ENFORCE(
    //    log_file_->good(),
    //    "Failed to open TensorPrinter file ",
    //    file_name,
    //    ". rdstate() = ",
    //    log_file_->rdstate());
  }
}

TensorPrinter::~TensorPrinter() {
  if (log_file_.get()) {
    log_file_->close();
  }
}

std::string TensorPrinter::MetaStr(const Tensor<CPUContext>& tensor) {
  std::stringstream meta_stream;
  meta_stream << "Tensor " << tensor_name_ << " of type "
              << tensor.meta().name() << ". Dims: (";
  for (const auto dim : tensor.dims()) {
    meta_stream << dim << ",";
  }
  meta_stream << "): ";
  return meta_stream.str();
}

static CaffeMap<CaffeTypeId, TypeCall> type_call_registry_ {
  {TypeMeta::Id<Tensor<CPUContext>>(), GetTensorType<CPUContext>}
};

TypeCall GetTypeCallFunction(CaffeTypeId id) {
  auto f = type_call_registry_.find(id);
  if (f == type_call_registry_.end()) {
    return nullptr;
  }
  return f->second;
}

void RegisterTypeCallFunction(CaffeTypeId id, TypeCall c) {
  type_call_registry_[id] = c;
}

static std::map<CaffeTypeId, TensorInfoCall> tensor_info_call_registry_{
    {TypeMeta::Id<Tensor<CPUContext>>(), GetTensorInfo<CPUContext>}};

TensorInfoCall GetTensorInfoFunction(CaffeTypeId id) {
  auto f = tensor_info_call_registry_.find(id);
  if (f == tensor_info_call_registry_.end()) {
    return nullptr;
  }
  return f->second;
}

void RegisterTensorInfoFunction(CaffeTypeId id, TensorInfoCall c) {
  tensor_info_call_registry_[id] = c;
}

}  // namespace mycaffe2
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_CAFFE2_TENSOR_H_