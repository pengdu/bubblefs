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

// tensorflow/tensorflow/core/framework/types.proto
// tensorflow/tensorflow/core/framework/types.h

#ifndef BUBBLEFS_UTILS_DATA_TYPES_H_
#define BUBBLEFS_UTILS_DATA_TYPES_H_

#include <map>
#include <set>
#include <string>
#include "utils/array_slice.h"
#include "utils/inlined_vector.h"
#include "utils/stringpiece.h"
#include "platform/logging.h"
#include "platform/types.h"

namespace bubblefs {

// LINT.IfChange
enum DataType {
  // Not a legal value for DataType.  Used to indicate a DataType field
  // has not been set.
  DT_INVALID = 0,

  // Data types that all computation devices are expected to be
  // capable to support.
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_UINT8 = 4,
  DT_INT16 = 5,
  DT_INT8 = 6,
  DT_STRING = 7,
  DT_COMPLEX64 = 8,  // Single-precision complex
  DT_INT64 = 9,
  DT_BOOL = 10,
  DT_QINT8 = 11,     // Quantized int8
  DT_QUINT8 = 12,    // Quantized uint8
  DT_QINT32 = 13,    // Quantized int32
  DT_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
  DT_QINT16 = 15,    // Quantized int16
  DT_QUINT16 = 16,   // Quantized uint16
  DT_UINT16 = 17,
  DT_COMPLEX128 = 18,  // Double-precision complex
  DT_HALF = 19,
  DT_RESOURCE = 20,
  DT_VARIANT = 21,  // Arbitrary C++ data types
  DT_UINT32 = 22,
  DT_UINT64 = 23,

  // Do not use!  These are only for parameters.  Every enum above
  // should have a corresponding value below (verified by types_test).
  DT_FLOAT_REF = 101,
  DT_DOUBLE_REF = 102,
  DT_INT32_REF = 103,
  DT_UINT8_REF = 104,
  DT_INT16_REF = 105,
  DT_INT8_REF = 106,
  DT_STRING_REF = 107,
  DT_COMPLEX64_REF = 108,
  DT_INT64_REF = 109,
  DT_BOOL_REF = 110,
  DT_QINT8_REF = 111,
  DT_QUINT8_REF = 112,
  DT_QINT32_REF = 113,
  DT_BFLOAT16_REF = 114,
  DT_QINT16_REF = 115,
  DT_QUINT16_REF = 116,
  DT_UINT16_REF = 117,
  DT_COMPLEX128_REF = 118,
  DT_HALF_REF = 119,
  DT_RESOURCE_REF = 120,
  DT_VARIANT_REF = 121,
  DT_UINT32_REF = 122,
  DT_UINT64_REF = 123,
};  // DataType
  
// MemoryType is used to describe whether input or output Tensors of
// an OpKernel should reside in "Host memory" (e.g., CPU memory) or
// "Device" Memory (CPU memory for CPU devices, GPU memory for GPU
// devices).
enum MemoryType {
  DEVICE_MEMORY = 0,
  HOST_MEMORY = 1,
};

// A DeviceType is just a string, but we wrap it up in a class to give
// some type checking as we're passing these around
class DeviceType {
 public:
  DeviceType(const char* type)  // NOLINT(runtime/explicit)
      : type_(type) {}

  explicit DeviceType(StringPiece type) : type_(type.data(), type.size()) {}

  const char* type() const { return type_.c_str(); }
  const string& type_string() const { return type_; }

  bool operator<(const DeviceType& other) const;
  bool operator==(const DeviceType& other) const;
  bool operator!=(const DeviceType& other) const { return !(*this == other); }

 private:
  string type_;
};
std::ostream& operator<<(std::ostream& os, const DeviceType& d);

// Convenient constants that can be passed to a DeviceType constructor
BASE_EXPORT extern const char* const DEVICE_CPU;   // "CPU"
BASE_EXPORT extern const char* const DEVICE_GPU;   // "GPU"
BASE_EXPORT extern const char* const DEVICE_SYCL;  // "SYCL"

template <typename Device>
struct DeviceName {};

/*
template <>
struct DeviceName<Eigen::ThreadPoolDevice> {
  static const std::string value;
};
*/

#if GOOGLE_CUDA
template <>
struct DeviceName<Eigen::GpuDevice> {
  static const std::string value;
};
#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
template <>
struct DeviceName<Eigen::SyclDevice> {
  static const std::string value;
};
#endif  // TENSORFLOW_USE_SYCL

typedef gtl::InlinedVector<MemoryType, 4> MemoryTypeVector;
typedef gtl::ArraySlice<MemoryType> MemoryTypeSlice;

typedef gtl::InlinedVector<DataType, 4> DataTypeVector;
typedef gtl::ArraySlice<DataType> DataTypeSlice;

typedef gtl::InlinedVector<DeviceType, 4> DeviceTypeVector;

// Convert the enums to strings for errors:
string DataTypeString(DataType dtype);
string DeviceTypeString(const DeviceType& device_type);
string DataTypeSliceString(const DataTypeSlice dtypes);
inline string DataTypeVectorString(const DataTypeVector& dtypes) {
  return DataTypeSliceString(dtypes);
}

// If "sp" names a valid type, store it in "*dt" and return true.  Otherwise,
// return false.
bool DataTypeFromString(StringPiece sp, DataType* dt);

// DT_FLOAT + kDataTypeRefOffset == DT_FLOAT_REF, etc.
enum { kDataTypeRefOffset = 100 };
inline bool IsRefType(DataType dtype) {
  return dtype > static_cast<DataType>(kDataTypeRefOffset);
}
inline DataType MakeRefType(DataType dtype) {
  DCHECK(!IsRefType(dtype));
  return static_cast<DataType>(dtype + kDataTypeRefOffset);
}
inline DataType RemoveRefType(DataType dtype) {
  DCHECK(IsRefType(dtype));
  return static_cast<DataType>(dtype - kDataTypeRefOffset);
}
inline DataType BaseType(DataType dtype) {
  return IsRefType(dtype) ? RemoveRefType(dtype) : dtype;
}

// Returns true if the actual type is the same as or ref of the expected type.
inline bool TypesCompatible(DataType expected, DataType actual) {
  return expected == actual || expected == BaseType(actual);
}

// Does not include _ref types.
DataTypeVector AllTypes();

// Return the list of all numeric types.
// NOTE: On Android, we only include the float and int32 types for now.
DataTypeVector RealNumberTypes();  // Types that support '<' and '>'.
DataTypeVector NumberTypes();      // Includes complex and quantized types.

DataTypeVector QuantizedTypes();
DataTypeVector RealAndQuantizedTypes();  // Types that support '<' and
                                         // '>', including quantized
                                         // types

// Validates type T for whether it is a supported DataType.
template <class T>
struct IsValidDataType;

// DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
// constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
template <class T>
struct DataTypeToEnum {
  static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
};  // Specializations below

// EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
// EnumToDataType<DT_FLOAT>::Type is float.
template <DataType VALUE>
struct EnumToDataType {};  // Specializations below

// Template specialization for both DataTypeToEnum and EnumToDataType.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)                 \
  template <>                                           \
  struct DataTypeToEnum<TYPE> {                         \
    static DataType v() { return ENUM; }                \
    static DataType ref() { return MakeRefType(ENUM); } \
    static constexpr DataType value = ENUM;             \
  };                                                    \
  template <>                                           \
  struct IsValidDataType<TYPE> {                        \
    static constexpr bool value = true;                 \
  };                                                    \
  template <>                                           \
  struct EnumToDataType<ENUM> {                         \
    typedef TYPE Type;                                  \
  }

MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
MATCH_TYPE_AND_ENUM(double, DT_DOUBLE);
MATCH_TYPE_AND_ENUM(int32, DT_INT32);
MATCH_TYPE_AND_ENUM(uint32, DT_UINT32);
MATCH_TYPE_AND_ENUM(uint16, DT_UINT16);
MATCH_TYPE_AND_ENUM(uint8, DT_UINT8);
MATCH_TYPE_AND_ENUM(int16, DT_INT16);
MATCH_TYPE_AND_ENUM(int8, DT_INT8);
MATCH_TYPE_AND_ENUM(string, DT_STRING);
//MATCH_TYPE_AND_ENUM(complex64, DT_COMPLEX64);
//MATCH_TYPE_AND_ENUM(complex128, DT_COMPLEX128);
MATCH_TYPE_AND_ENUM(int64, DT_INT64);
MATCH_TYPE_AND_ENUM(uint64, DT_UINT64);
MATCH_TYPE_AND_ENUM(bool, DT_BOOL);
//MATCH_TYPE_AND_ENUM(qint8, DT_QINT8);
//MATCH_TYPE_AND_ENUM(quint8, DT_QUINT8);
//MATCH_TYPE_AND_ENUM(qint16, DT_QINT16);
//MATCH_TYPE_AND_ENUM(quint16, DT_QUINT16);
//MATCH_TYPE_AND_ENUM(qint32, DT_QINT32);
//MATCH_TYPE_AND_ENUM(bfloat16, DT_BFLOAT16);
//MATCH_TYPE_AND_ENUM(Eigen::half, DT_HALF);
//MATCH_TYPE_AND_ENUM(ResourceHandle, DT_RESOURCE);
//MATCH_TYPE_AND_ENUM(Variant, DT_VARIANT);

#undef MATCH_TYPE_AND_ENUM

// All types not specialized are marked invalid.
template <class T>
struct IsValidDataType {
  static constexpr bool value = false;
};

// Extra validity checking; not part of public API.
static_assert(IsValidDataType<int64>::value, "Incorrect impl for int64");
static_assert(IsValidDataType<int32>::value, "Incorrect impl for int32");

bool DataTypeCanUseMemcpy(DataType dt);

bool DataTypeIsQuantized(DataType dt);

// Is the dtype nonquantized integral?
bool DataTypeIsInteger(DataType dt);

// Returns a 0 on failure
int DataTypeSize(DataType dt);

}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_DATA_TYPES_H_