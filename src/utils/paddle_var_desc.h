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

#pragma once

#include <vector>
#include "glog/logging.h"
#include "utils/paddle_framework_proto.h"

namespace bubblefs {
namespace mypaddle {
namespace framework {

// convert between std::vector and protobuf repeated.
template <typename T>
inline std::vector<T> RepeatedToVector(
    const google::protobuf::RepeatedField<T> &repeated_field) {
  std::vector<T> ret;
  ret.reserve(repeated_field.size());
  std::copy(repeated_field.begin(), repeated_field.end(),
            std::back_inserter(ret));
  return ret;
}

template <typename T, typename RepeatedField>
inline void VectorToRepeated(const std::vector<T> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Clear();
  repeated_field->Reserve(vec.size());
  for (const auto &elem : vec) {
    *repeated_field->Add() = elem;
  }
}

// Specialize vector<bool>.
template <typename RepeatedField>
inline void VectorToRepeated(const std::vector<bool> &vec,
                             RepeatedField *repeated_field) {
  repeated_field->Clear();
  repeated_field->Reserve(vec.size());
  for (auto elem : vec) {
    *repeated_field->Add() = elem;
  }
}

class VarDesc {
 public:
  explicit VarDesc(const std::string &name) {
    desc_.set_name(name);
    desc_.set_type(proto::VarDesc::LOD_TENSOR);
  }

  explicit VarDesc(const proto::VarDesc &desc) : desc_(desc) {}

  proto::VarDesc *Proto() { return &desc_; }

  std::string Name() const { return desc_.name(); }

  void SetShape(const std::vector<int64_t> &dims);

  void SetDataType(proto::DataType data_type);

  std::vector<int64_t> Shape() const;

  proto::DataType GetDataType() const;

  void SetLoDLevel(int32_t lod_level);

  int32_t GetLoDLevel() const;

  proto::VarDesc::VarType GetType() const;

  void SetType(proto::VarDesc::VarType type);

  bool Persistable() const { return desc_.persistable(); }

  void SetPersistable(bool persistable) { desc_.set_persistable(persistable); }

 private:
  const proto::TensorDesc &tensor_desc() const;
  proto::TensorDesc *mutable_tensor_desc();

  proto::VarDesc desc_;
};

proto::VarDesc::VarType VarDesc::GetType() const { return desc_.type(); }

void VarDesc::SetType(proto::VarDesc::VarType type) { desc_.set_type(type); }

void VarDesc::SetShape(const std::vector<int64_t> &dims) {
  VectorToRepeated(dims, mutable_tensor_desc()->mutable_dims());
}

void VarDesc::SetDataType(proto::DataType data_type) {
  mutable_tensor_desc()->set_data_type(data_type);
}

std::vector<int64_t> VarDesc::Shape() const {
  return RepeatedToVector(tensor_desc().dims());
}

proto::DataType VarDesc::GetDataType() const {
  return tensor_desc().data_type();
}

void VarDesc::SetLoDLevel(int32_t lod_level) {
  switch (desc_.type()) {
    case proto::VarDesc::LOD_TENSOR:
      desc_.mutable_lod_tensor()->set_lod_level(lod_level);
      break;
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      desc_.mutable_tensor_array()->set_lod_level(lod_level);
      break;
    default:
      PADDLE_THROW("Tensor type=%d does not support LoDLevel",
                   desc_.tensor_array().lod_level());
  }
}

int32_t VarDesc::GetLoDLevel() const {
  switch (desc_.type()) {
    case proto::VarDesc::LOD_TENSOR:
      return desc_.lod_tensor().lod_level();
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      return desc_.tensor_array().lod_level();
    default:
      PADDLE_THROW("Tensor type=%d does not support LoDLevel",
                   desc_.tensor_array().lod_level());
  }
}

const proto::TensorDesc &VarDesc::tensor_desc() const {
  PADDLE_ENFORCE(desc_.has_type(), "invoke TensorDesc must after set type");
  switch (desc_.type()) {
    case proto::VarDesc::SELECTED_ROWS:
      return desc_.selected_rows();
    case proto::VarDesc::LOD_TENSOR:
      return desc_.lod_tensor().tensor();
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      return desc_.tensor_array().tensor();
    default:
      PADDLE_THROW("The type of var '", this->Name(), "' is unsupported.");
  }
}

proto::TensorDesc *VarDesc::mutable_tensor_desc() {
  PADDLE_ENFORCE(desc_.has_type(),
                 "invoke MutableTensorDesc must after set type");
  switch (desc_.type()) {
    case proto::VarDesc::SELECTED_ROWS:
      return desc_.mutable_selected_rows();
    case proto::VarDesc::LOD_TENSOR:
      return desc_.mutable_lod_tensor()->mutable_tensor();
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      return desc_.mutable_tensor_array()->mutable_tensor();
    default:
      PADDLE_THROW("Unexpected branch.");
  }
}

}  // namespace framework
}  // namespace mypaddle
}  // namespace bubblefs