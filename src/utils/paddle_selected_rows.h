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

// Paddle/paddle/framework/selected_rows.h
// Paddle/paddle/framework/selected_rows.cc

#pragma once

#include "utils/paddle_tensor.h"

namespace bubblefs {
namespace mypaddle {
namespace framework {

class SelectedRows {
 public:
  SelectedRows(const std::vector<int64_t>& rows, const int64_t& height)
      : rows_(rows), height_(height) {
    value_.reset(new Tensor());
  }

  SelectedRows() {
    height_ = 0;
    value_.reset(new Tensor());
  }

  platform::Place place() const { return value_->place(); }

  const Tensor& value() const { return *value_; }

  Tensor* mutable_value() { return value_.get(); }

  int64_t height() const { return height_; }

  void set_height(int64_t height) { height_ = height; }

  const Vector<int64_t>& rows() const { return rows_; }

  Vector<int64_t>* mutable_rows() { return &rows_; }

  void set_rows(const Vector<int64_t>& rows) { rows_ = rows; }

  DDim GetCompleteDims() const {
    std::vector<int64_t> dims = vectorize(value_->dims());
    dims[0] = height_;
    return make_ddim(dims);
  }

 private:
  // Notice: rows can be duplicate. We can have {0, 4, 7, 0, 5, 7, 9} here.
  // SelectedRows are simplely concated when adding together. Until a
  // SelectedRows add a Tensor, will the duplicate rows be handled.
  Vector<int64_t> rows_;
  std::unique_ptr<Tensor> value_{nullptr};
  int64_t height_;
};

/*
 * Serialize/Desiralize SelectedRows to std::ostream
 * You can pass ofstream or ostringstream to serilize to file
 * or to a in memory string. GPU tensor will be copied to CPU.
 */
void SerializeToStream(std::ostream& os, const SelectedRows& selected_rows,
                       const platform::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is, SelectedRows* selected_rows,
                           const platform::DeviceContext& dev_ctx);

void SerializeToStream(std::ostream& os, const SelectedRows& selected_rows,
                       const platform::DeviceContext& dev_ctx) {
  {  // the 1st field, uint32_t version
    constexpr uint32_t version = 0;
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
  }
  {
    // the 2st field, rows information
    auto& rows = selected_rows.rows();
    uint64_t size = rows.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (uint64_t i = 0; i < size; ++i) {
      os.write(reinterpret_cast<const char*>(&rows[i]), sizeof(rows[i]));
    }
  }
  {
    // the 3st field, the height of SelectedRows
    int64_t height = selected_rows.height();
    os.write(reinterpret_cast<const char*>(&height), sizeof(height));
  }
  // the 4st field, Tensor data
  SerializeToStream(os, selected_rows.value(), dev_ctx);
}

void DeserializeFromStream(std::istream& is, SelectedRows* selected_rows,
                           const platform::DeviceContext& dev_ctx) {
  {
    // the 1st field, unit32_t version for SelectedRows
    uint32_t version;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(version, 0U, "Only version 0 is supported");
  }
  {
    // the 2st field, rows information
    uint64_t size;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    auto& rows = *selected_rows->mutable_rows();
    rows.resize(size);
    for (uint64_t i = 0; i < size; ++i) {
      is.read(reinterpret_cast<char*>(&rows[i]), sizeof(int64_t));
    }
  }
  {
    // the 3st field, the height of the SelectedRows
    int64_t height;
    is.read(reinterpret_cast<char*>(&height), sizeof(int64_t));
    selected_rows->set_height(height);
  }
  // the 4st field, tensor which contains the data
  DeserializeFromStream(is, selected_rows->mutable_value(), dev_ctx);
}

}  // namespace framework
}  // namespace mypaddle
}  // namespace bubblefs