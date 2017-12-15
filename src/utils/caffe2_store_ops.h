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

// caffe2/caffe2/distributed/store_ops.h
// caffe2/caffe2/distributed/store_ops.cc

#ifndef BUBBLEFS_UTILS_CAFFE2_STORE_OPS_H_
#define BUBBLEFS_UTILS_CAFFE2_STORE_OPS_H_

#include "utils/caffe2_store_handler.h"
#include "utils/caffe2_operator.h"

namespace bubblefs {
namespace mycaffe2 {

class StoreSetOp final : public Operator<CPUContext> {
 public:
  StoreSetOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::string blobName_;

  INPUT_TAGS(HANDLER, DATA);
};

class StoreGetOp final : public Operator<CPUContext> {
 public:
  StoreGetOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::string blobName_;

  INPUT_TAGS(HANDLER);
  OUTPUT_TAGS(DATA);
};

class StoreAddOp final : public Operator<CPUContext> {
 public:
  StoreAddOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::string blobName_;
  int addValue_;

  INPUT_TAGS(HANDLER);
  OUTPUT_TAGS(VALUE);
};

class StoreWaitOp final : public Operator<CPUContext> {
 public:
  StoreWaitOp(const OperatorDef& operator_def, Workspace* ws);
  bool RunOnDevice() override;

 private:
  std::vector<std::string> blobNames_;

  INPUT_TAGS(HANDLER);
};

bool StoreSetOp::RunOnDevice() {
  // Serialize and pass to store
  auto* handler =
      OperatorBase::Input<std::unique_ptr<StoreHandler>>(HANDLER).get();
  handler->set(blobName_, InputBlob(DATA).Serialize(blobName_));
  return true;
}

bool StoreGetOp::RunOnDevice() {
  // Get from store and deserialize
  auto* handler =
      OperatorBase::Input<std::unique_ptr<StoreHandler>>(HANDLER).get();
  OperatorBase::Outputs()[DATA]->Deserialize(handler->get(blobName_));
  return true;
}

bool StoreWaitOp::RunOnDevice() {
  auto* handler =
      OperatorBase::Input<std::unique_ptr<StoreHandler>>(HANDLER).get();
  if (InputSize() == 2 && Input(1).IsType<std::string>()) {
    PANIC_ENFORCE(blobNames_.empty(), "cannot specify both argument and input blob");
    std::vector<std::string> blobNames;
    auto* namesPtr = Input(1).data<std::string>();
    for (int i = 0; i < Input(1).size(); ++i) {
      blobNames.push_back(namesPtr[i]);
    }
    handler->wait(blobNames);
  } else {
    handler->wait(blobNames_);
  }
  return true;
}


} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_STORE_OPS_H_