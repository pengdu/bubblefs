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

// Paddle/paddle/framework/block_desc.h
// Paddle/paddle/framework/program_desc.h
// Paddle/paddle/framework/op_info.h
// Paddle/paddle/framework/op_desc.h
// Paddle/paddle/framework/operator.h
// Paddle/paddle/framework/op_registry.h
// // Paddle/paddle/framework/operator.cc
// Paddle/paddle/framework/op_registry.cc

#pragma once

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "utils/paddle_scope.h"
#include "utils/paddle_selected_rows.h"
#include "utils/paddle_tensor.h"
#include "utils/paddle_type_def.h"
#include "utils/paddle_device_context.h"
#include "utils/paddle_framework_proto.h"

namespace bubblefs {
namespace mypaddle {
namespace framework {
  
class BlockDesc;
class ProgramDesc;

// Each Protobuf Message, we provide a XXXBind class. In that class, we optimize
// read/write speed. Only when we want the protobuf message, the local changes
// will be synchronized (by `Sync` method).

class BlockDesc {
 public:
  BlockDesc(ProgramDesc *prog, proto::BlockDesc *desc);

  BlockDesc(const BlockDesc &other, proto::BlockDesc *desc, ProgramDesc *prog);

  ~BlockDesc() {
    this->ClearPBVars();
    this->ClearPBOps();
  }

  int32_t ID() const { return desc_->idx(); }

  int32_t Parent() const { return desc_->parent_idx(); }

  VarDesc *Var(const std::string &name_bytes);

  VarDesc *FindVar(const std::string &name_bytes) const;

  bool HasVar(const std::string &var_name) const;

  VarDesc *FindVarRecursive(const std::string &name_bytes) const;

  VarDesc *FindRecursiveOrCreateVar(const std::string &name_bytes);

  bool HasVarRecursive(const std::string &var_name) const;

  std::set<std::string> LocalVarNames() const {
    std::set<std::string> var_names;
    for (auto &var : vars_) {
      var_names.insert(var.first);
    }
    return var_names;
  }

  std::vector<VarDesc *> AllVars() const;

  BlockDesc *ParentBlock() const;

  OpDesc *AppendOp();

  void AppendAllocatedOp(std::unique_ptr<OpDesc> &&op_desc);

  OpDesc *PrependOp();

  void RemoveOp(size_t s, size_t e);

  std::vector<OpDesc *> AllOps() const;

  size_t OpSize() const { return ops_.size(); }

  OpDesc *Op(int idx) { return ops_.at(idx).get(); }

  void Flush();

  proto::BlockDesc *Proto();

  ProgramDesc *Program() { return this->prog_; }

 private:
  void ClearPBOps();
  void ClearPBVars();

 private:
  ProgramDesc *prog_;       // not_own
  proto::BlockDesc *desc_;  // not_own
  bool need_update_;

  std::deque<std::unique_ptr<OpDesc>> ops_;
  std::unordered_map<std::string, std::unique_ptr<VarDesc>> vars_;

  DISABLE_COPY_AND_ASSIGN(BlockDesc);
};

VarDesc *BlockDesc::Var(const std::string &name) {
  auto it = vars_.find(name);
  if (it != vars_.end()) {
    return it->second.get();
  }
  need_update_ = true;
  auto *var = new VarDesc(name);
  vars_[name].reset(var);
  return var;
}

VarDesc *BlockDesc::FindVar(const std::string &name) const {
  auto it = vars_.find(name);
  if (it == vars_.end()) {
    return nullptr;
  }
  return it->second.get();
}

bool BlockDesc::HasVar(const std::string &name) const {
  return vars_.find(name) != vars_.end();
}

VarDesc *BlockDesc::FindVarRecursive(const std::string &name) const {
  if (name == kEmptyVarName) return nullptr;

  auto it = vars_.find(name);
  if (it == vars_.end()) {
    return Parent() == kNoneBlockIndex ? nullptr
                                       : ParentBlock()->FindVarRecursive(name);
  }
  return it->second.get();
}

VarDesc *BlockDesc::FindRecursiveOrCreateVar(const std::string &name_bytes) {
  VarDesc *res = FindVarRecursive(name_bytes);
  if (res == nullptr) {
    res = Var(name_bytes);
  }
  return res;
}

bool BlockDesc::HasVarRecursive(const std::string &name) const {
  return FindVarRecursive(name) != nullptr;
}

std::vector<VarDesc *> BlockDesc::AllVars() const {
  std::vector<VarDesc *> res;
  for (const auto &p : vars_) {
    res.push_back(p.second.get());
  }
  return res;
}

OpDesc *BlockDesc::AppendOp() {
  need_update_ = true;
  ops_.emplace_back(new OpDesc());
  return ops_.back().get();
}

void BlockDesc::AppendAllocatedOp(std::unique_ptr<OpDesc> &&op_desc) {
  need_update_ = true;
  ops_.emplace_back(std::move(op_desc));
}

OpDesc *BlockDesc::PrependOp() {
  need_update_ = true;
  ops_.emplace_front(new OpDesc());
  return ops_.front().get();
}

void BlockDesc::RemoveOp(size_t s, size_t e) {
  if (ops_.begin() + s == ops_.end() || ops_.begin() + e == ops_.end()) {
    return;
  }
  need_update_ = true;
  for (auto it = ops_.begin() + s; it != ops_.begin() + e; it++) {
    auto names = (*it)->InputArgumentNames();
    for (auto n : names) {
      // TODO(typhoonzero): delete vars if no other op use it.
      VLOG(3) << "deleting var " << n;
    }
  }
  ops_.erase(ops_.begin() + s, ops_.begin() + e);
}

std::vector<OpDesc *> BlockDesc::AllOps() const {
  std::vector<OpDesc *> res;
  for (const auto &op : ops_) {
    res.push_back(op.get());
  }
  return res;
}

void BlockDesc::Flush() {
  for (auto &op_desc : ops_) {
    op_desc->Flush();
  }

  if (need_update_) {
    auto &op_field = *this->desc_->mutable_ops();
    this->ClearPBOps();
    op_field.Reserve(static_cast<int>(ops_.size()));
    for (auto &op_desc : ops_) {
      op_field.AddAllocated(op_desc->Proto());
    }
    auto &var_field = *this->desc_->mutable_vars();
    this->ClearPBVars();
    var_field.Reserve(static_cast<int>(vars_.size()));
    for (auto &var_desc : vars_) {
      var_field.AddAllocated(var_desc.second->Proto());
    }
    need_update_ = false;
  }
}

BlockDesc *BlockDesc::ParentBlock() const {
  if (this->desc_->parent_idx() == kNoneBlockIndex) {
    return nullptr;
  }
  return prog_->MutableBlock(static_cast<size_t>(this->desc_->parent_idx()));
}

proto::BlockDesc *BlockDesc::Proto() {
  Flush();
  return desc_;
}

BlockDesc::BlockDesc(ProgramDesc *prog, proto::BlockDesc *desc)
    : prog_(prog), desc_(desc), need_update_(false) {
  for (const proto::VarDesc &var_desc : desc_->vars()) {
    vars_[var_desc.name()].reset(new VarDesc(var_desc));
  }
  for (const proto::OpDesc &op_desc : desc_->ops()) {
    ops_.emplace_back(new OpDesc(op_desc, prog));
  }
}

BlockDesc::BlockDesc(const BlockDesc &other, proto::BlockDesc *desc,
                     ProgramDesc *prog)
    : prog_(prog), desc_(desc) {
  need_update_ = true;
  for (auto &op : other.ops_) {
    ops_.emplace_back(new OpDesc(*op));
  }

  for (auto &it : other.vars_) {
    auto *var = new VarDesc(*it.second);
    vars_[it.first].reset(var);
  }
}

void BlockDesc::ClearPBOps() {
  auto ops = this->desc_->mutable_ops();
  while (!ops->empty()) {
    // we do not own the OpDesc, so release the ownership.
    ops->ReleaseLast();
  }
}

void BlockDesc::ClearPBVars() {
  auto vars = this->desc_->mutable_vars();
  while (!vars->empty()) {
    // we do not own the VarDesc, so release the ownership.
    vars->ReleaseLast();
  }
}

class ProgramDesc {
 public:
  ProgramDesc();

  explicit ProgramDesc(const proto::ProgramDesc &desc);

  ProgramDesc(const ProgramDesc &o);

  explicit ProgramDesc(const std::string &binary_str);

  BlockDesc *AppendBlock(const BlockDesc &parent);

  BlockDesc *MutableBlock(size_t idx) { return blocks_[idx].get(); }

  const BlockDesc &Block(size_t idx) const { return *blocks_[idx]; }

  size_t Size() const { return blocks_.size(); }

  proto::ProgramDesc *Proto();

 private:
  proto::ProgramDesc desc_;

  std::vector<std::unique_ptr<BlockDesc>> blocks_;
};

BlockDesc *ProgramDesc::AppendBlock(const BlockDesc &parent) {
  auto *b = desc_.add_blocks();
  b->set_parent_idx(parent.ID());
  b->set_idx(desc_.blocks_size() - 1);
  blocks_.emplace_back(new BlockDesc(this, b));
  return blocks_.back().get();
}

proto::ProgramDesc *ProgramDesc::Proto() {
  for (auto &block : blocks_) {
    block->Flush();
  }
  return &desc_;
}

ProgramDesc::ProgramDesc() {
  auto *block = desc_.mutable_blocks()->Add();
  block->set_idx(kRootBlockIndex);
  block->set_parent_idx(kNoneBlockIndex);
  blocks_.emplace_back(new BlockDesc(this, block));
}

ProgramDesc::ProgramDesc(const ProgramDesc &o) {
  desc_ = o.desc_;

  for (int i = 0; i < desc_.blocks_size(); ++i) {
    auto *block = desc_.mutable_blocks(i);
    blocks_.emplace_back(new BlockDesc(*o.blocks_[i], block, this));
  }
}

ProgramDesc::ProgramDesc(const proto::ProgramDesc &desc) {
  desc_ = desc;
  for (auto &block_desc : *desc_.mutable_blocks()) {
    blocks_.emplace_back(new BlockDesc(this, &block_desc));
  }
}

ProgramDesc::ProgramDesc(const std::string &binary_str) {
  PADDLE_ENFORCE(desc_.ParseFromString(binary_str),
                 "Fail to parse program_desc from binary string.");
  for (auto &block_desc : *desc_.mutable_blocks()) {
    blocks_.emplace_back(new BlockDesc(this, &block_desc));
  }
}

class InferShapeBase {
 public:
  virtual ~InferShapeBase() = default;
  virtual void operator()(InferShapeContext*) const = 0;
};

struct OpInfo {
  OpCreator creator_;
  GradOpMakerFN grad_op_maker_;
  proto::OpProto* proto_{nullptr};
  OpAttrChecker* checker_{nullptr};
  InferVarTypeFN infer_var_type_;
  InferShapeFN infer_shape_;

  bool HasOpProtoAndChecker() const {
    return proto_ != nullptr && checker_ != nullptr;
  }

  const proto::OpProto& Proto() const {
    PADDLE_ENFORCE_NOT_NULL(proto_, "Operator Proto has not been registered");
    PADDLE_ENFORCE(proto_->IsInitialized(),
                   "Operator Proto must be initialized in op info");
    return *proto_;
  }

  const OpCreator& Creator() const {
    PADDLE_ENFORCE_NOT_NULL(creator_,
                            "Operator Creator has not been registered");
    return creator_;
  }

  const GradOpMakerFN& GradOpMaker() const {
    PADDLE_ENFORCE_NOT_NULL(grad_op_maker_,
                            "Operator GradOpMaker has not been registered.");
    return grad_op_maker_;
  }

  const OpAttrChecker* Checker() const { return checker_; }
};

class OpInfoMap {
 public:
  static OpInfoMap& Instance();

  bool Has(const std::string& op_type) const {
    return map_.find(op_type) != map_.end();
  }

  void Insert(const std::string& type, const OpInfo& info) {
    PADDLE_ENFORCE(!Has(type), "Operator %s has been registered", type);
    map_.insert({type, info});
  }

  const OpInfo& Get(const std::string& type) const {
    auto op_info_ptr = GetNullable(type);
    PADDLE_ENFORCE_NOT_NULL(op_info_ptr, "Operator %s has not been registered",
                            type);
    return *op_info_ptr;
  }

  const OpInfo* GetNullable(const std::string& type) const {
    auto it = map_.find(type);
    if (it == map_.end()) {
      return nullptr;
    } else {
      return &it->second;
    }
  }

  const std::unordered_map<std::string, OpInfo>& map() const { return map_; }

  std::unordered_map<std::string, OpInfo>* mutable_map() { return &map_; }

 private:
  OpInfoMap() = default;
  std::unordered_map<std::string, OpInfo> map_;

  DISABLE_COPY_AND_ASSIGN(OpInfoMap);
};

class OpDesc {
 public:
  OpDesc() {}

  OpDesc(const std::string &type, const VariableNameMap &inputs,
         const VariableNameMap &outputs, const AttributeMap &attrs);

  OpDesc(const proto::OpDesc &desc, ProgramDesc *prog);

  void CopyFrom(const OpDesc &op_desc);

  proto::OpDesc *Proto();

  std::string Type() const { return desc_.type(); }

  void SetType(const std::string &type) { desc_.set_type(type); }

  const std::vector<std::string> &Input(const std::string &name) const;

  std::vector<std::string> InputArgumentNames() const;

  void SetInput(const std::string &param_name,
                const std::vector<std::string> &args);

  const std::vector<std::string> &Output(const std::string &name) const;

  std::vector<std::string> OutputArgumentNames() const;

  void SetOutput(const std::string &param_name,
                 const std::vector<std::string> &args);

  bool HasAttr(const std::string &name) const {
    return attrs_.find(name) != attrs_.end();
  }

  proto::AttrType GetAttrType(const std::string &name) const;

  std::vector<std::string> AttrNames() const;

  void SetAttr(const std::string &name, const Attribute &v);

  void SetBlockAttr(const std::string &name, BlockDesc &block);

  Attribute GetAttr(const std::string &name) const;

  int GetBlockAttr(const std::string &name) const;

  void Rename(const std::string &old_name, const std::string &new_name);

  void RenameOutput(const std::string &old_name, const std::string &new_name);

  void RenameInput(const std::string &old_name, const std::string &new_name);

  // Only be used in C++
  const AttributeMap &GetAttrMap() const;

  // Only be used in C++
  void SetAttrMap(const AttributeMap &attr_map);

  std::vector<std::string> InputNames() const { return MapKeys(inputs_); }
  std::vector<std::string> OutputNames() const { return MapKeys(outputs_); }

  void SetInputMap(const VariableNameMap &input) {
    this->inputs_ = input;
    this->need_update_ = true;
  }

  void SetOutputMap(const VariableNameMap &output) {
    this->outputs_ = output;
    this->need_update_ = true;
  }

  const VariableNameMap &Inputs() const { return inputs_; }

  const VariableNameMap &Outputs() const { return outputs_; }

  AttributeMap *MutableAttrMap() {
    this->need_update_ = true;
    return &this->attrs_;
  }

  void CheckAttrs();

  void InferShape(const BlockDesc &block) const;

  void InferVarType(BlockDesc *block) const;

  void MarkAsTarget() { desc_.set_is_target(true); }

  void Flush();

 private:
  template <typename MapType>
  static std::vector<typename MapType::key_type> MapKeys(const MapType &map) {
    std::vector<typename MapType::key_type> ret_val;
    ret_val.reserve(map.size());
    std::transform(
        map.begin(), map.end(), std::back_inserter(ret_val),
        [](const typename MapType::value_type &pair) { return pair.first; });
    return ret_val;
  }

  proto::OpDesc desc_;
  // input arg name => output variable names
  VariableNameMap inputs_;
  // output arg name => output variable names
  VariableNameMap outputs_;
  AttributeMap attrs_;

  // need_update_ indicate there some local changes not be synchronized. If
  // local changes should be synchronized, need_update_ should be set to true.
  bool need_update_{false};
};  
  
/// If a variable is a empty variable, that name will be used.
constexpr char kEmptyVarName[] = "@EMPTY@";

/// If a variable is a temporary variable, that name will be set in Python,
/// but it will be convert to a unique name in scope after OpCreator.
constexpr char kTempVarName[] = "@TEMP@";

/// If a variable's name has a certain suffix, it means that the
/// variable is the gradient of another varibale.
/// e.g. Variable "x@GRAD" is the gradient of varibale "x".
constexpr char kGradVarSuffix[] = "@GRAD";

/// Variables with this suffix are supposed to be filled up with zeros.
constexpr char kZeroVarSuffix[] = "@ZERO";

// define some kernel priority
extern std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority;

/**
 * @brief Use cpu kernel only
 */
void UseCPU();

/**
 * @brief Perfer MKLDNN kernel than Plain CPU kernel
 */
void UseMKLDNN();

/**
 * @brief Perfer CUDA kernel than Plain CPU kernel
 */
void UseCUDA();

/**
 * @brief Perfer cudnn kernel than Plain CUDA kernel
 */
void UseCUDNN();

/**
 * @brief Use all available kernels
 */
void UseALL();

inline std::string GradVarName(const std::string& var_name) {
  return var_name + kGradVarSuffix;
}

class OperatorBase;
class ExecutionContext;

/**
 * OperatorBase has the basic element that Net will call to do computation.
 * Only CreateOperator from OpRegistry will new Operator directly. User
 * should always construct a proto message OpDesc and call
 * OpRegistry::CreateOp(op_desc) to get an Operator instance.
 */
class OperatorBase {
 public:
  OperatorBase(const std::string& type, const VariableNameMap& inputs,
               const VariableNameMap& outputs, const AttributeMap& attrs);

  virtual ~OperatorBase() {}

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    PADDLE_ENFORCE(attrs_.count(name) != 0, "%s should be in AttributeMap",
                   name);
    return boost::get<T>(attrs_.at(name));
  }

  virtual std::string DebugString() const;

  /// Net will call this function to Run an op.
  virtual void Run(const Scope& scope, const platform::Place& place) const = 0;

  // FIXME(typhoonzero): this is only used for recv_op to stop event_loop.
  virtual void Stop() {}

  virtual bool IsNetOp() const { return false; }

  virtual bool SupportGPU() const { return false; }

  /// rename inputs outputs name
  void Rename(const std::string& old_name, const std::string& new_name);

  const VariableNameMap& Inputs() const { return inputs_; }
  const VariableNameMap& Outputs() const { return outputs_; }

  //! Get a input with argument's name described in `op_proto`
  std::string Input(const std::string& name) const;
  //! Get a input which has multiple variables.
  const std::vector<std::string>& Inputs(const std::string& name) const;

  std::vector<std::string> InputVars() const;

  //! Get a output with argument's name described in `op_proto`
  std::string Output(const std::string& name) const;
  //! Get an output which has multiple variables.
  //! TODO add a vector_view to prevent memory copy.
  const std::vector<std::string>& Outputs(const std::string& name) const;

  virtual std::vector<std::string> OutputVars(bool has_intermediate) const;

  const std::string& Type() const { return type_; }
  void SetType(const std::string& type) { type_ = type; }
  const AttributeMap& Attrs() const { return attrs_; }

  // Return a new operator instance, which is as same as this.
  // Use unique_ptr to prevent caller forget to delete this pointer.
  virtual std::unique_ptr<OperatorBase> Clone() const = 0;

 protected:
  std::string type_;
  // NOTE: in case of OpGrad, inputs_ contains:
  // I (Inputs)
  // O (Outputs)
  // OG (Output Gradients)
  VariableNameMap inputs_;

  // NOTE: in case of OpGrad, outputs_ contains
  // IG (Inputs Gradients)
  VariableNameMap outputs_;
  AttributeMap attrs_;

 private:
  void GenerateTemporaryNames();
  void CheckAllInputOutputSet() const;
};

// Macro for define a clone method.
// If you are writing an kernel operator, `Clone` will be defined when you
// register it. i.e. `Clone` method is not needed to define by yourself.
#define DEFINE_OP_CLONE_METHOD(cls)                                            \
  std::unique_ptr<::paddle::framework::OperatorBase> Clone() const final {     \
    return std::unique_ptr<::paddle::framework::OperatorBase>(new cls(*this)); \
  }

// Macro for define a default constructor for Operator.
// You can also use
//   using PARENT_CLASS::PARENT_CLASS;
// to use parent's constructor.
#define DEFINE_OP_CONSTRUCTOR(cls, parent_cls)             \
  cls(const std::string& type,                             \
      const ::paddle::framework::VariableNameMap& inputs,  \
      const ::paddle::framework::VariableNameMap& outputs, \
      const paddle::framework::AttributeMap& attrs)        \
      : parent_cls(type, inputs, outputs, attrs) {}

class NOP : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;
  void Run(const Scope& scope, const platform::Place& place) const override {}
  std::unique_ptr<OperatorBase> Clone() const override {
    return std::unique_ptr<OperatorBase>(new NOP(*this));
  }
};

class ExecutionContext {
 public:
  ExecutionContext(const OperatorBase& op, const Scope& scope,
                   const platform::DeviceContext& device_context)
      : op_(op), scope_(scope), device_context_(device_context) {}

  const OperatorBase& op() const { return op_; }

  const Scope& scope() const { return scope_; }

  template <typename T>
  inline const T& Attr(const std::string& name) const {
    return op_.Attr<T>(name);
  }

  size_t InputSize(const std::string& name) const {
    return op_.Inputs(name).size();
  }

  size_t OutputSize(const std::string& name) const {
    return op_.Outputs(name).size();
  }

  const Variable* InputVar(const std::string& name) const {
    auto ipt = op_.Input(name);
    return ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
  }

  Variable* OutputVar(const std::string& name) const {
    auto opt = op_.Output(name);
    return opt == kEmptyVarName ? nullptr : scope_.FindVar(opt);
  }

  const std::vector<const Variable*> MultiInputVar(
      const std::string& name) const {
    auto names = op_.Inputs(name);
    std::vector<const Variable*> res;
    res.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(res),
                   [this](const std::string& name) {
                     return name == kEmptyVarName ? nullptr
                                                  : scope_.FindVar(name);
                   });
    return res;
  }

  std::vector<Variable*> MultiOutputVar(const std::string& name) const {
    auto names = op_.Outputs(name);
    std::vector<Variable*> res;
    res.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(res),
                   [this](const std::string& name) {
                     return name == kEmptyVarName ? nullptr
                                                  : scope_.FindVar(name);
                   });
    return res;
  }

  template <typename T>
  const T* Input(const std::string& name) const {
    auto* var = InputVar(name);
    return var == nullptr ? nullptr : &var->Get<T>();
  }

  template <typename T>
  T* Output(const std::string& name) const {
    auto var = OutputVar(name);
    return var == nullptr ? nullptr : var->GetMutable<T>();
  }

  template <typename T>
  const std::vector<const T*> MultiInput(const std::string& name) const {
    auto names = op_.Inputs(name);
    std::vector<const T*> res;
    res.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(res),
                   [&](const std::string& sub_name) {
                     auto var = scope_.FindVar(sub_name);
                     return var == nullptr ? nullptr : &var->Get<T>();
                   });
    return res;
  }

  template <typename T>
  std::vector<T*> MultiOutput(const std::string& name) const {
    auto names = op_.Outputs(name);
    std::vector<T*> res;
    res.reserve(names.size());
    std::transform(names.begin(), names.end(), std::back_inserter(res),
                   [&](const std::string& sub_name) {
                     auto var = scope_.FindVar(sub_name);
                     return var == nullptr ? nullptr : var->GetMutable<T>();
                   });
    return res;
  }

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const {
    PADDLE_ENFORCE_LT(i, InputSize(in));
    PADDLE_ENFORCE_LT(j, OutputSize(out));
    auto* in_var = MultiInputVar(in)[i];
    auto* out_var = MultiOutputVar(out)[j];
    if (!in_var->IsType<LoDTensor>()) return;
    PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
                   "The %d-th output of Output(%s) must be LoDTensor.", j, out);
    auto in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_lod(in_tensor.lod());
  }

  platform::Place GetPlace() const { return device_context_.GetPlace(); }

  template <typename DeviceContextType>
  const DeviceContextType& device_context() const {
    return *reinterpret_cast<const DeviceContextType*>(&device_context_);
  }

  const platform::DeviceContext& device_context() const {
    return device_context_;
  }

#ifdef PADDLE_WITH_CUDA
  const inline platform::CUDADeviceContext& cuda_device_context() const {
    PADDLE_ENFORCE(platform::is_gpu_place(device_context_.GetPlace()));
    return *reinterpret_cast<const platform::CUDADeviceContext*>(
        &device_context_);
  }
#endif

  //! Get actual name vector for this input.
  const std::vector<std::string>& Inputs(const std::string& name) const {
    return op_.Inputs(name);
  }

  //! Get actual name vector for this output.
  const std::vector<std::string>& Outputs(const std::string& name) const {
    return op_.Outputs(name);
  }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
  const platform::DeviceContext& device_context_;
};

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const;

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const;

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const;

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const;

class OpKernelBase {
 public:
  /**
   * ExecutionContext is the only parameter of Kernel Run function.
   * Run will get input/output variables, state such as momentum and
   * device resource such as CUDA stream, cublas handle, etc. from
   * ExecutionContext. User should construct it before run the Operator.
   */

  virtual void Compute(const ExecutionContext& context) const = 0;

  virtual ~OpKernelBase() = default;
};

template <typename T>
class OpKernel : public OpKernelBase {
 public:
  using ELEMENT_TYPE = T;
};

class OperatorWithKernel : public OperatorBase {
 public:
  using OpKernelMap =
      std::unordered_map<OpKernelType, std::unique_ptr<OpKernelBase>,
                         OpKernelType::Hash>;

  OperatorWithKernel(const std::string& type, const VariableNameMap& inputs,
                     const VariableNameMap& outputs, const AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void Run(const Scope& scope, const platform::Place& place) const final;

  static std::unordered_map<std::string /* op_type */, OpKernelMap>&
  AllOpKernels() {
    static std::unordered_map<std::string, OpKernelMap> g_all_op_kernels;
    return g_all_op_kernels;
  }

  bool SupportGPU() const override {
    auto& op_kernels = OperatorWithKernel::AllOpKernels().at(type_);
    return std::any_of(op_kernels.begin(), op_kernels.end(),
                       [](OpKernelMap::const_reference kern_pair) {
                         return platform::is_gpu_place(kern_pair.first.place_);
                       });
  }

  virtual void InferShape(InferShapeContext* ctx) const {
    OpInfoMap::Instance().Get(Type()).infer_shape_(ctx);
  }

 protected:
  virtual OpKernelType GetActualKernelType(const ExecutionContext& ctx) const;
  virtual OpKernelType GetExpectedKernelType(
      const OpKernelType& actual_kernel_type) const;

 private:
  // indicate kernel DataType by input data. Defaultly all input data must be
  // same.
  proto::DataType IndicateDataType(const ExecutionContext& ctx) const;
};

extern bool OpSupportGPU(const std::string& op_type);

class Registrar {
 public:
  // In our design, various kinds of classes, e.g., operators and kernels,
  // have their corresponding registry and registrar. The action of
  // registration is in the constructor of a global registrar variable, which
  // are not used in the code that calls package framework, and would
  // be removed from the generated binary file by the linker. To avoid such
  // removal, we add Touch to all registrar classes and make USE_OP macros to
  // call this method. So, as long as the callee code calls USE_OP, the global
  // registrar variable won't be removed by the linker.
  void Touch() {}
};

template <typename... ARGS>
struct OperatorRegistrar : public Registrar {
  explicit OperatorRegistrar(const char* op_type) {
    PADDLE_ENFORCE(!OpInfoMap::Instance().Has(op_type),
                   "'%s' is registered more than once.", op_type);
    static_assert(sizeof...(ARGS) != 0,
                  "OperatorRegistrar should be invoked at least by OpClass");
    OpInfo info;
    details::OperatorRegistrarRecursive<0, false, ARGS...>(op_type, &info);
    OpInfoMap::Instance().Insert(op_type, info);
  }
};

class OpRegistry {
 public:
  static std::unique_ptr<OperatorBase> CreateOp(const std::string& type,
                                                const VariableNameMap& inputs,
                                                const VariableNameMap& outputs,
                                                AttributeMap attrs);

  static std::unique_ptr<OperatorBase> CreateOp(const proto::OpDesc& op_desc);

  static std::unique_ptr<OperatorBase> CreateOp(const OpDesc& op_desc);
};

template <typename PlaceType, bool at_end, size_t I, typename... KernelType>
struct OpKernelRegistrarFunctor;

template <typename PlaceType, size_t I, typename... KernelTypes>
struct OpKernelRegistrarFunctor<PlaceType, false, I, KernelTypes...> {
  using KERNEL_TYPE =
      typename std::tuple_element<I, std::tuple<KernelTypes...>>::type;

  void operator()(const char* op_type, const char* library_type) const {
    using T = typename KERNEL_TYPE::ELEMENT_TYPE;
    OpKernelType key(ToDataType(std::type_index(typeid(T))), PlaceType(),
                     DataLayout::kAnyLayout, StringToLibraryType(library_type));
    OperatorWithKernel::AllOpKernels()[op_type][key].reset(new KERNEL_TYPE);

    constexpr auto size = std::tuple_size<std::tuple<KernelTypes...>>::value;
    OpKernelRegistrarFunctor<PlaceType, I + 1 == size, I + 1, KernelTypes...>
        func;
    func(op_type, library_type);
  }
};

template <typename PlaceType, size_t I, typename... KernelType>
struct OpKernelRegistrarFunctor<PlaceType, true, I, KernelType...> {
  void operator()(const char* op_type, const char* library_type) const {}
};

// User can register many kernel in one place. The data type could be different.
template <typename PlaceType, typename... KernelType>
class OpKernelRegistrar : public Registrar {
 public:
  explicit OpKernelRegistrar(const char* op_type, const char* library_type) {
    OpKernelRegistrarFunctor<PlaceType, false, 0, KernelType...> func;
    func(op_type, library_type);
  }
};

std::vector<std::tuple<platform::Place, LibraryType>> kKernelPriority;

void UseCPU() {
  kKernelPriority.clear();
  /*Plain CPU*/
  auto pair0 = std::make_tuple(platform::CPUPlace(), LibraryType::kPlain);
  kKernelPriority.insert(kKernelPriority.begin(), pair0);
}

void UseMKLDNN() {
  UseCPU();
#if PADDLE_WITH_MKLML
  {
    /*MKLDNN Kernel*/
    auto pair0 = std::make_tuple(platform::CPUPlace(), LibraryType::kMKLDNN);
    kKernelPriority.insert(kKernelPriority.begin(), pair0);
  }
#endif
}

void UseCUDA() {
  UseMKLDNN();
#if PADDLE_WITH_CUDA
  /*Plain GPU*/
  auto pair0 = std::make_tuple(platform::CUDAPlace(0), LibraryType::kPlain);
  kKernelPriority.insert(kKernelPriority.begin(), pair0);
#endif
}

void UseCUDNN() {
  UseCUDA();
#if PADDLE_WITH_CUDA
  if (platform::dynload::HasCUDNN()) {
    /*CUDNN Kernel*/
    auto pair0 = std::make_tuple(platform::CUDAPlace(0), LibraryType::kCUDNN);
    kKernelPriority.insert(kKernelPriority.begin(), pair0);
  }
#endif
}

void UseALL() {
  UseCPU();
  UseMKLDNN();
  UseCUDA();
  UseCUDNN();
}

std::string OperatorBase::Input(const std::string& name) const {
  auto& ins = Inputs(name);
  PADDLE_ENFORCE_LE(ins.size(), 1UL,
                    "Operator %s's input %s should contain only one variable.",
                    type_, name);
  return ins.empty() ? kEmptyVarName : ins[0];
}

const std::vector<std::string>& OperatorBase::Inputs(
    const std::string& name) const {
  auto it = inputs_.find(name);
  PADDLE_ENFORCE(it != inputs_.end(), "Operator %s does not have the input %s.",
                 type_, name);
  return it->second;
}

std::string OperatorBase::Output(const std::string& name) const {
  auto& outs = Outputs(name);
  PADDLE_ENFORCE_LE(outs.size(), 1UL,
                    "Operator %s's output %s should contain only one variable.",
                    type_, name);
  return outs.empty() ? kEmptyVarName : outs[0];
}

const std::vector<std::string>& OperatorBase::Outputs(
    const std::string& name) const {
  auto it = outputs_.find(name);
  PADDLE_ENFORCE(it != outputs_.end(),
                 "Operator %s does not have an output called %s.", type_, name);
  return it->second;
}

std::string OperatorBase::DebugString() const {
  std::stringstream ss;
  ss << "Op(" << type_ << "), inputs:{";
  for (auto it = inputs_.begin(); it != inputs_.end();) {
    auto& input = *it;
    ss << input.first << "[";
    for (size_t i = 0; i < input.second.size(); ++i) {
      ss << input.second[i];
      if (i != input.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != inputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}, outputs:{";
  for (auto it = outputs_.begin(); it != outputs_.end();) {
    auto& output = *it;
    ss << output.first << "[";
    for (size_t i = 0; i < output.second.size(); ++i) {
      ss << output.second[i];
      if (i != output.second.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    ++it;
    if (it != outputs_.end()) {
      ss << ", ";
    }
  }
  ss << "}.";
  return ss.str();
}

void OperatorBase::Rename(const std::string& old_name,
                          const std::string& new_name) {
  for (auto& input : inputs_) {
    std::replace(input.second.begin(), input.second.end(), old_name, new_name);
  }
  for (auto& output : outputs_) {
    std::replace(output.second.begin(), output.second.end(), old_name,
                 new_name);
  }
}

OperatorBase::OperatorBase(const std::string& type,
                           const VariableNameMap& inputs,
                           const VariableNameMap& outputs,
                           const AttributeMap& attrs)
    : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attrs) {
  GenerateTemporaryNames();
  CheckAllInputOutputSet();
}

std::vector<std::string> OperatorBase::InputVars() const {
  std::vector<std::string> ret_val;
  for (auto& o : inputs_) {
    ret_val.reserve(ret_val.size() + o.second.size());
    ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
  }
  return ret_val;
}

std::vector<std::string> OperatorBase::OutputVars(bool has_intermediate) const {
  std::vector<std::string> ret_val;
  if (has_intermediate) {
    // push all outputs into ret_val
    for (auto& o : outputs_) {
      ret_val.reserve(ret_val.size() + o.second.size());
      ret_val.insert(ret_val.end(), o.second.begin(), o.second.end());
    }
    return ret_val;
  }
  auto& info = OpInfoMap::Instance().Get(Type());

  // get all OpProto::Var for outputs
  for (auto& o : info.Proto().outputs()) {
    // ignore all intermediate output
    if (o.intermediate()) continue;
    auto out = outputs_.find(o.name());
    if (out != outputs_.end()) {
      ret_val.reserve(ret_val.size() + out->second.size());
      ret_val.insert(ret_val.end(), out->second.begin(), out->second.end());
    }
  }
  return ret_val;
}

void OperatorBase::CheckAllInputOutputSet() const {
  auto& info_map = OpInfoMap::Instance();
  auto* op_info = info_map.GetNullable(Type());
  if (op_info == nullptr || op_info->proto_ == nullptr) return;

  for (auto& in : op_info->Proto().inputs()) {
    PADDLE_ENFORCE(inputs_.find(in.name()) != inputs_.end(),
                   "Type %s's input %s is not set", Type(), in.name());
  }

  for (auto& out : op_info->Proto().outputs()) {
    PADDLE_ENFORCE(outputs_.find(out.name()) != outputs_.end(),
                   "Type %s's output %s is not set", Type(), out.name());
  }
}

void OperatorBase::GenerateTemporaryNames() {
  static std::atomic<size_t> gUniqId(0UL);
  for (auto& output : outputs_) {
    for (auto& output_name : output.second) {
      if (output_name == kTempVarName) {
        output_name += type_;
        output_name += "@";
        output_name += std::to_string(gUniqId.fetch_add(1));
      }
    }
  }
}

static const Tensor* GetTensorFromVar(const Variable* var) {
  const Tensor* t = nullptr;
  if (var->IsType<LoDTensor>()) {
    t = &(var->Get<LoDTensor>());
  } else if (var->IsType<SelectedRows>()) {
    t = &(var->Get<SelectedRows>().value());
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 var->Type().name());
  }
  return t;
}

static Tensor* GetMutableTensorFromVar(Variable* var) {
  Tensor* t = nullptr;
  if (var->IsType<LoDTensor>()) {
    t = var->GetMutable<LoDTensor>();
  } else if (var->IsType<SelectedRows>()) {
    t = var->GetMutable<SelectedRows>()->mutable_value();
  } else {
    PADDLE_THROW("Variable type_id %s, expect LoDTensor/SelectedRows.",
                 var->Type().name());
  }
  return t;
}

template <>
const Tensor* ExecutionContext::Input<Tensor>(const std::string& name) const {
  auto* var = InputVar(name);
  return var == nullptr ? nullptr : GetTensorFromVar(var);
}

template <>
const std::vector<const Tensor*> ExecutionContext::MultiInput<Tensor>(
    const std::string& name) const {
  auto names = op().Inputs(name);
  std::vector<const Tensor*> res;
  res.reserve(names.size());
  std::transform(names.begin(), names.end(), std::back_inserter(res),
                 [&](const std::string& sub_name) {
                   auto var = scope_.FindVar(sub_name);
                   return var == nullptr ? nullptr : GetTensorFromVar(var);
                 });
  return res;
}

template <>
Tensor* ExecutionContext::Output<Tensor>(const std::string& name) const {
  auto var = OutputVar(name);
  return var == nullptr ? nullptr : GetMutableTensorFromVar(var);
}

template <>
std::vector<Tensor*> ExecutionContext::MultiOutput<Tensor>(
    const std::string& name) const {
  auto names = op().Outputs(name);
  std::vector<Tensor*> res;
  res.reserve(names.size());
  std::transform(names.begin(), names.end(), std::back_inserter(res),
                 [&](const std::string& sub_name) {
                   auto var = scope_.FindVar(sub_name);
                   return var == nullptr ? nullptr
                                         : GetMutableTensorFromVar(var);
                 });
  return res;
}

bool OpSupportGPU(const std::string& op_type) {
  auto& all_kernels = OperatorWithKernel::AllOpKernels();
  auto it = all_kernels.find(op_type);
  if (it == all_kernels.end()) {
    // All control operator must support GPU
    return true;
  }
  for (auto& kern_pair : it->second) {
    if (platform::is_gpu_place(kern_pair.first.place_)) {
      return true;
    }
  }
  return false;
}

class RuntimeInferShapeContext : public InferShapeContext {
 public:
  RuntimeInferShapeContext(const OperatorBase& op, const Scope& scope)
      : op_(op), scope_(scope) {}

  bool HasInput(const std::string& name) const override {
    auto& ins = Inputs(name);
    size_t length = ins.size();
    if (length == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(length, 1UL, "Input %s should have more than one inputs",
                      name);
    auto ipt = ins[0];
    auto* var = ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
    return var != nullptr;
  }

  bool HasOutput(const std::string& name) const override {
    auto& outs = Outputs(name);
    size_t length = outs.size();
    if (length == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(length, 1UL, "Output %s should have more than one inputs",
                      name);
    auto ipt = outs[0];
    auto* var = ipt == kEmptyVarName ? nullptr : scope_.FindVar(ipt);
    return var != nullptr;
  }

  bool HasInputs(const std::string& name) const override {
    auto inputs = op_.Inputs(name);
    if (inputs.empty()) {
      return false;
    }
    for (auto& input : inputs) {
      if (scope_.FindVar(input) == nullptr) {
        return false;
      }
    }
    return true;
  }

  bool HasOutputs(const std::string& name) const override {
    auto outputs = op_.Outputs(name);
    if (outputs.empty()) {
      return false;
    }
    for (auto& output : outputs) {
      if (scope_.FindVar(output) == nullptr) {
        return false;
      }
    }
    return true;
  }

  DDim GetInputDim(const std::string& name) const override {
    return GetDim(op_.Input(name));
  }

  void SetOutputDim(const std::string& name, const DDim& dim) override {
    SetDim(op_.Output(name), dim);
  }

  AttrReader Attrs() const override { return AttrReader(op_.Attrs()); }

  const std::vector<std::string>& Inputs(
      const std::string& name) const override {
    return op_.Inputs(name);
  }

  const std::vector<std::string>& Outputs(
      const std::string& name) const override {
    return op_.Outputs(name);
  }

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override {
    PADDLE_ENFORCE_LT(i, Inputs(in).size());
    PADDLE_ENFORCE_LT(j, Outputs(out).size());
    Variable* in_var = scope_.FindVar(Inputs(in)[i]);
    Variable* out_var = scope_.FindVar(Outputs(out)[j]);
    if (!in_var->IsType<LoDTensor>()) return;
    PADDLE_ENFORCE(out_var->IsType<LoDTensor>(),
                   "The %d-th output of Output(%s) must be LoDTensor.", j, out);
    auto in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_lod(in_tensor.lod());
  }

  bool IsRuntime() const override { return true; }

 protected:
  DDim GetDim(const std::string& name) const override {
    Variable* var = scope_.FindVar(name);
    if (var->IsType<LoDTensor>()) {
      return var->Get<LoDTensor>().dims();
    } else if (var->IsType<SelectedRows>()) {
      return var->Get<SelectedRows>().GetCompleteDims();
    } else {
      PADDLE_THROW("Variable %s type_id %s, expect LoDTensor/SelectedRows.",
                   name, var->Type().name());
    }
  }

  void SetDim(const std::string& name, const DDim& dim) override {
    Variable* var = scope_.FindVar(name);
    if (var->IsType<LoDTensor>()) {
      var->GetMutable<LoDTensor>()->Resize(dim);
    } else if (var->IsType<SelectedRows>()) {
      var->GetMutable<SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW("Variable %s type_id %s, expect LoDTensor/SelectedRows.",
                   name, var->Type().name());
    }
  }

  proto::VarDesc::VarType GetVarType(const std::string& name) const override {
    auto* var = scope_.FindVar(name);
    return ToVarType(var->Type());
  }

 private:
  const OperatorBase& op_;
  const Scope& scope_;
};

const platform::DeviceContext* GetDeviceContext(
    framework::KernelTypePair& kernel_pair) {
  auto& actual_kernel_key = kernel_pair.first;
  auto& expected_kernel_key = kernel_pair.second;
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

  if (platform::is_gpu_place(actual_kernel_key.place_) &&
      platform::is_cpu_place(expected_kernel_key.place_)) {
    return pool.Get(actual_kernel_key.place_);
  } else if (platform::is_cpu_place(actual_kernel_key.place_) &&
             platform::is_gpu_place(expected_kernel_key.place_)) {
    return pool.Get(expected_kernel_key.place_);
  } else {
    PADDLE_THROW(
        "Currently, model parallelism is only supported between CPU and CUDA");
  }
}

const platform::DeviceContext* GetDeviceContext(
    const framework::OpKernelType& kernel) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  return pool.Get(kernel.place_);
}

void OperatorWithKernel::Run(const Scope& scope,
                             const platform::Place& place) const {
  RuntimeInferShapeContext infer_shape_ctx(*this, scope);
  this->InferShape(&infer_shape_ctx);
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto dev_ctx = pool.Get(place);

  // check if op[type] has kernel registered.
  auto& all_op_kernels = AllOpKernels();
  auto kernels_iter = all_op_kernels.find(type_);
  if (kernels_iter == all_op_kernels.end()) {
    PADDLE_THROW(
        "There are no kernels which are registered in the %s operator.", type_);
  }

  // check if op[type] have kernel for kernel_key
  OpKernelMap& kernels = kernels_iter->second;

  ExecutionContext ctx(*this, scope, *dev_ctx);
  auto actual_kernel_key = GetActualKernelType(ctx);

  auto expected_kernel_key = GetExpectedKernelType(actual_kernel_key);

  if (actual_kernel_key == expected_kernel_key) {
    PADDLE_ENFORCE_EQ(actual_kernel_key.place_, expected_kernel_key.place_,
                      "Currently, model parallelism is only supported between "
                      "CPU and other devices. For example, multi-GPU model "
                      "parallelism will failed.");
  } else {
    // find the best key candidate
    const DataTransformFnMap& trans_map = DataTransformFnMap::Instance();
    for (auto& candidate : kKernelPriority) {
      auto candidate_key =
          OpKernelType(actual_kernel_key.data_type_, std::get<0>(candidate),
                       actual_kernel_key.data_layout_, std::get<1>(candidate));

      auto candidate_pair = std::make_pair(actual_kernel_key, candidate_key);
      if ((actual_kernel_key == candidate_key) ||
          (kernels.count(candidate_key) &&
           trans_map.GetNullable(candidate_pair))) {
        expected_kernel_key = candidate_key;
        break;
      }
    }

    auto kernel_pair = std::make_pair(actual_kernel_key, expected_kernel_key);
    const DataTransformFn* trans_fun = trans_map.GetNullable(kernel_pair);
    if (trans_fun) {
      auto input_vars = this->InputVars();
      // TODO(qijun) filter the input vars that do not need to be transformed

      // filter vars that has been transformed
      std::vector<std::string> need_trans;
      for (auto var_name : input_vars) {
        auto var_name_trans =
            var_name + framework::KernelTypeToString(expected_kernel_key);
        if (!scope.FindVar(var_name_trans)) {
          const_cast<Scope&>(scope).Var(var_name_trans);
          need_trans.push_back(var_name);
        }
      }

      if (!need_trans.empty()) {
        auto trans_dev_ctx = GetDeviceContext(kernel_pair);

        // Wait for transform starting
        dev_ctx->Wait();

        for (auto var_name : need_trans) {
          (*trans_fun)(trans_dev_ctx, kernel_pair, *(scope.FindVar(var_name)),
                       scope.FindVar(var_name + framework::KernelTypeToString(
                                                    expected_kernel_key)));
        }
        // Wait for data transform finishing
        trans_dev_ctx->Wait();
      }
    }
  }

  VLOG(10) << "Actual kernel: " << actual_kernel_key
           << "Expected kernel: " << expected_kernel_key;

  auto kernel_iter = kernels.find(expected_kernel_key);

  if (kernel_iter == kernels.end()) {
    PADDLE_THROW("The operator %s does not support %s", type_,
                 expected_kernel_key);
  }

  auto* expected_dev_ctx = GetDeviceContext(expected_kernel_key);
  ExecutionContext expected_ctx(*this, scope, *expected_dev_ctx);

  kernel_iter->second->Compute(expected_ctx);
}

OpKernelType OperatorWithKernel::GetActualKernelType(
    const ExecutionContext& ctx) const {
  return OpKernelType(IndicateDataType(ctx), ctx.GetPlace());
}

OpKernelType OperatorWithKernel::GetExpectedKernelType(
    const OpKernelType& actual_kernel_type) const {
  return actual_kernel_type;
}

proto::DataType OperatorWithKernel::IndicateDataType(
    const ExecutionContext& ctx) const {
  auto& scope = ctx.scope();
  int data_type = -1;
  for (auto& input : this->inputs_) {
    for (auto& ipt_name : input.second) {
      auto* var = scope.FindVar(ipt_name);
      if (var != nullptr) {
        const Tensor* t = nullptr;
        if (var->IsType<Tensor>()) {
          t = &var->Get<Tensor>();
        } else if (var->IsType<LoDTensor>()) {
          t = &var->Get<LoDTensor>();
        } else if (var->IsType<SelectedRows>()) {
          t = &(var->Get<SelectedRows>().value());
        }
        if (t != nullptr) {
          int tmp = static_cast<int>(ToDataType(t->type()));
          PADDLE_ENFORCE(tmp == data_type || data_type == -1,
                         "DataType of Paddle Op %s must be the same.", Type());
          data_type = tmp;
        }
      }
    }
  }
  PADDLE_ENFORCE(data_type != -1, "DataType should be indicated by input");
  return static_cast<proto::DataType>(data_type);
}

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(
    const std::string& type, const VariableNameMap& inputs,
    const VariableNameMap& outputs, AttributeMap attrs) {
  auto& info = OpInfoMap::Instance().Get(type);
  if (info.Checker() != nullptr) {
    info.Checker()->Check(attrs);
  }
  auto op = info.Creator()(type, inputs, outputs, attrs);
  return std::unique_ptr<OperatorBase>(op);
}

static VariableNameMap ConvertOpDescVarsToVarNameMap(
    const google::protobuf::RepeatedPtrField<proto::OpDesc::Var>&
        op_desc_vars) {
  VariableNameMap ret_val;
  for (auto& var : op_desc_vars) {
    auto& var_names = ret_val[var.parameter()];
    auto& var_names_in_proto = var.arguments();
    var_names.reserve(static_cast<size_t>(var_names_in_proto.size()));
    std::copy(var_names_in_proto.begin(), var_names_in_proto.end(),
              std::back_inserter(var_names));
  }
  return ret_val;
}

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(
    const proto::OpDesc& op_desc) {
  VLOG(1) << "CreateOp directly from OpDesc is deprecated. It should only be"
             "used in unit tests. Use CreateOp(const OpDesc& op_desc) "
             "instead.";
  VariableNameMap inputs = ConvertOpDescVarsToVarNameMap(op_desc.inputs());
  VariableNameMap outputs = ConvertOpDescVarsToVarNameMap(op_desc.outputs());
  AttributeMap attrs;
  for (auto& attr : op_desc.attrs()) {
    attrs[attr.name()] = GetAttrValue(attr);
  }

  return CreateOp(op_desc.type(), inputs, outputs, attrs);
}

std::unique_ptr<OperatorBase> OpRegistry::CreateOp(const OpDesc& op_desc) {
  return CreateOp(op_desc.Type(), op_desc.Inputs(), op_desc.Outputs(),
                  op_desc.GetAttrMap());
}

}  // namespace framework
}  // namespace mypaddle
}  // namespace bubblefs