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

// Paddle/paddle/optimizer/tensor.h
// Paddle/paddle/optimizer/serialization.h
// Paddle/paddle/optimizer/lr_policy.h
// Paddle/paddle/optimizer/optimizer.h
// Paddle/paddle/optimizer/parameter_optimizer.h
// Paddle/paddle/optimizer/sgd_optimizer.h
// Paddle/paddle/optimizer/adam_optimizer.h
// Paddle/paddle/optimizer/adadelta_optimizer.h

#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

/**
 * @brief optimizer library in independent with other module
 * which will be used in :
 * Case A, the gradient optimized locally on the trainer.
 *
 * Case B, the gradient optimized on the parameter server.
 */

namespace bubblefs {
namespace mypaddle {
namespace optimizer {
  
typedef enum {
  PADDLE_ELEMENT_TYPE_INT32 = 0,
  PADDLE_ELEMENT_TYPE_UINT32 = 1,
  PADDLE_ELEMENT_TYPE_INT64 = 2,
  PADDLE_ELEMENT_TYPE_UINT64 = 3,
  PADDLE_ELEMENT_TYPE_FLOAT32 = 4,
  PADDLE_ELEMENT_TYPE_FLOAT64 = 5,
} paddle_element_type;

/**
 * @brief execution status code
 */
const int32_t PADDLE_SUCCESS = 0;
const int32_t PADDLE_ERROR = -1;

typedef struct paddle_optimizer paddle_optimizer;
/**
 * this group interface called in order :
 * 1. create optimizer with config
 * 2. set weights
 * 3. update_parameter
 * 4. get_weights
 * 5. release optimizer
 */

/**
 *  @brief create optimizer with proto_config
 *  @param config_proto, optimizer protobuf, see OptimizerConfig.proto in detail
 *  @return return optimizer instance
 */
paddle_optimizer* paddle_create_optimizer(const unsigned char* config_proto,
                                          const int config_proto_len,
                                          const paddle_element_type data_type,
                                          void* param_buffer,
                                          int num_bytes,
                                          const char* state,
                                          const int state_len);

/**
 *  @brief release optimizer
 *  @param optimizer
 *  @return return exec status
 */
int paddle_release_optimizer(paddle_optimizer* o);

/**
 *  @brief optimizer instance
 *  @param datatype of gradient and parameter
 *  @param gradient, calculate by optimzizer caller.
 *       TODO(zhihong): just pass loss to reduce communicate overhead.
 *                     Project Adam Ms'14 paper for detail
 *  @param num_bytes, gradient size
 *  @return return exec status
 */
int paddle_update_parameter(paddle_optimizer* o,
                            const paddle_element_type data_type,
                            const void* gradient,
                            int num_bytes);

/**
 *  @brief optimizer for get parameter buffer
 *  @param param_buffer, initilized parameter buffer
 *  @return return content length
 */
int paddle_optimizer_get_weights(paddle_optimizer* o, void** param_buffer);

/**
 *  @brief optimzizer for saving training state
 *  @param training state for receive SerializeState
 *  @return return state_buffer length
 */
int paddle_optimizer_get_state(paddle_optimizer* o, const char** state);

template <class T>
class TensorT {
public:
  TensorT(size_t size) : height_(1), width_(size) {
    // new T[size]() initializes all element to zero value.
    data_ptr_ = std::shared_ptr<T>(new T[size](), std::default_delete<T[]>());
    data_ = data_ptr_.get();
  }

  TensorT(T* data, size_t size)
      : height_(1), width_(size), data_ptr_(nullptr), data_(data) {}

  TensorT(T* data, size_t h, size_t w)
      : height_(h), width_(w), data_ptr_(nullptr), data_(data) {}

  virtual ~TensorT() {}

  T* get_buffer() { return this->data_; }

  T& operator[](const size_t idx) {
    CHECK(idx >= 0 && idx < this->width_) << "out of index range";
    return data_[idx];
  }
  T& operator[](const size_t idx) const {
    CHECK(idx >= 0 && idx < this->width_) << "out of index range";
    return data_[idx];
  }
  // TODO: replace with tensorshape
  size_t size() const { return this->width_ * this->height_; }

protected:
  size_t height_;
  size_t width_;
  std::shared_ptr<T> data_ptr_;
  T* data_;
};

static void TensorToProto(const Tensor& tensor, TensorProto* proto) {
  proto->set_data_type(TensorProto::PADDLE_ELEMENT_TYPE_FLOAT32);
  std::stringstream os;
  for (size_t i = 0; i < tensor.size(); ++i) {
    os << tensor[i];
    proto->add_content(os.str());
    os.str(std::string());
  }
}

static void ProtoToTensor(const TensorProto& proto, Tensor* tensor) {
  std::stringstream sin;
  for (auto i = 0; i < proto.content_size(); ++i) {
    sin << proto.content(i);
    sin >> (*tensor)[i];
    sin.str(std::string());
    sin.clear();
  }

// TODO(zhihong): design problem of dynamic datatype, need to fix it
typedef TensorT<float> Tensor;

class LrPolicy {
public:
  virtual ~LrPolicy() {}
  virtual double LearningRate(const uint64_t num_sample_passed) = 0;
  virtual std::string SerializeState() = 0;
  virtual void DeserializeState(const std::string &state) = 0;
};

// constant learning rate policy
class ConstLr final : public LrPolicy {
public:
  ConstLr(double lr) : learning_rate_(lr){};
  double LearningRate(const uint64_t num_sample_passed) {
    return learning_rate_;
  }
  std::string SerializeState() {
    LrPolicyState state;
    state.set_learning_rate(learning_rate_);
    return state.SerializeAsString();
  }
  void DeserializeState(const std::string &str) {
    LrPolicyState state;
    state.ParseFromString(str);
    learning_rate_ = state.learning_rate();
  }

private:
  double learning_rate_;
};

class LinearLr final : public LrPolicy {
public:
  LinearLr(double lr, double lr_decay_a, double lr_decay_b)
      : learning_rate_(lr), lr_decay_a_(lr_decay_a), lr_decay_b_(lr_decay_b) {}
  double LearningRate(const uint64_t num_sample_passed) {
    return std::max(learning_rate_ - lr_decay_a_ * num_sample_passed,
                    lr_decay_b_);
  }
  std::string SerializeState() {
    LrPolicyState state;
    state.set_learning_rate(learning_rate_);
    state.set_lr_decay_a(lr_decay_a_);
    state.set_lr_decay_b(lr_decay_b_);
    return state.SerializeAsString();
  }
  void DeserializeState(const std::string &str) {
    LrPolicyState state;
    state.ParseFromString(str);
    learning_rate_ = state.learning_rate();
    lr_decay_a_ = state.lr_decay_a();
    lr_decay_b_ = state.lr_decay_b();
  }

private:
  double learning_rate_;
  double lr_decay_a_;
  double lr_decay_b_;
};

class ParameterOptimizer {
public:
  /**
   * @brief  update hook for algorithm need to traverse parameter more than
   * once.
   */
  ParameterOptimizer(Tensor *parameter, LrPolicy *lr)
      : parameter_(parameter), lr_policy_(lr), num_sample_passed_(0) {}
  virtual ~ParameterOptimizer() {
    delete parameter_;
    delete lr_policy_;
  }

  static ParameterOptimizer *Create(const std::string &config_proto,
                                    Tensor *parameter);
  virtual void Update(const Tensor *gradient) = 0;
  virtual float *get_weight(int *param_size) const;
  virtual std::string SerializeState() = 0;
  virtual void DeserializeState(const std::string &state) = 0;

protected:
  Tensor *parameter_;
  // learning rate policy
  LrPolicy *lr_policy_;
  uint64_t num_sample_passed_;
};

ParameterOptimizer *ParameterOptimizer::Create(std::string &config_proto,
                                               Tensor *parameter) {
  paddle::OptimizerConfig config;
  CHECK(config.ParseFromString(config_proto) == true)
      << "failed parse optimizer config";
  auto select_lr_policy = [=](const OptimizerConfig &config) -> LrPolicy * {
    if (config.lr_policy() == OptimizerConfig::Const)
      return new ConstLr(config.const_lr().learning_rate());
    if (config.lr_policy() == OptimizerConfig::Linear)
      return new LinearLr(config.linear_lr().learning_rate(),
                          config.linear_lr().lr_decay_a(),
                          config.linear_lr().lr_decay_b());
    // default
    LOG(WARNING) << " have not select any LrPolicy. use ConstLr in default";
    return new ConstLr(0.1);
  };

  LrPolicy *lr = select_lr_policy(config);
  auto select_optimizer = [=](
      Tensor *parameter,
      const OptimizerConfig &config) -> ParameterOptimizer * {
    if (config.optimizer() == OptimizerConfig::SGD) {
      LOG(INFO) << "creating SGD optimizer";
      return new SGDOptimizer(parameter,
                              lr,
                              config.sgd().momentum(),
                              config.sgd().decay(),
                              config.sgd().nesterov());
    }
    if (config.optimizer() == OptimizerConfig::Adadelta) {
      LOG(INFO) << "creating Adadelta optimizer";
      return new AdadeltaOptimizer(parameter,
                                   lr,
                                   config.adadelta().rho(),
                                   config.adadelta().epsilon(),
                                   config.adadelta().decay());
    }
    if (config.optimizer() == OptimizerConfig::Adagrad) {
      LOG(INFO) << "creating Adagrad optimizer";
      return new AdagradOptimizer(
          parameter, lr, config.adagrad().epsilon(), config.adagrad().decay());
    }
    if (config.optimizer() == OptimizerConfig::Adam) {
      LOG(INFO) << "creating Adam optimizer";
      return new AdamOptimizer(parameter,
                               lr,
                               config.adam().beta_1(),
                               config.adam().beta_2(),
                               config.adam().epsilon(),
                               config.adam().decay());
    }
    // default
    LOG(WARNING)
        << "have not select any Optimizer. use SGDOptimizer in default";
    return new SGDOptimizer(parameter, lr, 0.0, 0.0, false);
  };
  return select_optimizer(parameter, config);
}

float *ParameterOptimizer::get_weight(int *param_size) const {
  *param_size = (int)parameter_->size();
  return parameter_->get_buffer();
}

class SGDOptimizer : public ParameterOptimizer {
public:
  SGDOptimizer(Tensor* parameter, LrPolicy* lr, double m, double d, bool n)
      : ParameterOptimizer(parameter, lr),
        momentums_(nullptr),
        momentum_(m),
        decay_(d),
        nesterov_(n) {
    if (momentum_ != 0.0) {
      size_t size = parameter->size();
      momentums_ = new Tensor(size);
    }
  }
  virtual ~SGDOptimizer() {
    if (momentums_) delete momentums_;
  }
  void Update(const Tensor* gradient);
  std::string SerializeState();
  void DeserializeState(const std::string& state);

private:
  Tensor* momentums_;
  double momentum_;
  double decay_;
  bool nesterov_;
};

void SGDOptimizer::Update(const Tensor *gradient) {
  num_sample_passed_ += 1;
  double learning_rate = lr_policy_->LearningRate(num_sample_passed_);
  float velocity = 0.0;
  Tensor &param = *parameter_;
  const Tensor &grad = *gradient;
  Tensor &m = *momentums_;
  for (size_t i = 0; i < param.size(); ++i) {
    if (momentum_ == 0.0) {
      velocity = -learning_rate * grad[i] - learning_rate * decay_ * param[i];
    } else {
      m[i] = momentum_ * m[i] - learning_rate * grad[i] -
             learning_rate * decay_ * param[i];
      velocity = m[i];
    }
    if (nesterov_) {
      param[i] += momentum_ * velocity - learning_rate * grad[i];
    } else {
      param[i] += velocity;
    }
  }
}

std::string SGDOptimizer::SerializeState() {
  SGDOptimizerState state;
  state.set_num_sample_passed(num_sample_passed_);
  std::string lr_str = this->lr_policy_->SerializeState();
  state.mutable_lr_state()->ParseFromString(lr_str);
  TensorToProto(*parameter_, state.mutable_parameter());
  if (momentum_ != 0.0) TensorToProto(*momentums_, state.mutable_momentums());
  return state.SerializeAsString();
}

void SGDOptimizer::DeserializeState(const std::string &str) {
  SGDOptimizerState state;
  state.ParseFromString(str);
  auto lr_state = state.lr_state();
  this->lr_policy_->DeserializeState(lr_state.SerializeAsString());
  num_sample_passed_ = state.num_sample_passed();
  ProtoToTensor(state.parameter(), parameter_);
  if (momentum_ != 0.0) ProtoToTensor(state.momentums(), momentums_);
}

class AdamOptimizer : public ParameterOptimizer {
public:
  AdamOptimizer(Tensor *parameter,
                LrPolicy *lr,
                double beta_1,
                double beta_2,
                double epsilon,
                double decay)
      : ParameterOptimizer(parameter, lr),
        momentums_(new Tensor(parameter->size())),
        velocitys_(new Tensor(parameter->size())),
        beta_1_(beta_1),
        beta_2_(beta_2),
        epsilon_(epsilon),
        decay_(decay) {}
  ~AdamOptimizer() {
    if (momentums_) delete momentums_;
    if (velocitys_) delete velocitys_;
  }
  void Update(const Tensor *gradient);
  std::string SerializeState();
  void DeserializeState(const std::string &state);

private:
  Tensor *momentums_;
  Tensor *velocitys_;
  double beta_1_;
  double beta_2_;
  double epsilon_;
  double decay_;
};

void AdamOptimizer::Update(const Tensor *gradient) {
  num_sample_passed_ += 1;
  double learning_rate = lr_policy_->LearningRate(num_sample_passed_);
  double coef1 = 1.0 - std::pow(beta_1_, num_sample_passed_);
  double coef2 = 1.0 - std::pow(beta_2_, num_sample_passed_);
  learning_rate *= std::sqrt(coef2) / coef1;
  Tensor &param = *parameter_;
  const Tensor &grad = *gradient;
  Tensor &m = *momentums_;
  Tensor &v = *velocitys_;
  for (size_t i = 0; i < param.size(); ++i) {
    m[i] = beta_1_ * m[i] + (1.0 - beta_1_) * grad[i];
    v[i] = beta_2_ * v[i] + (1.0 - beta_2_) * grad[i] * grad[i];
    param[i] -=
        learning_rate * (m[i] / std::sqrt(v[i] + epsilon_) + decay_ * param[i]);
  }
}

std::string AdamOptimizer::SerializeState() {
  AdamOptimizerState state;
  std::string lr_str = this->lr_policy_->SerializeState();
  state.mutable_lr_state()->ParseFromString(lr_str);
  state.set_num_sample_passed(num_sample_passed_);

  TensorToProto(*parameter_, state.mutable_parameter());
  TensorToProto(*momentums_, state.mutable_momentums());
  TensorToProto(*velocitys_, state.mutable_velocitys());
  return state.SerializeAsString();
}

void AdamOptimizer::DeserializeState(const std::string &str) {
  AdamOptimizerState state;
  state.ParseFromString(str);
  auto lr_state = state.lr_state();
  this->lr_policy_->DeserializeState(lr_state.SerializeAsString());
  num_sample_passed_ = state.num_sample_passed();

  ProtoToTensor(state.parameter(), parameter_);
  ProtoToTensor(state.momentums(), momentums_);
  ProtoToTensor(state.velocitys(), velocitys_);
}

class AdadeltaOptimizer : public ParameterOptimizer {
public:
  AdadeltaOptimizer(
      Tensor *parameter, LrPolicy *lr, double rho, double epsilon, double decay)
      : ParameterOptimizer(parameter, lr),
        accum_gradient_(new Tensor(parameter->size())),
        accum_delta_(new Tensor(parameter->size())),
        update_delta_(new Tensor(parameter->size())),
        rho_(rho),
        epsilon_(epsilon),
        decay_(decay) {}

  ~AdadeltaOptimizer() {
    if (accum_gradient_) delete accum_gradient_;
    if (accum_delta_) delete accum_delta_;
    if (update_delta_) delete update_delta_;
  }
  void Update(const Tensor *gradient);
  std::string SerializeState();
  void DeserializeState(const std::string &state);

private:
  Tensor *accum_gradient_;
  Tensor *accum_delta_;
  Tensor *update_delta_;
  double rho_;
  double epsilon_;
  double decay_;
};

void AdadeltaOptimizer::Update(const Tensor* gradient) {
  num_sample_passed_ += 1;
  double learning_rate = lr_policy_->LearningRate(num_sample_passed_);
  Tensor& param = *parameter_;
  const Tensor& grad = *gradient;
  Tensor& accum_g = *accum_gradient_;
  Tensor& accum_d = *accum_delta_;
  Tensor& update_d = *update_delta_;
  for (size_t i = 0; i < param.size(); ++i) {
    accum_g[i] = rho_ * accum_g[i] + (1.0 - rho_) * grad[i] * grad[i];

    update_d[i] = std::sqrt(accum_d[i] + epsilon_) /
                  std::sqrt(accum_g[i] + epsilon_) * grad[i];

    accum_d[i] = rho_ * accum_d[i] + (1.0 - rho_) * update_d[i] * update_d[i];

    param[i] -= learning_rate * update_d[i] + learning_rate * decay_ * param[i];
  }
}

std::string AdadeltaOptimizer::SerializeState() {
  AdadeltaOptimizerState state;
  state.set_num_sample_passed(num_sample_passed_);
  std::string lr_str = this->lr_policy_->SerializeState();
  state.mutable_lr_state()->ParseFromString(lr_str);

  TensorToProto(*parameter_, state.mutable_parameter());
  TensorToProto(*accum_gradient_, state.mutable_accum_gradient());
  TensorToProto(*accum_delta_, state.mutable_accum_delta());
  TensorToProto(*update_delta_, state.mutable_update_delta());
  return state.SerializeAsString();
}

void AdadeltaOptimizer::DeserializeState(const std::string& str) {
  AdadeltaOptimizerState state;
  state.ParseFromString(str);
  auto lr_state = state.lr_state();
  this->lr_policy_->DeserializeState(lr_state.SerializeAsString());
  num_sample_passed_ = state.num_sample_passed();

  ProtoToTensor(state.parameter(), parameter_);
  ProtoToTensor(state.accum_gradient(), accum_gradient_);
  ProtoToTensor(state.accum_delta(), accum_delta_);
  ProtoToTensor(state.update_delta(), update_delta_);
}

} // namespace optimizer
} // namespace mypaddle
} // namespace bubblefs