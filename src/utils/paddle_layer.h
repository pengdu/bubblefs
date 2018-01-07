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

// Paddle/paddle/gserver/layers/Layer.h

#pragma once

#include <functional>
#include <memory>
#include "utils/paddle_function.h"
#include "utils/paddle_parameter.h"

/// Macro for registering a layer type.
/// Example: REGISTER_LAYER(crf_error, CRFDecodingErrorLayer);
#define REGISTER_LAYER(__type_name, __class_name) \
  static InitFunction __reg_type_##__type_name(   \
      []() { Layer::registrar_.registerClass<__class_name>(#__type_name); })

#define REGISTER_LAYER_CREATE_FUNC(__type_name, createFunction) \
  static InitFunction __reg_type_##__type_name(                 \
      []() { Layer::registrar_.registerClass(#__type_name, createFunction); })

namespace bubblefs {      
namespace mypaddle {

class Layer;
typedef std::shared_ptr<Layer> LayerPtr;
typedef std::map<std::string, LayerPtr> LayerMap;
class NeuralNetwork;

/// layer state, used for RNN and LSTM layers
struct LayerState {
  std::vector<MatrixPtr> value;
};
typedef std::shared_ptr<LayerState> LayerStatePtr;

/// Paddle device ID, MKLDNN is -2, CPU is -1
enum PADDLE_DEVICE_ID {
  MKLDNN_DEVICE = -2,
  CPU_DEVICE = -1,
};

/**
 * @brief Base class for layer.
 * Define necessary variables and functions for every layer.
 */
class Layer {
protected:
  /// Layer config
  LayerConfig config_;
  /// whether to use GPU
  bool useGpu_;
  /// Device Id. MKLDNN is -2, CPU is -1, and GPU is 0, 1, 2 ...
  int deviceId_;
  /// Input layers
  std::vector<LayerPtr> inputLayers_;
  /// Argument of input layers
  std::vector<std::string> inputArgument_;

  /// Parameter for each input layer.
  /// Parameters_[i] is nullptr if inputLayers_[i] does not need parameter.
  std::vector<ParameterPtr> parameters_;

  /// nullptr if bias is not needed.
  ParameterPtr biasParameter_;

  /// Output
  Argument output_;
  /// Several outputs stored on different devices, used in 'parallel_nn' case,
  /// and record them by deviceId_.
  /// Also used in 'use_mkldnn' case.
  std::vector<Argument> outputOtherDevice_;
  /// If there are several outputs, map them by each name.
  /// MKLDNNLayer use it only to merge output grad
  std::map<std::string, Argument*> outputMap_;
  /// Used to merge grad on different devices.
  MatrixPtr tmpGrad_;

  std::unique_ptr<ActivationFunction> activation_;

  /// Current passType, PASS_TRAIN or PASS_TEST
  PassType passType_;

  /// Random 0-1 matrix for dropOut
  MatrixPtr dropOutMask_;

  /// Whether the layer need to compute gradient
  bool needGradient_;
  /// Whether the layer need to compute re-sequence information
  bool needSequenceInfo_;

  /// Mark input grad in(true) or out(false) of backward function.
  std::vector<bool> markInBackward_;

  /// Layer forward function
  std::vector<std::shared_ptr<FunctionBase>> forward_;
  /// Layer backward function
  std::vector<std::shared_ptr<FunctionBase>> backward_;

public:
  /**
   * Wait until all input value ready.
   * Called before Layer::forward() function.
   */
  virtual void waitInputValue();

  /**
   * Copy layer's output_ to other device.
   * If output layer is in other device, called after Layer::forward() function.
   */
  virtual void copyOutputToOtherDevice();

  /**
   * Wait until all output grad ready and merge them to output_.grad.
   * Called before Layer::backward() function.
   */
  virtual void waitAndMergeOutputGrad();

  /**
   * Notify previous layer the output grad ready.
   * Called after Layer::backward() function.
   */
  virtual void markAllInputGrad();

protected:
  /**
   * Create layer function. Function is called in forward or backward.
   * \param function, Layer::forward_ or Layer::backward_
   * \param name, function name
   * \param config, initialization configuration for the function
   */
  void createFunction(std::vector<std::shared_ptr<FunctionBase>>& function,
                      const std::string& name,
                      const FuncConfig& config) {
    if (useGpu_) {
      function.emplace_back(
          FunctionBase::funcRegistrar_.createByType(name + "-GPU"));
    } else {
      function.emplace_back(
          FunctionBase::funcRegistrar_.createByType(name + "-CPU"));
    }
    auto& func = function.back();
    func->init(config);
  }

  /**
   * Notify specified layer the output grad ready.
   * Called in the backward function.
   * If do mark input grad in the backward function, you should to ensure
   * that all input grad will be marked in the backward function.
   */
  void markInputGrad(int inputIndex);

  /**
   * Get the argument of input layer.
   */
  const Argument& getInput(size_t inputIndex) const {
    return inputLayers_[inputIndex]->getOutput(deviceId_);
  }

  /**
   * Get the argument of input layer.
   */
  const Argument& getInput(const Layer& inputLayer) const {
    return inputLayer.getOutput(deviceId_);
  }

  /**
   * Get the argument of input layer with deviceId.
   */
  const Argument& getInput(size_t inputIndex, int deviceId) const {
    return inputLayers_[inputIndex]->getOutput(deviceId);
  }

  /**
   * Get the forward-input value.
   */
  const MatrixPtr& getInputValue(int inputIndex) {
    return inputLayers_[inputIndex]->getOutput(deviceId_).value;
  }

  /**
   * Get the forward-input value.
   */
  const MatrixPtr& getInputValue(const Layer& inputLayer) {
    return inputLayer.getOutput(deviceId_).value;
  }

  /**
   * Get the forward-input value with deviceId.
   */
  const MatrixPtr& getInputValue(int inputIndex, int deviceId) {
    return inputLayers_[inputIndex]->getOutput(deviceId).value;
  }

  /**
   * Get the forward-input grad.
   */
  const MatrixPtr& getInputGrad(int inputIndex) {
    return inputLayers_[inputIndex]->getOutput(deviceId_).grad;
  }

  /**
   * Get the forward-input grad.
   */
  const MatrixPtr& getInputGrad(const Layer& inputLayer) {
    return inputLayer.getOutput(deviceId_).grad;
  }

  /**
   * Get the forward-input grad.
   */
  const MatrixPtr& getInputGrad(int inputIndex, int deviceId) {
    return inputLayers_[inputIndex]->getOutput(deviceId).grad;
  }

  /**
   * Get the forward-input label.
   */
  const IVectorPtr& getInputLabel(const Layer& inputLayer) {
    return inputLayer.getOutput(deviceId_).ids;
  }

  /**
   * Change the size of output (value, grad).
   * Reset to value zero if isValueClean = true,
   * Reset to grad zero if isGradClean = true.
   */
  void resetSpecifyOutput(Argument& output,
                          size_t height,
                          size_t width,
                          bool isValueClean,
                          bool isGradClean);

  /**
   * Add output argument to other devices.
   */
  void addOutputArgument(int deviceId);

public:
  explicit Layer(const LayerConfig& config, bool useGpu = FLAGS_use_gpu);
  virtual ~Layer() {}

  /// Register a Layer
  static ClassRegistrar<Layer, LayerConfig> registrar_;

  /**
   * Get the flag whether layer need to compute gradient.
   */
  bool needGradient() const { return needGradient_; }

  /**
   * Set the flag whether layer need to compute gradient.
   */
  void setNeedGradient(bool need) { needGradient_ = need; }

  /**
   * Set the flag whether layer need to re-compute sequence information,
   * which includes sequenceStartPositions or subSequenceStartPositions.
   */
  void setNeedSequenceInfo(bool need) { needSequenceInfo_ = need; }

  /**
   * Get layer's name.
   */
  const std::string& getName() const { return config_.name(); }

  /**
   * Get layer's type.
   */
  const std::string& getType() const { return config_.type(); }

  /**
   * Get layer's size.
   */
  size_t getSize() const { return config_.size(); }

  /**
   * Get layer's deviceId.
   */
  int getDeviceId() const { return deviceId_; }

  /**
   * Add the inputLayer.
   */
  void addPrev(LayerPtr l) { inputLayers_.push_back(l); }

  /**
   * Get the size of inputLayer[i].
   */
  const LayerPtr& getPrev(size_t i) { return inputLayers_[i]; }

  /**
   * Get the forward-output value.
   */
  const MatrixPtr& getOutputValue() { return output_.value; }

  /**
   * Get the forward-output label.
   */
  const IVectorPtr& getOutputLabel() { return output_.ids; }

  /**
   * Get the backward-Loss value.
   */
  const MatrixPtr& getOutputGrad() { return output_.grad; }
  /**
   * If layer has multi-output, set output into outputMap_.
   */
  void setOutput(const std::string& name, Argument* output) {
    outputMap_[name] = output;
  }

  /**
   * Get the output map size, if layer has multi-output.
   */
  size_t getOutputMapSize() { return outputMap_.size(); }

  /**
   * Get the output based on layer's name.
   */
  Argument& getOutput(const std::string& str = "") {
    if (str == "") {
      return output_;
    } else {
      auto output = outputMap_.find(str);
      if (output != outputMap_.end()) {
        return *output->second;
      } else {
        LOG(FATAL) << "No specific output " << str;
        return *((Argument*)nullptr);
      }
    }
  }

  /**
   * Get the output based on deviceId.
   */
  const Argument& getOutput(int deviceId) const {
    if (deviceId == getDeviceId()) {
      return output_;
    } else {
      for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
        if (outputOtherDevice_[i].deviceId == deviceId) {
          return outputOtherDevice_[i];
        }
      }

      LOG(FATAL) << "No specific device output ";
      return *((Argument*)nullptr);
    }
  }

  /**
   * Get layer's parameters.
   */
  const std::vector<ParameterPtr>& getParameters() { return parameters_; }

  /**
   * Get layer's bias-parameters.
   */
  const ParameterPtr& getBiasParameter() { return biasParameter_; }

  /**
   * Create pointer of layer.
   */
  static LayerPtr create(const LayerConfig& config);

  /**
   * Resize the output matrix size.
   */
  void resizeOutput(size_t height, size_t width);

  /**
   * Resize the output matrix size,
   * and reset value to zero.
   */
  void reserveOutput(size_t height, size_t width);

  /**
   * Resize the output matrix size,
   * and reset value and grad to zero.
   */
  void resetOutput(size_t height, size_t width);

  /**
   * Clear the gradient of output.
   */
  void zeroGrad();

  /**
   * Intialization.
   * For example, adding input layers from layerMap and parameterMap.
   */
  virtual bool init(const LayerMap& layerMap, const ParameterMap& parameterMap);

  /**
   * Intialization for sub network if there has sub network.
   * @param rootNetwork root network
   * @param config model config
   * @param parameterTypes parameter's type
   * @param useGpu whether to use gpu or not
   */
  virtual void initSubNetwork(NeuralNetwork* rootNetwork,
                              const ModelConfig& config,
                              const std::vector<ParameterType>& parameterTypes,
                              bool useGpu) {}

  /**
   * @brief Access SubNetwork Object.
   *        If subnetwork exists, then invoke callback with subnetwrk.
   * @param callback if sub-network is exist, the callback is invoked.
   */
  virtual void accessSubNetwork(
      const std::function<void(NeuralNetwork&)>& callback) {}

  /**
   * If use sparse row matrix as parameter,
   * prefetch feature ids in input label.
   */
  virtual void prefetch() {}

  /**
   * Forward propagation.
   * All inherited implementation should call Layer::foward() function.
   */
  virtual void forward(PassType passType) {
    passType_ = passType;
    if (!inputLayers_.empty() && needSequenceInfo_) {
      const Argument& input = getInput(0);
      output_.sequenceStartPositions = input.sequenceStartPositions;
      output_.subSequenceStartPositions = input.subSequenceStartPositions;
      output_.cpuSequenceDims = input.cpuSequenceDims;
    }
  }

  /**
   * Reset the internal state variables.
   * Allocate them if they have not been allocated.
   * This function need to called before Layer::forward() for generating
   * sequence.
   *
   * This is used for sequence generation. When generating sequence, the
   * calculation at current timestamp depends on the state from previous
   * timestamp. The model needs to keep the information about the previous
   * timestamp in the state variables. Layers such as RecurrentLayer,
   * LstmLayer and ContextLayer have state variables.
   */
  virtual void resetState() {}

  /**
   * Set layer state.
   */
  virtual void setState(LayerStatePtr state) {}

  /**
   * Get layer state.
   * @return A copy of internal state.
   */
  virtual LayerStatePtr getState() { return nullptr; }

  /**
   * Show output state.
   */
  void showOutputStats();

  /**
   * Backward propagation.
   * Should only be called after Layer::forward() function.
   */
  virtual void backward(const UpdateCallback& callback = nullptr) = 0;

  /**
   * One pass is finished.
   */
  virtual void onPassEnd() {}

protected:
  /**
   * Forward of activation function.
   */
  void forwardActivation();
  /**
   * Backward of activation function.
   */
  void backwardActivation();
  /**
   * Forward of dropOut.
   */
  void forwardDropOut();
  /**
   * Initilize the needGradient_ flag.
   */
  void initNeedFlags();
};

Layer::Layer(const LayerConfig& config, bool useGpu)
    : config_(config),
      useGpu_(useGpu),
      deviceId_(CPU_DEVICE),
      needSequenceInfo_(true) {}

bool Layer::init(const LayerMap& layerMap, const ParameterMap& parameterMap) {
  if (useGpu_ && FLAGS_parallel_nn) {
    /* gpu environment is specified by device property */
    deviceId_ = config_.device();
    if (deviceId_ < 0) {
      useGpu_ = false;
    }
  }

  output_.deviceId = deviceId_;

  for (auto& inputConfig : config_.inputs()) {
    std::string inputName = inputConfig.input_layer_name();
    LayerPtr inputLayer;
    CHECK(mapGet(inputName, layerMap, &inputLayer))
        << "Cannot find input layer " << inputName << " for layer "
        << getName();
    this->addPrev(inputLayer);

    inputLayer->addOutputArgument(deviceId_);

    if (inputConfig.has_input_parameter_name()) {
      ParameterPtr parameter;
      CHECK(
          mapGet(inputConfig.input_parameter_name(), parameterMap, &parameter))
          << "Cannot find input parameter "
          << inputConfig.input_parameter_name() << " for layer " << getName();
      parameter->incShared();
      CHECK_EQ(parameter->getDeviceId(), getDeviceId());
      parameters_.push_back(parameter);
    } else {
      parameters_.push_back(nullptr);
    }

    if (inputConfig.has_input_layer_argument()) {
      inputArgument_.push_back(inputConfig.input_layer_argument());
    } else {
      inputArgument_.push_back("");
    }
  }

  if (config_.has_bias_parameter_name()) {
    CHECK(mapGet(config_.bias_parameter_name(), parameterMap, &biasParameter_))
        << "Cannot find bias parameter " << config_.bias_parameter_name()
        << " for layer " << getName();
    biasParameter_->incShared();
    CHECK_EQ(biasParameter_->getDeviceId(), getDeviceId());
  }

  /* specify the activation function according to the configuration */
  std::string action_type = config_.active_type();
  activation_.reset(ActivationFunction::create(action_type));
  CHECK(activation_);

  initNeedFlags();
  markInBackward_.assign(inputLayers_.size(), false);

  return true;
}

ClassRegistrar<Layer, LayerConfig> Layer::registrar_;

LayerPtr Layer::create(const LayerConfig& config) {
  std::string type = config.type();

#ifndef PADDLE_MOBILE_INFERENCE
  // NOTE: As following types have illegal character '-',
  // they can not use REGISTER_LAYER to registrar.
  // Besides, to fit with old training models,
  // they can not use '_' instead.
  if (type == "multi-class-cross-entropy")
    return LayerPtr(new MultiClassCrossEntropy(config));
  else if (type == "rank-cost")
    return LayerPtr(new RankingCost(config));
  else if (type == "auc-validation")
    return LayerPtr(new AucValidation(config));
  else if (type == "pnpair-validation")
    return LayerPtr(new PnpairValidation(config));
#endif

  return LayerPtr(registrar_.createByType(config.type(), config));
}

void Layer::resetSpecifyOutput(Argument& output,
                               size_t height,
                               size_t width,
                               bool isValueClean,
                               bool isGradClean) {
  SetDevice device(output.deviceId);

  Matrix::resizeOrCreate(
      output.value, height, width, /* trans */ false, useGpu(output.deviceId));
  if (isValueClean) {
    output.value->zeroMem();
  }

  if (passType_ != PASS_TEST && needGradient()) {
    Matrix::resizeOrCreate(
        output.grad, height, width, /* trans */ false, useGpu(output.deviceId));
    if (isGradClean) {
      output.grad->zeroMem();
    }
  }
}

void Layer::resizeOutput(size_t height, size_t width) {
  resetSpecifyOutput(output_, height, width, false, false);

  for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
    resetSpecifyOutput(outputOtherDevice_[i], height, width, false, false);
  }
}

void Layer::reserveOutput(size_t height, size_t width) {
  resetSpecifyOutput(output_, height, width, false, true);

  for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
    resetSpecifyOutput(outputOtherDevice_[i], height, width, false, true);
  }
}

void Layer::resetOutput(size_t height, size_t width) {
  resetSpecifyOutput(output_, height, width, true, true);

  for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
    resetSpecifyOutput(outputOtherDevice_[i], height, width, true, true);
  }
}

void Layer::addOutputArgument(int deviceId) {
  if (deviceId == deviceId_) {
    output_.countIncrement();
    return;
  } else {
    for (size_t i = 0; i < outputOtherDevice_.size(); i++) {
      if (outputOtherDevice_[i].deviceId == deviceId) {
        outputOtherDevice_[i].countIncrement();
        return;
      }
    }
  }

  Argument argu;
  argu.deviceId = deviceId;
  outputOtherDevice_.push_back(argu);
  outputOtherDevice_.back().countIncrement();
}

void Layer::copyOutputToOtherDevice() {
  for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
    SetDevice device(outputOtherDevice_[i].deviceId);
    // If outputOtherDevice_[i].value is a CpuMatrix,
    // the copyFrom is a synchronous interface.
    // If outputOtherDevice_[i].value is a GpuMatrix, since subsequent
    // calculations are all on HPPL_STREAM_DEFAULT,
    // copyFrom can be an asynchronous interface.
    outputOtherDevice_[i].value->copyFrom(*getOutputValue(),
                                          HPPL_STREAM_DEFAULT);
    outputOtherDevice_[i].sequenceStartPositions =
        output_.sequenceStartPositions;
    outputOtherDevice_[i].subSequenceStartPositions =
        output_.subSequenceStartPositions;
    outputOtherDevice_[i].cpuSequenceDims = output_.cpuSequenceDims;

    outputOtherDevice_[i].notifyValueReady();
  }
}

void Layer::waitInputValue() {
  for (size_t i = 0; i != inputLayers_.size(); i++) {
    if (inputLayers_[i]->getDeviceId() != deviceId_) {
      getInput(i).waitValueReady();
    }
  }
}

void Layer::waitAndMergeOutputGrad() {
  if (!output_.grad || !outputOtherDevice_.size()) {
    return;
  }

  for (size_t i = 0; i != outputOtherDevice_.size(); i++) {
    outputOtherDevice_[i].waitGradReady();
  }

  /* merge output grad */
  size_t i = 0;
  if (!output_.getAllCount()) {
    output_.grad->copyFrom(*outputOtherDevice_[0].grad, HPPL_STREAM_1);
    hl_stream_synchronize(HPPL_STREAM_1);

    i++;
    if (outputOtherDevice_.size() == 1) return;
  }

  Matrix::resizeOrCreate(tmpGrad_,
                         output_.grad->getHeight(),
                         output_.grad->getWidth(),
                         /* trans */ false,
                         useGpu(output_.deviceId));

  for (; i != outputOtherDevice_.size(); i++) {
    tmpGrad_->copyFrom(*outputOtherDevice_[i].grad, HPPL_STREAM_1);
    hl_stream_synchronize(HPPL_STREAM_1);
    output_.grad->add(*tmpGrad_);
  }
}

void Layer::markAllInputGrad() {
  for (size_t i = 0; i != inputLayers_.size(); ++i) {
    if (!markInBackward_[i]) {
      inputLayers_[i]->getOutput(deviceId_).notifyGradReady();
    }
    markInBackward_[i] = false;
  }
}

void Layer::markInputGrad(int inputIndex) {
  inputLayers_[inputIndex]->getOutput(deviceId_).notifyGradReady();
  markInBackward_[inputIndex] = true;
}

void Layer::zeroGrad() {
  CHECK(output_.grad.get() != NULL);
  output_.grad->zeroMem();
}

void Layer::initNeedFlags() {
  auto initFlag = [this](
      bool& flag, bool (Layer::*flagQueryFunc)() const, ParameterType type) {
    flag = false;
    if (biasParameter_ && biasParameter_->hasType(type)) {
      flag = true;
    }
    if (!flag) {
      for (auto& para : parameters_) {
        if (para && para->hasType(type)) {
          flag = true;
          break;
        }
      }
    }
    if (!flag) {
      for (auto& layer : inputLayers_) {
        if ((layer.get()->*flagQueryFunc)()) {
          flag = true;
        }
      }
    }
  };
  initFlag(needGradient_, &Layer::needGradient, PARAMETER_GRADIENT);
}

void Layer::showOutputStats() {
  MatrixPtr out = getOutputValue();
  if (!out) return;
  if (!out->getElementCnt()) {
    LOG(INFO) << "The number of output of " << config_.name()
              << " is 0, skip to show the statistics";
    return;
  }
  MatrixPtr outSquare;
  if (dynamic_cast<GpuSparseMatrix*>(out.get())) {
    GpuSparseMatrix* tmp = dynamic_cast<GpuSparseMatrix*>(out.get());
    outSquare = std::make_shared<CpuSparseMatrix>(tmp->getHeight(),
                                                  tmp->getWidth(),
                                                  tmp->getElementCnt(),
                                                  tmp->getValueType(),
                                                  tmp->getFormat());
  } else {
    outSquare = out->clone();
  }
  outSquare->copyFrom(*out, HPPL_STREAM_DEFAULT);
  hl_stream_synchronize(HPPL_STREAM_DEFAULT);

  real mean = outSquare->getSum() / out->getElementCnt();
  real min;
  real max;
  if (dynamic_cast<CpuSparseMatrix*>(outSquare.get())) {
    auto tmpMat = dynamic_cast<CpuSparseMatrix*>(outSquare.get());
    min = tmpMat->getMin();
    max = tmpMat->getMax();
    tmpMat->square2();
    LOG(INFO) << "show statistics of [none zero values] in sparse matrix";
  } else {
    min = outSquare->getMin();
    max = outSquare->getMax();
    outSquare->square2();
  }
  real std = (outSquare->getSum() / outSquare->getElementCnt()) - mean * mean;
  std = std > 0 ? std : 0;
  LOG(INFO) << "The output state of " << config_.name() << ": mean=" << mean
            << ", "
            << "std=" << std << ", "
            << "min=" << min << ", "
            << "max=" << max;
}

void Layer::forwardActivation() {
  /* activation */
  auto status = activation_->forward(output_);
  status.check();

  /* dropout */
  if (config_.drop_rate() > 0) {
    forwardDropOut();
    CHECK_NE(activation_->getName(), "softmax")
        << "Softmax activation cannot be used with Dropout";
  }

  if (FLAGS_show_layer_stat) {
    showOutputStats();
  }
}

void Layer::backwardActivation() {
  /* Do error clipping */
  if (config_.error_clipping_threshold() > 0.0f) {
    if (FLAGS_log_error_clipping) {
      VectorPtr outGradVec = Vector::create(
          output_.grad->getData(), output_.grad->getElementCnt(), useGpu_);
      real maxAbsGrad = outGradVec->getAbsMax();
      if (maxAbsGrad > config_.error_clipping_threshold()) {
        real avgAbsGrad = outGradVec->getAbsSum() / outGradVec->getSize();
        LOG(INFO) << " layer=" << config_.name() << " need clipping,"
                  << " max error=" << maxAbsGrad << " avg error=" << avgAbsGrad;
      }
    }
    output_.grad->clip(-config_.error_clipping_threshold(),
                       config_.error_clipping_threshold());
  }

  /* Do dropout for delta*/
  if (config_.drop_rate() > 0 && passType_ != PASS_TEST) {
    MatrixPtr oGrad = getOutputGrad();
    oGrad->dotMul(*oGrad, *dropOutMask_);
  }

  auto status = activation_->backward(output_);
  status.check();
}

void Layer::forwardDropOut() {
  auto& outV = getOutputValue();

  if (passType_ == PASS_TRAIN) {
    // new dropOutMask_ if dropOutMask_ is null ptr
    Matrix::resizeOrCreate(dropOutMask_,
                           outV->getHeight(),
                           outV->getWidth(),
                           false,
                           useGpu(deviceId_));
    dropOutMask_->randomizeUniform();  // generate a uniform random matrix
    dropOutMask_->biggerThanScalar(config_.drop_rate());  // random mask
    outV->dotMul(*outV, *dropOutMask_);                   // dropout
  } else if (passType_ == PASS_GC) {
    // only initialize once
    if (!dropOutMask_) {
      dropOutMask_ = Matrix::create(
          outV->getHeight(), outV->getWidth(), false, useGpu(deviceId_));
      // We use cpu matrix to generate mask so that the mask
      // will be same for both gpu version and cpu version.
      // This will help unittest to make sure they have same result.
      MatrixPtr tmpMask = Matrix::create(outV->getHeight(), outV->getWidth());
      tmpMask->randomizeUniform();  // generate a uniform random matrix
      tmpMask->biggerThanScalar(config_.drop_rate());  // random mask
      dropOutMask_->copyFrom(*tmpMask);
    }
    outV->dotMul(*outV, *dropOutMask_);
  } else {  // passType == PASS_TEST
    outV->mulScalar(1.0 - config_.drop_rate());
  }
}

}  // namespace mypaddle
}  // namespace bubblefs