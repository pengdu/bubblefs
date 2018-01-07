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

// Paddle/paddle/pserver/ParameterServerController.h
// Paddle/paddle/pserver/ParameterServerController.cpp

#pragma once

#include "utils/paddle_pserver_parameter_server.h"
#include "utils/paddle_string_util.h"

namespace bubblefs {
namespace mypaddle {

/**
 * @brief ParameterServerController is used for create, init and manage multi
 * parameter server instances. The num of the instances is decided by port
 * num(the ports number for parameter send) and network devices configured
 * by gflags or proto.
 */
class ParameterServerController final {
public:
  DISABLE_COPY(ParameterServerController);

  /**
   * @brief Ctor, Create a ParameterServerController from ParameterServerConfig.
   */
  explicit ParameterServerController(const ParameterServerConfig& config);

  /**
   * @brief Dtor.
   */
  ~ParameterServerController();

  /**
   * @brief create ParameterServerController from gflags, this is used for
   * compatibility with the old usage of configuration by gflags.
   */
  static ParameterServerController* createFromGflags();

  /**
   * @brief create ParameterServerController with ParameterServerConfig, remove
   * gflags from ParameterServer. Init all ParameterServer2 instances according
   * to
   * the config.
   */
  static ParameterServerController* create(const ParameterServerConfig& config);

  /**
   * @brief start all ParameterServer2 instances in this
   * ParameterServerController.
   */
  void start();

  /**
   * @brief join and wait for all ParameterServer2 instances thread in this
   * ParameterServerController.
   */
  void wait();

private:
  std::vector<std::unique_ptr<ParameterServer2>> parameterServers_;
};

ParameterServerController::ParameterServerController(
    const ParameterServerConfig& config) {
  // round robin to load balance RDMA server ENGINE
  std::vector<std::string> devices;
  int rdmaCpu = 0;
  int onlineCpus = rdma::numCpus();
  int numPorts = config.ports_num() + config.ports_num_for_sparse();

  if (config.nics().empty()) {
    parameterServers_.resize(numPorts);
    for (int i = 0; i < numPorts; ++i) {
      if (config.rdma_tcp() == "rdma") {
        parameterServers_[i].reset(
            new ParameterServer2(std::string(), config.port() + i, rdmaCpu++));
        rdmaCpu = rdmaCpu % onlineCpus;
      } else {
        parameterServers_[i].reset(
            new ParameterServer2(std::string(), config.port() + i));
      }
      CHECK(parameterServers_[i]->init()) << "Fail to initialize parameter "
                                             "server on port "
                                          << config.port() + i;
    }
  } else {
    str::split(config.nics(), ',', &devices);
    parameterServers_.resize(devices.size() * numPorts);
    for (int i = 0; i < numPorts; ++i) {
      for (size_t j = 0; j < devices.size(); ++j) {
        if (config.rdma_tcp() == "rdma") {
          parameterServers_[i * devices.size() + j].reset(new ParameterServer2(
              getIpAddr(devices[j]), config.port() + i, rdmaCpu++));
          rdmaCpu = rdmaCpu % onlineCpus;
        } else {
          parameterServers_[i * devices.size() + j].reset(
              new ParameterServer2(getIpAddr(devices[j]), config.port() + i));
        }
        CHECK(parameterServers_[i * devices.size() + j]->init())
            << "Fail to initialize parameter server with device " << devices[j]
            << config.port() + i;
      }
    }
  }
}

ParameterServerController::~ParameterServerController() { this->wait(); }

ParameterServerController* ParameterServerController::createFromGflags() {
  ParameterServerConfig config;

  config.set_nics(FLAGS_nics);
  config.set_rdma_tcp(FLAGS_rdma_tcp);
  config.set_port(FLAGS_port);
  config.set_ports_num(FLAGS_ports_num);
  config.set_ports_num_for_sparse(FLAGS_ports_num_for_sparse);

  return create(config);
}

ParameterServerController* ParameterServerController::create(
    const ParameterServerConfig& config) {
  return new ParameterServerController(config);
}

void ParameterServerController::start() {
  LOG(INFO) << "number of parameterServer instances: "
            << parameterServers_.size();
  int i = 0;
  for (const auto& parameterServer : parameterServers_) {
    LOG(INFO) << "Starting parameterServer[" << i << "]";
    parameterServer->start();
    i++;
  }
}

void ParameterServerController::wait() {
  int i = 0;
  for (const auto& parameterServer : parameterServers_) {
    LOG(INFO) << "Waiting parameterServer[" << i << "]";
    parameterServer->join();
    i++;
  }
}

}  // namespace mypaddle
}  // namespace bubblefs