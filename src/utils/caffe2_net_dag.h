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

// caffe2/caffe2/core/net_dag.h

#ifndef BUBBLEFS_UTILS_CAFFE2_NET_DAG_H_
#define BUBBLEFS_UTILS_CAFFE2_NET_DAG_H_

#include <atomic>
#include <climits>
#include <cstddef>
#include <thread> // NOLINT
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "platform/base_error.h"
#include "platform/types.h"
#include "platform/caffe2_timer.h"
#include "utils/caffe2_blob.h"
#include "utils/caffe2_net_dag_utils.h"
#include "utils/caffe2_observer.h"
#include "utils/caffe2_operator_schema.h"
#include "utils/caffe2_tensor.h"
#include "utils/caffe2_workspace.h"
#include "utils/caffe2_proto_caffe2.h"
#include "utils/caffe2_simple_queue.h"

namespace bubblefs {
namespace mycaffe2 {

class DAGNetBase : public NetBase {
 public:
  DAGNetBase(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  ~DAGNetBase() override;

  // WorkerFunction() is a function wrapper to allow us to run worker threads.
  // It checks out one ready-to-run operator from the job queue, runs it,
  // notifies all its children, and for any children that is ready, enqueues
  // it to the job queue.
  void WorkerFunction();
  std::vector<float> TEST_Benchmark(
      const int warmup_runs,
      const int main_runs,
      const bool run_individual) override;

  const dag_utils::ExecutionChains& TEST_execution_chains() const {
    return execution_chains_;
  }

  std::vector<OperatorBase*> GetOperators() const override {
    return operators_;
  }

 protected:
  bool DoRunAsync() override;

  virtual bool RunAt(int chain_id, const std::vector<int>& chain) = 0;

  std::vector<dag_utils::OperatorNode> operator_nodes_;
  std::vector<OperatorBase*> operators_;
  dag_utils::ExecutionChains execution_chains_;
  std::vector<int> initial_frontier_;
  std::unique_ptr<SimpleQueue<int>> job_queue_;
  std::vector<std::thread> workers_;
  int num_workers_;
  int remaining_ops_;

  bool success_;
  int iter_;
  std::mutex remaining_ops_mutex_;
  std::condition_variable cv_;
  std::mutex run_in_progress_;

  struct DAGNetStats {
    CAFFE_STAT_CTOR(DAGNetStats);
    CAFFE_AVG_EXPORTED_STAT(task_pool_wait_time_us);
    CAFFE_AVG_EXPORTED_STAT(task_time_to_scheduled_us);
    CAFFE_AVG_EXPORTED_STAT(task_time_to_succeeded_ms);
    CAFFE_AVG_EXPORTED_STAT(task_wait_time_us);
  };
  mutable std::vector<DAGNetStats> stats_;
  std::unordered_map<int, std::unique_ptr<Timer>> task_timers_;

  DISALLOW_COPY_AND_ASSIGN(DAGNetBase);
};

class DAGNet : public DAGNetBase {
 public:
  using DAGNetBase::DAGNetBase;

 protected:
  bool RunAt(int chain_id, const std::vector<int>& chain) override;
  bool SupportsAsync() override {
    return false;
  }
};

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_NET_DAG_H_