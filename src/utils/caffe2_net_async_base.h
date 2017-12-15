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

// caffe2/caffe2/core/net_async_base.h

#ifndef BUBBLEFS_UTILS_CAFFE2_NET_ASYNC_POLLING_H_
#define BUBBLEFS_UTILS_CAFFE2_NET_ASYNC_POLLING_H_

#include "platform/base_error.h"
#include "platform/types.h"
#include "platform/caffe2_timer.h"
#include "utils/caffe2_blob.h"
#include "utils/caffe2_net_dag_utils.h"
#include "utils/caffe2_observer.h"
#include "utils/caffe2_tensor.h"
#include "utils/caffe2_thread_pool.h"
#include "utils/caffe2_workspace.h"
#include "utils/caffe2_proto_caffe2.h"
#include "utils/caffe2_simple_queue.h"

namespace bubblefs {
namespace mycaffe2 {

class AsyncNetBase : public NetBase {
 public:
  AsyncNetBase(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  ~AsyncNetBase() override;

  bool SupportsAsync() override {
    return true;
  }

  std::vector<OperatorBase*> GetOperators() const override {
    return operators_;
  }

 protected:
  bool AsyncNetBase::canSchedule(
    int task_id,
    const std::vector<EventStatus>* status) {
    auto first_child_op_id = chains_[task_id].front();
    for (auto parent_id : parents(task_id)) {
      auto last_parent_op_id = chains_[parent_id].back();
      EventStatus parent_status;
      if (status) {
        parent_status = status->at(parent_id);
      } else {
        parent_status = operators_[last_parent_op_id]->event().Query();
      }
      bool can_schedule = Event::CanSchedule(
          operators_[last_parent_op_id]->event().GetType(),
          parent_status,
          operators_[first_child_op_id]->event().GetType(),
          operators_[first_child_op_id]->SupportsAsyncScheduling());
      if (!can_schedule) {
        return false;
      }
    }

    return true;
  }

  int tasksNum() const {
    return chains_.size();
  }
  
  Event& event(int task_id) const {
    auto& task = chains_[task_id];
    auto& last_task_op = operators_[task.back()];
    return last_task_op->event();
  }
  
  EventStatus query(int task_id) const {
    return event(task_id).Query();
  }
  
  const std::vector<int>& children(int task_id) const {
    const auto& task_node = chain_nodes_[task_id];
    return task_node.children_;
  }
  
  const std::vector<int>& parents(int task_id) const {
    const auto& task_node = chain_nodes_[task_id];
    return task_node.parents_;
  }

  void asyncWait(
    int task_id,
    int stream_id,
    const std::vector<int>& wait_task_ids) const {
    auto first_op_id = chains_[task_id].front();
    auto& first_op = operators_[first_op_id];
    std::vector<const Event*> events;
    events.reserve(wait_task_ids.size());
    for (auto wait_task_id : wait_task_ids) {
      events.push_back(&event(wait_task_id));
    }
    first_op->WaitEvents(events, stream_id);
  }
  
  bool run(int task_id, int stream_id) {
    bool failed = false;
    std::string err_msg;
    for (auto& op_id : chains_[task_id]) {
      auto& op = operators_[op_id];
      try {
        if (!op->RunAsync(stream_id)) {
          failed = true;
          err_msg = "Failed to execute task: op " +
              (op->has_debug_def() ? op->type() : " unknown");
          break;
        }
      } catch (const std::exception& e) {
        failed = true;
        err_msg = e.what();
        break;
      } catch (...) {
        failed = true;
        err_msg = "Failed to execute task: unknown error";
        break;
      }
    }

    if (!failed) {
      operators_[chains_[task_id].back()]->event().Finish();
    }

    return !failed;
  }
  
  /*
     const auto& device_option = event(task_id).GetDeviceOption();
     int gpu_id = device_option.cuda_gpu_id();
     stream_id = stream_counters_[gpu_id]++;
  */
  int stream(int task_id);
  
  std::shared_ptr<TaskThreadPool> pool(const DeviceOption& device_option);

  void finishTasks(const std::unordered_set<int>& task_ids) {
    for (const auto& task_id : task_ids) {
      event(task_id).Finish();
    }
  }
  
  void finalizeEvents() {
    for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
      auto status = query(task_id);
      if (status == EventStatus::EVENT_SCHEDULED) {
        event(task_id).Finish();
      } else if (status == EventStatus::EVENT_INITIALIZED) {
        event(task_id).SetFinished();
      }
    }
  }

  bool isStreamFree(int task_id, int stream_id) const {
    auto& task = chains_[task_id];
    auto& last_task_op = operators_[task.back()];
    return last_task_op->IsStreamFree(stream_id);
  }

  // Operator/task graph
  /*
    events_.reserve(chains_.size());
    for (const auto& chain : chains_) {
      const auto& op = operators_[chain.back()];
      events_.push_back(&op->event());
    }
  */
  std::vector<OperatorBase*> operators_;
  std::vector<dag_utils::OperatorNode> operator_nodes_;
  std::vector<std::vector<int>> chains_;
  std::vector<dag_utils::OpGraphNode> chain_nodes_; // chains' parents/children

  // Pools and streams
  std::mutex pools_mutex_;
  /*
    gpu_pool_ = ThreadPoolRegistry()->Create(
          DeviceTypeName(gpu_option.device_type()), gpu_option);
    std::unique_lock<std::mutex> pools_lock(pools_mutex_);
    pool = gpu_pools_[gpu_id];
    pool = ThreadPoolRegistry()->Create(
           DeviceTypeName(device_option.device_type()), device_option);
    gpu_pools_[gpu_id] = pool; 
  */
  std::vector<std::shared_ptr<TaskThreadPool>> gpu_pools_;
  std::shared_ptr<TaskThreadPool> cpu_pool_;
  std::shared_ptr<TaskThreadPool> gpu_pool_;
  static thread_local std::vector<int> stream_counters_;

  DISALLOW_COPY_AND_ASSIGN(AsyncNetBase);
};

std::shared_ptr<TaskThreadPool> GetAsyncNetCPUThreadPool();

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_NET_ASYNC_POLLING_H_