
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

// caffe2/caffe2/core/net_async_polling.h
// caffe2/caffe2/core/net_async_polling.cc

#ifndef BUBBLEFS_UTILS_CAFFE2_NET_ASYNC_POLLING_H_
#define BUBBLEFS_UTILS_CAFFE2_NET_ASYNC_POLLING_H_

#include "utils/caffe2_net_async_base.h"
#include "platform/caffe2_timer.h"
#include "utils/caffe2_operator.h"
#include "utils/caffe2_stats.h"

namespace bubblefs {
namespace mycaffe2 {

extern bool FLAGS_caffe2_dag_net_collect_stats;  
  
class AsyncPollingNet : public AsyncNetBase {
 public:
  AsyncPollingNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  ~AsyncPollingNet() override;

 protected:
  bool DoRunAsync() override;

  bool pollAndSchedule();
  void schedule(int task_id);

  // Synchronization
  std::mutex running_mutex_;
  std::condition_variable running_cv_;
  std::atomic<bool> running_;

  // Stats
  struct AsyncPollingNetStats {
    CAFFE_STAT_CTOR(AsyncPollingNetStats);
   CAFFE_AVG_EXPORTED_STAT(poll_time_ms);
    CAFFE_AVG_EXPORTED_STAT(task_pool_wait_time_us);
    CAFFE_AVG_EXPORTED_STAT(task_run_time_us);
    CAFFE_AVG_EXPORTED_STAT(poll_status_update_time_us);
    CAFFE_AVG_EXPORTED_STAT(task_time_to_scheduled_us);
    CAFFE_AVG_EXPORTED_STAT(task_time_to_succeeded_ms);
  };
  mutable std::vector<AsyncPollingNetStats> stats_;
  std::vector<std::unique_ptr<Timer>> task_timers_;
  void updateTaskStats(int task_id);

  // Polling
  std::vector<EventStatus> status_;
  void reset();
  std::atomic<bool> has_chain_failed_;

  DISALLOW_COPY_AND_ASSIGN(AsyncPollingNet);
};

AsyncPollingNet::AsyncPollingNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : AsyncNetBase(net_def, ws), running_(false) {
  task_timers_.resize(tasksNum());
  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    task_timers_[task_id] = mycaffe2::make_unique<Timer>();
  }

  stats_.reserve(DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);
  for (auto device_idx = 0;
       device_idx < DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES;
       ++device_idx) {
    stats_.emplace_back(
        "async_net/stats/" + net_def->name() + "/" +
        DeviceTypeName(device_idx));
  }

  reset();
}

bool AsyncPollingNet::DoRunAsync() {
  PANIC_ENFORCE(!running_, "Concurrent RunAsync calls");
  running_ = true;
  reset();

  StartAllObservers();

  Timer timer;
  bool success = pollAndSchedule();
  if (FLAGS_caffe2_dag_net_collect_stats) {
    CAFFE_EVENT(stats_[CPU], poll_time_ms, timer.MilliSeconds());
  }
  if (!success) {
    finalizeEvents();
  }

  StopAllObservers();
  running_ = false;
  return success;
}

void AsyncPollingNet::schedule(int task_id) {
  if (FLAGS_caffe2_dag_net_collect_stats) {
    task_timers_[task_id]->Start();
  }
  const auto& device_option = event(task_id).GetDeviceOption();
  pool(device_option)->run([this, task_id, device_option]() {
    int stream_id = stream(task_id);

    if (FLAGS_caffe2_dag_net_collect_stats) {
      CAFFE_EVENT(
          stats_[device_option.device_type()],
          task_pool_wait_time_us,
          task_timers_[task_id]->MicroSeconds());
    }

    // Non-blocking wait, setups scheduling of dependent async computations;
    // canSchedule ensures that there's no busy wait,
    // for CUDA events we need to insert CUDA event synchronization to ensure
    // that async CUDA computations are executed in correct order
    asyncWait(task_id, stream_id, parents(task_id));
    bool result;
    if (FLAGS_caffe2_dag_net_collect_stats) {
      Timer run_time;
      result = run(task_id, stream_id);
      CAFFE_EVENT(
          stats_[device_option.device_type()],
          task_run_time_us,
          run_time.MicroSeconds());
    } else {
      result = run(task_id, stream_id);
    }
    if (!result) {
      has_chain_failed_ = true;
    }
  });
}

void AsyncPollingNet::reset() {
  status_.clear();
  status_.resize(tasksNum(), EventStatus::EVENT_INITIALIZED);
  has_chain_failed_ = false;
}

bool AsyncPollingNet::pollAndSchedule() {
  std::unordered_set<int> scheduled_tasks;
  std::unordered_set<int> current_tasks;

  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    if (parents(task_id).empty()) {
      current_tasks.insert(task_id);
      scheduled_tasks.insert(task_id);
      schedule(task_id);
    }
  }

  Timer timer;
  while (!current_tasks.empty()) {
    std::unordered_set<int> updated_tasks;
    std::unordered_set<int> next_tasks;
    updated_tasks.reserve(current_tasks.size());

    if (FLAGS_caffe2_dag_net_collect_stats) {
      timer.Start();
    }
    if (has_chain_failed_) {
      finishTasks(current_tasks);
      return false;
    }
    for (auto& task_id : current_tasks) {
      auto prev_status = status_[task_id];
      status_[task_id] = query(task_id);
      if (status_[task_id] == EventStatus::EVENT_FAILED) {
        finishTasks(current_tasks);
        return false;
      }

      if (prev_status != status_[task_id]) {
        updated_tasks.insert(task_id);
        if (FLAGS_caffe2_dag_net_collect_stats) {
          updateTaskStats(task_id);
        }
      }

      if (status_[task_id] != EventStatus::EVENT_SUCCESS) {
        next_tasks.insert(task_id);
      }
    }
    if (FLAGS_caffe2_dag_net_collect_stats) {
      CAFFE_EVENT(
          stats_[CPU], poll_status_update_time_us, timer.MicroSeconds());
    }

    std::unordered_set<int> visited_children;
    for (auto& task_id : updated_tasks) {
      PANIC_ENFORCE(
          status_[task_id] == EventStatus::EVENT_SCHEDULED ||
          status_[task_id] == EventStatus::EVENT_SUCCESS, "");

      for (auto& child_id : children(task_id)) {
        if (!visited_children.count(child_id)) {
          visited_children.insert(child_id);
          // Important - check whether we have already scheduled the task,
          // e.g. a child CUDA task can be scheduled after parent CUDA
          // task becomes EventStatus::EVENT_SCHEDULED and also later when
          // parent CUDA task becomes EventStatus::EVENT_SUCCESS
          if (!scheduled_tasks.count(child_id) &&
              canSchedule(child_id, &status_)) {
            next_tasks.insert(child_id);
            scheduled_tasks.insert(child_id);
            schedule(child_id);
          }
        }
      }
    }

    current_tasks.swap(next_tasks);
  }
  return true;
}

void AsyncPollingNet::updateTaskStats(int task_id) {
  const auto& device_option = event(task_id).GetDeviceOption();
  if (status_[task_id] == EventStatus::EVENT_SCHEDULED) {
    CAFFE_EVENT(
        stats_[device_option.device_type()],
        task_time_to_scheduled_us,
        task_timers_[task_id]->MicroSeconds());
  }
  if (status_[task_id] == EventStatus::EVENT_SUCCESS) {
    CAFFE_EVENT(
        stats_[device_option.device_type()],
        task_time_to_succeeded_ms,
        task_timers_[task_id]->MilliSeconds());
  }
}

AsyncPollingNet::~AsyncPollingNet() {}

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_NET_ASYNC_POLLING_H_