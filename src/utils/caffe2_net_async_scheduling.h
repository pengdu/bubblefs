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

// caffe2/caffe2/core/net_async_scheduling.h
// caffe2/caffe2/core/net_async_scheduling.cc

#ifndef BUBBLEFS_UTILS_CAFFE2_NET_ASYNC_SCHEDULING_H_
#define BUBBLEFS_UTILS_CAFFE2_NET_ASYNC_SCHEDULING_H_

#include "utils/caffe2_net_async_base.h"
#include "utils/caffe2_unique_ptr.h"

namespace bubblefs {
namespace mycaffe2 {

extern bool FLAGS_caffe2_net_async_always_schedule_child = false; // "Always schedule child chains from parent chain");

extern int FLAGS_caffe2_net_async_polling_threads_num = 1; // "Number of polling threads in async_scheduling executor");  
  
class AsyncSchedulingNet : public AsyncNetBase {
 public:
  AsyncSchedulingNet(
      const std::shared_ptr<const NetDef>& net_def,
      Workspace* ws);
  ~AsyncSchedulingNet() override;

  void Wait() override;

 protected:
  bool DoRunAsync() override;

  void pollAndSchedule(int thread_id);
  void schedule(int task_id);
  void reset();
  void finishRun();
  int updateParentCount(int child_id);

  std::mutex running_mutex_;
  std::condition_variable running_cv_;
  std::atomic<bool> running_;
  std::atomic<bool> success_;

  std::mutex cleanup_mutex_;
  std::atomic<bool> cleanup_;

  std::atomic<int> processed_tasks_num_;

  std::vector<std::unique_ptr<SimpleQueue<int>>> pending_tasks_;
  std::vector<std::thread> polling_threads_;
  std::atomic<int> next_polling_thread_counter_;

  DISALLOW_COPY_AND_ASSIGN(AsyncSchedulingNet);
};

AsyncSchedulingNet::AsyncSchedulingNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : AsyncNetBase(net_def, ws), running_(false) {
  pending_tasks_.reserve(FLAGS_caffe2_net_async_polling_threads_num);
  for (auto thread_num = 0;
       thread_num < FLAGS_caffe2_net_async_polling_threads_num;
       ++thread_num) {
    pending_tasks_.push_back(mycaffe2::make_unique<SimpleQueue<int>>());
  }

  polling_threads_.reserve(FLAGS_caffe2_net_async_polling_threads_num);
  for (auto thread_num = 0;
       thread_num < FLAGS_caffe2_net_async_polling_threads_num;
       ++thread_num) {
    polling_threads_.push_back(
        std::thread(&AsyncSchedulingNet::pollAndSchedule, this, thread_num));
  }

  reset();
}

void AsyncSchedulingNet::reset() {
  processed_tasks_num_ = 0;
  cleanup_ = false;
  success_ = true;
  next_polling_thread_counter_ = 0;

  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    auto& task_ops = chains_[task_id];
    auto& task_op_node = operator_nodes_[task_ops.front()];
    task_op_node.runtime_parent_count_ = parents(task_id).size();
  }
}

void AsyncSchedulingNet::Wait() {
  std::unique_lock<std::mutex> lock(running_mutex_);
  while (running_) {
    running_cv_.wait(lock);
  }
}

void AsyncSchedulingNet::schedule(int task_id) {
  const auto& device_option = event(task_id).GetDeviceOption();
  pool(device_option)->run([this, task_id]() {
    if (success_) {
      int stream_id = stream(task_id);
      asyncWait(task_id, stream_id, parents(task_id));
      if (!run(task_id, stream_id)) {
        success_ = false;
      }
    }

    auto task_count = ++processed_tasks_num_;

    for (auto child_id : children(task_id)) {
      int parent_count = updateParentCount(child_id);
      if (parent_count == 0) {
        if (cleanup_ || FLAGS_caffe2_net_async_always_schedule_child ||
            canSchedule(child_id)) {
          schedule(child_id);
        } else {
          auto polling_thread_id = next_polling_thread_counter_++;
          polling_thread_id %= FLAGS_caffe2_net_async_polling_threads_num;
          pending_tasks_[polling_thread_id]->Push(child_id);
        }
      }
    }

    if (success_) {
      if (task_count == tasksNum()) {
        // All tasks are finished, polling thread is sleeping;
        // only one thread enters here
        finalizeEvents();
        finishRun();
        return;
      }
    } else {
      // Before setting running_ to false and notifying waiters we need to
      // 1. Ensure that only one thread does the cleanup
      // 2. Ensure that all other pending tasks in workers and polling threads
      //    are finished and
      // 3. Ensure that all tasks that were not scheduled have their events set
      {
        std::unique_lock<std::mutex> cleanup_lock(cleanup_mutex_);
        if (cleanup_) {
          return;
        }
        cleanup_ = true;
      }

      // Errors are not recoverable and happen in exceptional cases,
      // ok to busy wait
      while (processed_tasks_num_ != tasksNum()) {
      }

      // Make sure all events are set, wait for scheduled events
      finalizeEvents();

      // Notify observers and waiters
      finishRun();
    }
  });
}

void AsyncSchedulingNet::pollAndSchedule(int thread_id) {
  int task_id;
  while (pending_tasks_[thread_id]->Pop(&task_id)) {
    if (canSchedule(task_id) || cleanup_) {
      // force schedule the rest of the tasks if cleanup is started
      schedule(task_id);
    } else {
      pending_tasks_[thread_id]->Push(task_id);
    }
  }
}

int AsyncSchedulingNet::updateParentCount(int child_id) {
  auto& child_ops = chains_[child_id];
  auto& child_node = operator_nodes_[child_ops.front()];
  int parent_count = --child_node.runtime_parent_count_;
  PANIC_ENFORCE_GE(parent_count, 0);
  return parent_count;
}

void AsyncSchedulingNet::finishRun() {
  // notify observers and waiters
  StopAllObservers();
  running_ = false;
  running_cv_.notify_all();
}

bool AsyncSchedulingNet::DoRunAsync() {
  std::unique_lock<std::mutex> lock(running_mutex_);
  PANIC_ENFORCE(!running_, "Concurrent RunAsync calls");
  running_ = true;
  reset();

  StartAllObservers();

  for (auto task_id = 0; task_id < tasksNum(); ++task_id) {
    if (parents(task_id).empty()) {
      schedule(task_id);
    }
  }

  return true;
}

AsyncSchedulingNet::~AsyncSchedulingNet() {
  for (auto& task_queue : pending_tasks_) {
    task_queue->NoMoreJobs();
  }
  for (auto& polling_thread : polling_threads_) {
    polling_thread.join();
  }
}

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_NET_ASYNC_SCHEDULING_H_