//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// task_queue.h
//
// Identification: src/threadpool/task_queue.h
//
// Copyright (c) 2015-17, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/include/threadpool/task_queue.h

#ifndef BUBBLEFS_UTILS_PELOTON_TASK_QUEUE_H_
#define BUBBLEFS_UTILS_PELOTON_TASK_QUEUE_H_

#include <memory>
#include "utils/peloton_blocking_queue.h"
#include "utils/peloton_task.h"

namespace bubblefs {
namespace mypeloton {
namespace threadpool {

/**
 * @class TaskQueue
 * @brief A queue for user to submit task and for user to poll tasks
 */
class TaskQueue {
 public:
  TaskQueue(const size_t size) : task_queue_(size) {};

  bool Poll(std::shared_ptr<Task> &task) {
    return task_queue_.Dequeue(task);
  }

  bool IsEmpty() { return task_queue_.IsEmpty(); }

  void Enqueue(void (*task_ptr)(void *), void *task_arg,
               void (*callback_ptr)(void *), void *callback_arg) {
    std::shared_ptr<Task> task =
        std::make_shared<Task>(task_ptr, task_arg, callback_ptr, callback_arg);

    task_queue_.Enqueue(task);
  }

 private:
  // peloton::LockFreeQueue<std::shared_ptr<Task>> task_queue_;
  BlockingQueue<std::shared_ptr<Task>> task_queue_;
};

}  // namespace threadpool
}  // namespace peloton
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_TASK_QUEUE_H_