//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// worker.h
//
// Identification: src/threadpool/worker.h
//
// Copyright (c) 2015-17, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/include/threadpool/worker.h

#ifndef BUBBLEFS_UTILS_PELOTON_WORKER_H_
#define BUBBLEFS_UTILS_PELOTON_WORKER_H_

#include <thread>

#include "utils/peloton_task_queue.h"

namespace bubblefs {
namespace mypeloton{
namespace threadpool{
/**
 * @class Worker
 * @brief A worker that can execute task
 */
class Worker {
 public:
  void Start(TaskQueue *task_queue);

  void Stop();

  // execute
  static void Execute(Worker *current_thread, TaskQueue *task_queue);

 private:
  volatile bool shutdown_thread_;
  std::thread worker_thread_;
};

}  // namespace threadpool
}  // namespace mypeloton
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_WORKER_H_