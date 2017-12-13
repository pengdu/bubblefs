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

// peloton/src/include/threadpool/worker_pool.h

#ifndef BUBBLEFS_UTILS_PELOTON_WORKER_POOL_H_
#define BUBBLEFS_UTILS_PELOTON_WORKER_POOL_H_

#include <vector>
#include <unistd.h>

#include "utils/peloton_task_queue.h"
#include "utils/peloton_worker.h"

namespace bubblefs {
namespace mypeloton{
namespace threadpool{

/**
 * @class WorkerPool
 * @brief A worker pool that maintains a group to worker thread
 */
class WorkerPool {
 public:
  // submit a threadpool for asynchronous execution.
  WorkerPool(const size_t num_workers, TaskQueue *task_queue);

  // explicitly start up the pool
  void Startup();

  // explicitly shut down the pool
  void Shutdown();

 private:
  friend class Worker;

  std::vector<std::unique_ptr<Worker>> workers_;

  size_t num_workers_;
  TaskQueue* task_queue_;
};

}  // namespace threadpool
}  // namespace mypeloton
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_WORKER_POOL_H_