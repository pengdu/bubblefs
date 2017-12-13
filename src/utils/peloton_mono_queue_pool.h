//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// mono_queue_pool.h
//
// Identification: test/threadpool/mono_queue_pool.h
//
// Copyright (c) 2015-17, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// peloton/src/include/threadpool/mono_queue_pool.h

#ifndef BUBBLEFS_UTILS_PELOTON_MONO_QUEUE_POOL_H_
#define BUBBLEFS_UTILS_PELOTON_MONO_QUEUE_POOL_H_

#include "utils/peloton_task_queue.h"
#include "utils/peloton_worker_pool.h"

namespace bubblefs {
namespace mypeloton {
namespace threadpool {
/**
 * @class MonoQueuePool
 * @brief Wrapper class for single queue and single pool
 * One should use this if possible.
 */
class MonoQueuePool {
 public:
  MonoQueuePool() : task_queue_(4),
                    worker_pool_(32, &task_queue_),
                    startup_(false) {}
  ~MonoQueuePool() {
    if (startup_ == true)
      Shutdown();
  }

  void Startup() {
    worker_pool_.Startup();
    startup_ = true;
  }

  void Shutdown() {
    worker_pool_.Shutdown();
    startup_ = false;
  }

  void SubmitTask(void (*task_ptr)(void *), void *task_arg,
                  void (*callback_ptr)(void *), void *callback_arg) {
    if (startup_ == false)
      Startup();
    task_queue_.Enqueue(task_ptr, task_arg, callback_ptr, callback_arg);
  }

  static MonoQueuePool &GetInstance() {
    static MonoQueuePool mono_queue_pool;
    return mono_queue_pool;
  }

 private:
  TaskQueue task_queue_;
  WorkerPool worker_pool_;
  bool startup_;
};

} // namespace threadpool
} // namespace mypeloton
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_PELOTON_MONO_QUEUE_POOL_H_