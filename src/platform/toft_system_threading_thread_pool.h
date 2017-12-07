// Copyright (C) 2013, The Toft Authors.
// Author: An Qin <anqin.qin@gmail.com>
//
// Description:

// toft/system/threading/thread_pool.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THREAD_POOL_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THREAD_POOL_H_

#include <stdint.h>
#include <functional>

#include "platform/toft_system_threading_event.h"
#include "platform/toft_system_threading_mutex.h"
#include "platform/toft_system_threading_thread.h"
#include "utils/toft_base_closure.h"
#include "utils/toft_base_intrusive_list.h"

namespace bubblefs {
namespace mytoft {

class ThreadPool {
public:
    /// @param mun_threads number of threads, -1 means cpu number
    explicit ThreadPool(int num_threads = -1);
    ~ThreadPool();


    void AddTask(Closure<void ()>* callback);
    void AddTask(const std::function<void ()>& callback);

    void AddTask(Closure<void ()>* callback, int dispatch_key);
    void AddTask(const std::function<void ()>& callback, int dispatch_key);

    void WaitForIdle();
    void Terminate();

private:
    struct Task;
    struct ThreadContext;

    void AddTaskInternal(Closure<void ()>* callback,
                         const std::function<void ()>& function,
                         int dispatch_key);

    bool AnyTaskPending() const;
    void WorkRoutine(ThreadContext* thread);
    bool AnyThreadRunning() const;

private:
    ThreadContext* m_thread_contexts;
    size_t m_num_threads;
    size_t m_num_busy_threads;
    Mutex m_exit_lock;
    ConditionVariable m_exit_cond;
    bool m_exit;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_THREAD_POOL_H_