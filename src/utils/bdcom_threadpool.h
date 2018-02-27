// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// baidu/common/include/thread_pool.h

#ifndef BUBBLEFS_UTILS_BDCOM_THREAD_POOL_H_
#define BUBBLEFS_UTILS_BDCOM_THREAD_POOL_H_

#include <time.h>
#include <deque>
#include <functional>
#include <map>
#include <queue>
#include <sstream>
#include <vector>
#include "platform/mutexlock.h"

namespace bubblefs {
namespace mybdcom {

static const int kDebugCheckTime = 5000;

// An unscalable thread pool implimention.
class ThreadPool {
public:
    ThreadPool(int thread_num = 10)
        : threads_num_(thread_num),
          pending_num_(0),
          work_cv_(&mutex_),
          stop_(false),
          last_task_id_(0),
          running_task_id_(0),
          schedule_cost_sum_(0),
          schedule_count_(0),
          task_cost_sum_(0),
          task_count_(0) {
        Start();
    }
    ~ThreadPool() {
        Stop(false);
    }
    // Start a thread_num threads pool.
    bool Start() {
        MutexLock lock(&mutex_);
        if (tids_.size()) {
            return false;
        }
        stop_ = false;
        for (int i = 0; i < threads_num_; i++) {
            pthread_t tid;
            int ret = pthread_create(&tid, nullptr, ThreadWrapper, this);
            if (ret) {
                abort();
            }
            tids_.push_back(tid);
        }
        return true;
    }

    // Stop the thread pool.
    // Wait for all pending task to complete if wait is true.
    bool Stop(bool wait) {
        if (wait) {
            while (pending_num_ > 0) {
                usleep(10000);
            }
        }

        {
            MutexLock lock(&mutex_);
            stop_ = true;
            work_cv_.Broadcast();
        }
        for (uint32_t i = 0; i < tids_.size(); i++) {
            pthread_join(tids_[i], nullptr);
        }
        tids_.clear();
        return true;
    }

    // Task definition.
    typedef std::function<void ()> Task;

    // Add a task to the thread pool.
    void AddTask(const Task& task) {
        MutexLock lock(&mutex_, "AddTask");
        if (stop_) return;
        queue_.push_back(BGItem(0, get_micros(), task));
        ++pending_num_;
        work_cv_.Signal();
    }
    void AddPriorityTask(const Task& task) {
        MutexLock lock(&mutex_);
        if (stop_) return;
        queue_.push_front(BGItem(0, get_micros(), task));
        ++pending_num_;
        work_cv_.Signal();
    }
    int64_t DelayTask(int64_t delay, const Task& task) {
        MutexLock lock(&mutex_);
        if (stop_) return 0;
        int64_t now_time = get_micros();
        int64_t exe_time = now_time + delay * 1000;
        BGItem bg_item(++last_task_id_, exe_time, task); // id > 1
        time_queue_.push(bg_item);
        latest_[bg_item.id] = bg_item;
        work_cv_.Signal();
        return bg_item.id;
    }
    /// Cancel a delayed task
    /// if running, wait if non_block==false; return immediately if non_block==true
    bool CancelTask(int64_t task_id, bool non_block = false, bool* is_running = nullptr) {
        if (task_id == 0) { // not delay task
            if (is_running != nullptr) {
                *is_running = false;
            }
            return false;
        }
        while (1) {
            {
                MutexLock lock(&mutex_);
                if (running_task_id_ != task_id) {
                    BGMap::iterator it = latest_.find(task_id);
                    if (it == latest_.end()) {
                        if (is_running != nullptr) {
                            *is_running = false;
                        }
                        return false;
                    }
                    latest_.erase(it);  // cancel task
                    return true;
                } else if (non_block) { // already running
                    if (is_running != nullptr) {
                        *is_running = true;
                    }
                    return false;
                }
                // else block
            }
            struct timespec ts = {0, 100000};
            nanosleep(&ts, &ts);
        }
    }
    int64_t PendingNum() const {
        return pending_num_;
    }

    // log format: 3 numbers seperated by " ", e.g. "15 24 32"
    // 1st: thread pool schedule average cost (ms)
    // 2nd: user task average cost (ms)
    // 3rd: total task count since last ProfilingLog called
    std::string ProfilingLog() {
        int64_t schedule_cost_sum;
        int64_t schedule_count;
        int64_t task_cost_sum;
        int64_t task_count;
        {
            MutexLock lock(&mutex_);
            schedule_cost_sum = schedule_cost_sum_;
            schedule_cost_sum_ = 0;
            schedule_count = schedule_count_;
            schedule_count_ = 0;
            task_cost_sum = task_cost_sum_;
            task_cost_sum_ = 0;
            task_count = task_count_;
            task_count_ = 0;
        }
        std::stringstream ss;
        ss << (schedule_count == 0 ? 0 : schedule_cost_sum / schedule_count / 1000)
            << " " << (task_count == 0 ? 0 : task_cost_sum / task_count / 1000)
            << " " << task_count;
        return ss.str();
    }

private:
    ThreadPool(const ThreadPool&);
    void operator=(const ThreadPool&);

    static void* ThreadWrapper(void* arg) {
        reinterpret_cast<ThreadPool*>(arg)->ThreadProc();
        return nullptr;
    }
    static int64_t get_micros() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return static_cast<int64_t>(ts.tv_sec) * 1000000 + static_cast<int64_t>(ts.tv_nsec) / 1000; // us
    }
    void ThreadProc() {
        while (true) {
            Task task;
            MutexLock lock(&mutex_, "ThreadProc");
            while (time_queue_.empty() && queue_.empty() && !stop_) {
                work_cv_.Wait();
            }
            if (stop_) {
                break;
            }
            // Timer task
            if (!time_queue_.empty()) {
                int64_t now_time = get_micros();
                BGItem bg_item = time_queue_.top();
                int64_t wait_time = (bg_item.exe_time - now_time) / 1000; // in ms
                if (wait_time <= 0) {
                    time_queue_.pop();
                    BGMap::iterator it = latest_.find(bg_item.id);
                    if (it != latest_.end() && it->second.exe_time == bg_item.exe_time) {
                        schedule_cost_sum_ += now_time - bg_item.exe_time;
                        schedule_count_++;
                        task = bg_item.task;
                        latest_.erase(it);
                        running_task_id_ = bg_item.id;
                        mutex_.Unlock();
                        task(); // not use mutex in task, may call threadpool funcs
                        task_cost_sum_ += get_micros() - now_time;
                        task_count_++;
                        mutex_.Lock("ThreadProcRelock");
                        running_task_id_ = 0;
                    }
                    continue;
                } else if (queue_.empty() && !stop_) {
                    work_cv_.TimedWait(wait_time);
                    continue;
                }
            }
            // Normal task;
            if (!queue_.empty()) {
                task = queue_.front().task;
                int64_t exe_time = queue_.front().exe_time;
                queue_.pop_front();
                --pending_num_;
                int64_t start_time = get_micros();
                schedule_cost_sum_ += start_time - exe_time;
                schedule_count_++;
                mutex_.Unlock();
                task();
                int64_t finish_time = get_micros();
                task_cost_sum_ += finish_time - start_time;
                task_count_++;
                mutex_.Lock("ThreadProcRelock2");
            }
        }
    }

private:
    struct BGItem {
        int64_t id;
        int64_t exe_time;
        Task task;
        bool operator<(const BGItem& item) const { // top is min-heap
            if (exe_time != item.exe_time) {
                return exe_time > item.exe_time;
            } else {
                return id > item.id;
            }
        }

        BGItem() {}
        BGItem(int64_t id_t, int64_t exe_time_t, const Task& task_t)
            : id(id_t), exe_time(exe_time_t), task(task_t) {}
    };
    typedef std::priority_queue<BGItem> BGQueue;
    typedef std::map<int64_t, BGItem> BGMap;

    int32_t threads_num_;
    std::deque<BGItem> queue_;
    volatile int pending_num_;
    port::Mutex mutex_;
    port::CondVar work_cv_;
    bool stop_;
    std::vector<pthread_t> tids_;

    BGQueue time_queue_;
    BGMap latest_;
    int64_t last_task_id_;
    int64_t running_task_id_;

    // for profiling
    int64_t schedule_cost_sum_;
    int64_t schedule_count_;
    int64_t task_cost_sum_;
    int64_t task_count_;
};

} // namespace mybdcom
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_BDCOM_THREAD_POOL_H_