/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// eigen/unsupported/Eigen/CXX11/src/ThreadPool/SimpleThreadPool.h
// tensorflow/tensorflow/core/lib/core/threadpool.cc

#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>
#include "platform/logging.h"
#include "platform/macros.h"
#include "platform/types.h"
#include "utils/threadpool.h"

namespace bubblefs {
namespace thread {
  
struct StlThreadEnvironment {
  struct Task {
    std::function<void()> f;
  };

  // EnvThread constructor must start the thread,
  // destructor must join the thread.
  class EnvThread {
   public:
    EnvThread(std::function<void()> f) : thr_(std::move(f)) {}
    ~EnvThread() { thr_.join(); }
    // This function is called when the threadpool is cancelled.
    void OnCancel() {}

   private:
    std::thread thr_;
  };

  EnvThread* CreateThread(std::function<void()> f) { return new EnvThread(std::move(f)); }
  Task CreateTask(std::function<void()> f) { return Task{std::move(f)}; }
  void ExecuteTask(const Task& t) { t.f(); }
}; 

// The implementation of the ThreadPool type ensures that the Schedule method
// runs the functions it is provided in FIFO order when the scheduling is done
// by a single thread.
// Environment provides a way to create threads and also allows to intercept
// task submission and execution.
template <typename Environment>
class SimpleThreadPoolTempl : public ThreadPoolInterface {
 public:
  // Construct a pool that contains "num_threads" threads.
  explicit SimpleThreadPoolTempl(int num_threads, Environment env = Environment())
      : env_(env), threads_(num_threads), waiters_(num_threads) {
    for (int i = 0; i < num_threads; i++) {
      threads_.push_back(env.CreateThread([this, i]() { WorkerLoop(i); }));
    }
  }

  // Wait until all scheduled work has finished and then destroy the
  // set of threads.
  ~SimpleThreadPoolTempl() {
    {
      // Wait for all work to get done.
      std::unique_lock<std::mutex> l(mu_);
      while (!pending_.empty()) {
        empty_.wait(l);
      }
      exiting_ = true;

      // Wakeup all waiters.
      for (auto w : waiters_) {
        w->ready = true;
        w->task.f = nullptr;
        w->cv.notify_one();
      }
    }

    // Wait for threads to finish.
    for (auto t : threads_) {
      delete t;
    }
  }

  // Schedule fn() for execution in the pool of threads. The functions are
  // executed in the order in which they are scheduled.
  void Schedule(std::function<void()> fn) final {
    Task t = env_.CreateTask(std::move(fn));
    std::unique_lock<std::mutex> l(mu_);
    if (waiters_.empty()) {
      pending_.push_back(std::move(t));
    } else {
      Waiter* w = waiters_.back();
      waiters_.pop_back();
      w->ready = true;
      w->task = std::move(t);
      w->cv.notify_one();
    }
  }
  
  void Cancel() {
    for (size_t i = 0; i < threads_.size(); i++) {
      threads_[i]->OnCancel();
    }
  }

  int NumThreads() const final {
    return static_cast<int>(threads_.size());
  }

  int CurrentThreadId() const final {
    const PerThread* pt = this->GetPerThread();
    if (pt->pool == this) {
      return pt->thread_id;
    } else {
      return -1;
    }
  }

 protected:
  void WorkerLoop(int thread_id) {
    std::unique_lock<std::mutex> l(mu_);
    PerThread* pt = GetPerThread();
    pt->pool = this;
    pt->thread_id = thread_id;
    Waiter w;
    Task t;
    while (!exiting_) {
      if (pending_.empty()) {
        // Wait for work to be assigned to me
        w.ready = false;
        waiters_.push_back(&w);
        while (!w.ready) {
          w.cv.wait(l);
        }
        t = w.task;
        w.task.f = nullptr;
      } else {
        // Pick up pending work
        t = std::move(pending_.front());
        pending_.pop_front();
        if (pending_.empty()) {
          empty_.notify_all();
        }
      }
      if (t.f) {
        mu_.unlock();
        env_.ExecuteTask(t);
        t.f = nullptr;
        mu_.lock();
      }
    }
  }

 private:
  typedef typename Environment::Task Task;
  typedef typename Environment::EnvThread Thread;

  struct Waiter {
    std::condition_variable cv;
    Task task;
    bool ready;
  };

  struct PerThread {
    constexpr PerThread() : pool(NULL), thread_id(-1) { }
    SimpleThreadPoolTempl* pool;  // Parent pool, or null for normal threads.
    int thread_id;                // Worker thread index in pool.
  };

  Environment env_;
  std::mutex mu_;
  std::vector<Thread*> threads_;  // All threads
  std::vector<Waiter*> waiters_;  // Stack of waiting threads.
  std::deque<Task> pending_;        // Queue of pending work
  std::condition_variable empty_;   // Signaled on pending_.empty()
  bool exiting_ = false;

  PerThread* GetPerThread() const {
    STATIC_THREAD_LOCAL PerThread per_thread;
    return &per_thread;
  }
};

typedef SimpleThreadPoolTempl<StlThreadEnvironment> SimpleThreadPool;  
  
struct ThreadPool::Impl : SimpleThreadPool {
  Impl(Env* env, const ThreadOptions& thread_options, const string& name,
       int num_threads, bool low_latency_hint): SimpleThreadPool(num_threads) { }
};

ThreadPool::ThreadPool(Env* env, const string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads, true) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads)
    : ThreadPool(env, thread_options, name, num_threads, true) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads,
                       bool low_latency_hint) {
  CHECK_GE(num_threads, 1);
  impl_.reset(new ThreadPool::Impl(env, thread_options, "tf_" + name,
                                   num_threads, low_latency_hint));
}

ThreadPool::~ThreadPool() {}

void ThreadPool::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  impl_->Schedule(std::move(fn));
}

int ThreadPool::NumThreads() const { return impl_->NumThreads(); }

int ThreadPool::CurrentThreadId() const { return impl_->CurrentThreadId(); }

}  // namespace thread
}  // namespace bubblefs