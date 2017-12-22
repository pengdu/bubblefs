// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// baidu/common/include/thread.h

#ifndef BUBBLEFS_UTILS_BDCOM_THREAD_H_
#define BUBBLEFS_UTILS_BDCOM_THREAD_H_

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <syscall.h>
#include <time.h>
#include <unistd.h>
#include <functional>
#include "platform/macros.h"

namespace bubblefs {
namespace mybdcom {

// simple thread implement
class Thread {
public:
    Thread() { 
      memset(&tid_, 0, sizeof(tid_));
    }
    static int64_t CurrentThreadId() {
        return static_cast<int64_t>(pthread_self());
    }
    static void Exit() {
        pthread_exit(nullptr);
    }
    static void YieldCurrentThread() {
        sched_yield();
    }
    static void Sleep(struct timespec &duration) {
        struct timespec sleep_time, remaining;

        // Break the duration into seconds and nanoseconds.
        // NOTE: TimeDelta's microseconds are int64s while timespec's
        // nanoseconds are longs, so this unpacking must prevent overflow.
        sleep_time.tv_sec = duration.tv_sec;
        sleep_time.tv_nsec = duration.tv_nsec;  // nanoseconds

        while (nanosleep(&sleep_time, &remaining) == -1 && errno == EINTR)
          sleep_time = remaining;
    }
    const int64_t GetThreadId() const {
        return static_cast<int64_t>(tid_);
    }
    bool IsStarted() const {
        return (0 != tid_);
    }
    bool AmSelf() const {
        return (pthread_self() == tid_);
    }
    bool Start(std::function<void ()> thread_proc) {
        user_proc_ = thread_proc;
        int ret = pthread_create(&tid_, nullptr, ProcWrapper, this);
        return (ret == 0);
    }
    typedef void* (Proc)(void*);
    bool Start(Proc proc, void* arg) {
        int ret = pthread_create(&tid_, nullptr, proc, arg);
        return (ret == 0);
    }
    bool Start(Proc proc, void* arg, size_t stack_size, bool joinable = true);
    bool Join() {
        int ret = pthread_join(tid_, nullptr);
        return (ret == 0);
    }
    bool Kill(int signal) {
      int ret = 0;
      if (tid_)
        ret = pthread_kill(tid_, signal);
      return (ret == 0);
    }
    bool Detach() {
      int ret = pthread_detach(tid_);
      return (ret == 0);
    }
private:
    static void* ProcWrapper(void* arg) {
        reinterpret_cast<Thread*>(arg)->user_proc_();
        return nullptr;
    }
    void set_thread_attrs();
    DISALLOW_COPY_AND_ASSIGN(Thread);
private:
    std::function<void ()> user_proc_;
    pthread_t tid_;
};

bool Thread::Start(Proc proc, void* arg, size_t stack_size, bool joinable) {
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  // Pthreads are joinable by default, so only specify the detached
  // attribute if the thread should be non-joinable.
  if (!joinable) {
    pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
  }
  // Get a better default if available.
  if (stack_size <= 0)
    stack_size = 2 * (1 << 23);  // 2 times 8192K (the default stack size on Linux).
  pthread_attr_setstacksize(&attributes, stack_size);
        
  // The child thread will inherit our signal mask.
  // Set our signal mask to the set of signals we want to block ?
        
  int ret = pthread_create(&tid_, nullptr, proc, arg);
        
  pthread_attr_destroy(&attributes);
  return (ret == 0);
}

} // namespace mybdcom
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOM_THREAD_H_