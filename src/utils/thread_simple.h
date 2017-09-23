// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// baidu/common/include/thread.h

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <syscall.h>
#include <time.h>
#include <unistd.h>
#include <functional>
#include <boost/concept_check.hpp>
#include "platform/io_prio.h"
#include "platform/macros.h"

namespace bubblefs {
namespace bdcommon {

class ThisThread {
public:
    /// Sleep in ms
    static void Sleep(int64_t time_ms) {
        if (time_ms > 0) {
            struct timespec ts = {time_ms / 1000, (time_ms % 1000) * 1000000 };
            nanosleep(&ts, &ts);
        }
    }
    /// Get thread id
    static int GetId() {
        static __thread int s_thread_id = 0;
        if (s_thread_id == 0) {
            s_thread_id = syscall(__NR_gettid);
        }
        return s_thread_id;
    }
    /// Yield cpu
    static void Yield() {
        sched_yield();
    }
};  
  
class ThreadAttributes {
public:
    ThreadAttributes() {
        cpu_num_ = sysconf(_SC_NPROCESSORS_CONF);
        mask_ = GetCpuAffinity();
    }
    ~ThreadAttributes() {}

    int32_t GetCpuNum() {
        return cpu_num_;
    }

    cpu_set_t GetCpuAffinity() {
        ResetCpuMask();
        if (sched_getaffinity(0, sizeof(mask_), &mask_) == -1) {
            ResetCpuMask();
        }
        return mask_;
    }
    bool SetCpuAffinity() {
        if (sched_setaffinity(0, sizeof(mask_), &mask_) == -1) {
            return false;
        }
        return true;
    }

    bool SetCpuMask(int32_t cpu_id) {
        if (cpu_id < 0 || cpu_id > cpu_num_) {
            return false;
        }

        if (CPU_ISSET(cpu_id, &mask_)) {
            return true;
        }
        CPU_SET(cpu_id, &mask_);
        return true;
    }
    void ResetCpuMask() {
        CPU_ZERO(&mask_);
    }
    void MarkCurMask() {
        CPU_ZERO(&last_mask_);
        last_mask_ = mask_;
    }
    bool RevertCpuAffinity() {
        ResetCpuMask();
        mask_ = last_mask_;
        return SetCpuAffinity();
    }

private:
    int32_t cpu_num_;
    cpu_set_t mask_;
    cpu_set_t last_mask_;
};
  
// simple thread implement
class Thread {
public:
    Thread() : tid_(0), pid_(0), ioprio_class_(-1), ioprio_prio_(-1),
               cpuid_(-1), thread_name_("ThreadUnknown") { }
    static int64_t CurrentId() {
        return static_cast<int64_t>(pthread_self());
    }
    const pthread_t GetThreadId() const {
        return tid_;
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
    bool Start(Proc proc, void* arg, std::string name, size_t stack_size, bool joinable = true);
    bool Join() {
        int ret = pthread_join(tid_, nullptr);
        return (ret == 0);
    }
    void Exit() {
        pthread_exit(nullptr);
    }
    void YieldCurrentThread() {
        sched_yield();
    }
    void Sleep(struct timespec &duration) {
        struct timespec sleep_time, remaining;

        // Break the duration into seconds and nanoseconds.
        // NOTE: TimeDelta's microseconds are int64s while timespec's
        // nanoseconds are longs, so this unpacking must prevent overflow.
        sleep_time.tv_sec = duration.tv_sec;
        sleep_time.tv_nsec = duration.tv_nsec;  // nanoseconds

        while (nanosleep(&sleep_time, &remaining) == -1 && errno == EINTR)
          sleep_time = remaining;
    }
    bool kill(int signal) {
      int ret = 0;
      if (tid_)
        ret = pthread_kill(tid_, signal);
      return (ret == 0);
    }
    bool Detach() {
      int ret = pthread_detach(tid_);
      return (ret == 0);
    }
    void SetIoprio(int cls, int prio);
    bool SetAffinity(int id);
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
    pid_t pid_;
    int ioprio_class_, ioprio_prio_;
    int cpuid_;
    std::string thread_name_;
};

} // namespace bdcommon
} // namespace bubblefs