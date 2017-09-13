// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// baidu/common/include/thread.h

#ifndef  BUBBLEFS_UTILS_THREAD_H_
#define  BUBBLEFS_UTILS_THREAD_H_

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>
#include <syscall.h>
#include <time.h>
#include <unistd.h>
#include <functional>
#include "platform/macros.h"

namespace bubblefs {
namespace baiducomm {

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
    Thread() {
        memset(&tid_, 0, sizeof(tid_));
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
    bool Join() {
        int ret = pthread_join(tid_, nullptr);
        return (ret == 0);
    }
    void Exit() {
        pthread_exit(nullptr);
    }
private:
    static void* ProcWrapper(void* arg) {
        reinterpret_cast<Thread*>(arg)->user_proc_();
        return nullptr;
    }
    TF_DISALLOW_COPY_AND_ASSIGN(Thread);
private:
    std::function<void ()> user_proc_;
    pthread_t tid_;
};

} // namespace baiducomm
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_THREAD_H_