// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// baidu/common/include/thread.h

#ifndef  BUBBLEFS_UTILS_THREAD_H_
#define  BUBBLEFS_UTILS_THREAD_H_

#include <pthread.h>
#include <string.h>
#include <functional>

namespace bubblefs {
namespace baiducomm {

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
    Thread(const Thread&);
    void operator=(const Thread&);
    static void* ProcWrapper(void* arg) {
        reinterpret_cast<Thread*>(arg)->user_proc_();
        return NULL;
    }
private:
    std::function<void ()> user_proc_;
    pthread_t tid_;
};

} // namespace baiducomm
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_THREAD_H_