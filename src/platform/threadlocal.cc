/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
// Copyright (c) 2011 Baidu, Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
 
// Author: Ge,Jun (gejun@baidu.com)
// Date: Mon. Nov 7 14:47:36 CST 2011

// Paddle/paddle/utils/ThreadLocal.cpp
// brpc/src/butil/thread_local.cpp

#include "platform/threadlocal.h"
#include <sys/syscall.h>
#include <errno.h>                       // errno
#include <pthread.h>                     // pthread_key_t
#include <stdio.h>
#include <algorithm>                     // std::find
#include <vector>
#include "platform/mutexlock.h"
#include "platform/port.h"
#include "gflags/gflags.h"

DEFINE_bool(thread_local_rand_use_global_seed,
            false,
            "Whether to use global seed in thread local rand.");

namespace bubblefs {
  
namespace internal {
  
unsigned int ThreadLocalRand::defaultSeed_ = 1;
ThreadLocal<unsigned int> ThreadLocalRand::seed_;

unsigned int* ThreadLocalRand::getSeed() {
  unsigned int* p = seed_.get(false /*createLocal*/);
  if (!p) {  // init seed
    if (FLAGS_thread_local_rand_use_global_seed) {
      p = new unsigned int(defaultSeed_);
    } else if (getpid() == gettid()) {  // main thread
      // deterministic, but differs from global srand()
      p = new unsigned int(defaultSeed_ - 1);
    } else {
      p = new unsigned int(defaultSeed_ + gettid());
      VLOG(3) << "thread use undeterministic rand seed:" << *p;
    }
    seed_.set(p);
  }
  return p;
}

ThreadLocal<std::default_random_engine> ThreadLocalRandomEngine::engine_;
std::default_random_engine& ThreadLocalRandomEngine::get() {
  auto engine = engine_.get(false);
  if (!engine) {
    engine = new std::default_random_engine;
    int defaultSeed = ThreadLocalRand::getDefaultSeed();
    engine->seed(FLAGS_thread_local_rand_use_global_seed
                     ? defaultSeed
                     : defaultSeed + getTID());
    engine_.set(engine);
  }
  return *engine;
}

}  // namespace internal

namespace base {
  
namespace detail {

class ThreadExitHelper {
public:
    typedef void (*Fn)(void*);
    typedef std::pair<Fn, void*> Pair;
    
    ~ThreadExitHelper() {
        // Call function reversely.
        while (!_fns.empty()) {
            Pair back = _fns.back();
            _fns.pop_back();
            // Notice that _fns may be changed after calling Fn.
            back.first(back.second);
        }
    }

    int add(Fn fn, void* arg) {
        try {
            if (_fns.capacity() < 16) {
                _fns.reserve(16);
            }
            _fns.push_back(std::make_pair(fn, arg));
        } catch (...) {
            errno = ENOMEM;
            return -1;
        }
        return 0;
    }

    void remove(Fn fn, void* arg) {
        std::vector<Pair>::iterator
            it = std::find(_fns.begin(), _fns.end(), std::make_pair(fn, arg));
        if (it != _fns.end()) {
            std::vector<Pair>::iterator ite = it + 1;
            for (; ite != _fns.end() && ite->first == fn && ite->second == arg;
                  ++ite) {}
            _fns.erase(it, ite);
        }
    }

private:
    std::vector<Pair> _fns;
};

static pthread_key_t thread_atexit_key;
static pthread_once_t thread_atexit_once = PTHREAD_ONCE_INIT;

static void delete_thread_exit_helper(void* arg) {
    delete static_cast<ThreadExitHelper*>(arg);
}

static void helper_exit_global() {
    detail::ThreadExitHelper* h = 
        (detail::ThreadExitHelper*)pthread_getspecific(detail::thread_atexit_key);
    if (h) {
        pthread_setspecific(detail::thread_atexit_key, nullptr);
        delete h;
    }
}

static void make_thread_atexit_key() {
    if (pthread_key_create(&thread_atexit_key, delete_thread_exit_helper) != 0) {
        fprintf(stderr, "Fail to create thread_atexit_key, abort\n");
        abort();
    }
    // If caller is not pthread, delete_thread_exit_helper will not be called.
    // We have to rely on atexit().
    atexit(helper_exit_global);
}

detail::ThreadExitHelper* get_or_new_thread_exit_helper() {
    pthread_once(&detail::thread_atexit_once, detail::make_thread_atexit_key);

    detail::ThreadExitHelper* h =
        (detail::ThreadExitHelper*)pthread_getspecific(detail::thread_atexit_key);
    if (nullptr == h) {
        h = new (std::nothrow) detail::ThreadExitHelper;
        if (nullptr != h) {
            pthread_setspecific(detail::thread_atexit_key, h);
        }
    }
    return h;
}

detail::ThreadExitHelper* get_thread_exit_helper() {
    pthread_once(&detail::thread_atexit_once, detail::make_thread_atexit_key);
    return (detail::ThreadExitHelper*)pthread_getspecific(detail::thread_atexit_key);
}

static void call_single_arg_fn(void* fn) {
    ((void (*)())fn)();
}

}  // namespace detail

int thread_atexit(void (*fn)(void*), void* arg) {
    if (nullptr == fn) {
        errno = EINVAL;
        return -1;
    }
    detail::ThreadExitHelper* h = detail::get_or_new_thread_exit_helper();
    if (h) {
        return h->add(fn, arg);
    }
    errno = ENOMEM;
    return -1;
}

int thread_atexit(void (*fn)()) {
    if (nullptr == fn) {
        errno = EINVAL;
        return -1;
    }
    return thread_atexit(detail::call_single_arg_fn, (void*)fn);
}

void thread_atexit_cancel(void (*fn)(void*), void* arg) {
    if (fn != nullptr) {
        detail::ThreadExitHelper* h = detail::get_thread_exit_helper();
        if (h) {
            h->remove(fn, arg);
        }
    }
}

void thread_atexit_cancel(void (*fn)()) {
    if (nullptr != fn) {
        thread_atexit_cancel(detail::call_single_arg_fn, (void*)fn);
    }
}

}  // namespace base

}  // namespace bubblefs
