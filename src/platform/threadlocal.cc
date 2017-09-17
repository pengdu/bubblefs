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

// Paddle/paddle/utils/ThreadLocal.cpp
// rocksdb/util/thread_local.cc

#include "platform/threadlocal.h"
#include <sys/syscall.h>
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

}  // namespace bubblefs
