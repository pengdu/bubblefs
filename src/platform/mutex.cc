//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//

// rocksdb/port/port_posix.cc

#include "platform/mutex.h"
#include <sys/time.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

namespace bubblefs {
  
namespace port {
  
static void make_timeout(struct timespec* pts, long millisecond) {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    pts->tv_sec = millisecond / 1000 + tv.tv_sec;
    pts->tv_nsec = (millisecond % 1000) * 1000000 + tv.tv_usec * 1000;

    pts->tv_sec += pts->tv_nsec / 1000000000;
    pts->tv_nsec = pts->tv_nsec % 1000000000;
}

static int PthreadCall(const char* label, int result) {
  if (result != 0) {
    fprintf(stderr, "pthreadcall %s: %s\n", label, strerror(result));
    abort();
  }
  return result;
}

Mutex::Mutex() : owner_(0) {
  pthread_mutexattr_t attr;
  PthreadCall("init mutexattr", pthread_mutexattr_init(&attr));
  PthreadCall("set mutexattr errorcheck", pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK));
  PthreadCall("init mutex errorcheck", pthread_mutex_init(&mu_, &attr));
  PthreadCall("destroy mutexattr errorcheck", pthread_mutexattr_destroy(&attr));
}

Mutex::Mutex(bool adaptive) : owner_(0) {
  if (!adaptive) {
    PthreadCall("init mutex default", pthread_mutex_init(&mu_, nullptr));
  } else {
    pthread_mutexattr_t mutex_attr;
    PthreadCall("init mutexattr", pthread_mutexattr_init(&mutex_attr));
    PthreadCall("set mutexattr adaptive_np", pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_ADAPTIVE_NP));
    PthreadCall("init mutex adaptive_np", pthread_mutex_init(&mu_, &mutex_attr));
    PthreadCall("destroy mutexattr adaptive_np", pthread_mutexattr_destroy(&mutex_attr));
  }
}

Mutex::~Mutex() { PthreadCall("destroy mutex", pthread_mutex_destroy(&mu_)); }

void Mutex::Lock(const char* msg, int64_t msg_threshold) {
  PthreadCall("mutex lock", pthread_mutex_lock(&mu_));
  AfterLock(msg);
}

bool Mutex::TryLock() {
  int ret = pthread_mutex_trylock(&mu_);
  switch (ret) {
    case 0: AfterLock(); return true;
    case EBUSY: return false;
    case EINVAL: abort();
    case EAGAIN: abort();
    case EDEADLK: abort();
    default: abort();
  }
  return false;
}

bool Mutex::TimedLock(long _millisecond) {
  struct timespec ts;
  make_timeout(&ts, _millisecond);
  int ret =  pthread_mutex_timedlock(&mu_, &ts);
  switch (ret) {
    case 0: AfterLock(); return true;
    case ETIMEDOUT: return false;
    case EAGAIN: abort();
    case EDEADLK: abort();
    case EINVAL: abort();
    default: abort();
  }
  return false;
}

void Mutex::Unlock() {
  BeforeUnlock();
  PthreadCall("mutex unlock", pthread_mutex_unlock(&mu_));
}

bool Mutex::IsLocked() {
    int ret = pthread_mutex_trylock(&mu_);
    if (0 == ret) Unlock();
    return 0 != ret;
}

void Mutex::AssertHeld() {
#ifndef NDEBUG
  assert(locked_);
#endif
  if (0 == pthread_equal(owner_, pthread_self())) {
    fprintf(stderr, "mutex is held by two calling threads %lu:%lu\n",
            (unsigned long)owner_, (unsigned long)pthread_self());
    abort();
  }
}

void Mutex::AfterLock(const char* msg, int64_t msg_threshold) {
#ifndef NDEBUG
  locked_ = true;
#endif
  owner_ = pthread_self();
}

void Mutex::BeforeUnlock(const char* msg) {
#ifndef NDEBUG
  locked_ = false;
#endif
  owner_ = 0;
}

CondVar::CondVar(Mutex* mu)
    : mu_(mu) {
    PthreadCall("init cv", pthread_cond_init(&cv_, nullptr));
}

CondVar::~CondVar() { PthreadCall("destroy cv", pthread_cond_destroy(&cv_)); }

void CondVar::Wait(const char* msg) {
  mu_->BeforeUnlock();
  PthreadCall("cv wait", pthread_cond_wait(&cv_, &mu_->mu_));
  mu_->AfterLock();
}

bool CondVar::TimedWait(uint64_t abs_time_us, const char* msg) {
  struct timespec ts;
  ts.tv_sec = static_cast<time_t>(abs_time_us / 1000000);
  ts.tv_nsec = static_cast<suseconds_t>((abs_time_us % 1000000) * 1000);

  mu_->BeforeUnlock();
  int err = pthread_cond_timedwait(&cv_, &mu_->mu_, &ts);
  mu_->AfterLock();
  if (err == ETIMEDOUT) {
    return true;
  }
  if (err != 0) {
    PthreadCall("cv timedwait", err);
  }
  return false;
}

bool CondVar::IntervalWait(uint64_t timeout_interval, const char* msg) {
  timespec ts;
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  int64_t usec = tv.tv_usec + timeout_interval * 1000LL;
  ts.tv_sec = tv.tv_sec + usec / 1000000;
  ts.tv_nsec = (usec % 1000000) * 1000;
  mu_->BeforeUnlock();
  int ret = pthread_cond_timedwait(&cv_, &mu_->mu_, &ts);
  mu_->AfterLock(msg);
  return (ret == 0);
}

void CondVar::Signal() {
  PthreadCall("cv signal", pthread_cond_signal(&cv_));
}

void CondVar::SignalAll() {
  PthreadCall("cv broadcast", pthread_cond_broadcast(&cv_));
}

void CondVar::Broadcast() {
  PthreadCall("cv broadcast", pthread_cond_broadcast(&cv_));
}

RWMutex::RWMutex() {
  PthreadCall("init rwmutex", pthread_rwlock_init(&mu_, nullptr));
}

RWMutex::~RWMutex() { PthreadCall("destroy rwmutex", pthread_rwlock_destroy(&mu_)); }

void RWMutex::ReadLock() { PthreadCall("read lock", pthread_rwlock_rdlock(&mu_)); }

void RWMutex::WriteLock() { PthreadCall("write lock", pthread_rwlock_wrlock(&mu_)); }

void RWMutex::ReadUnlock() { PthreadCall("read unlock", pthread_rwlock_unlock(&mu_)); }

void RWMutex::WriteUnlock() { PthreadCall("write unlock", pthread_rwlock_unlock(&mu_)); }
  
} // namespace port 
  
} // namespace bubblefs