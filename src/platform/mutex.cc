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
// slash/slash/src/slash_mutex.cc

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

// Return false if timeout
static bool PthreadTimeoutCall(const char* label, int result) {
  if (result != 0) {
    if (result == ETIMEDOUT) {
      return false;
    }
    fprintf(stderr, "pthreadtimeoutcall %s: %s\n", label, strerror(result));
    abort();
  }
  return true;
}

void InitOnce(OnceType* once, void (*initializer)()) {
  PthreadCall("once", pthread_once(once, initializer));
}  

Mutex::Mutex() : owner_(0) {
  // PthreadCall("init mutex", pthread_mutex_init(&mu_, NULL));
  // prevent called by the same thread.
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
    fprintf(stderr, "mutex is held by two calling threads " PRIu64_FORMAT ":" PRIu64_FORMAT "\n",
            (uint64_t)owner_, (uint64_t)pthread_self());
    abort();
  }
}

void Mutex::AfterLock(const char* msg, int64_t msg_threshold) {
#ifndef NDEBUG
  locked_ = true;
  //printf("AfterLock %p\n", &mu_);
#endif
  owner_ = pthread_self();
}

void Mutex::BeforeUnlock(const char* msg) {
#ifndef NDEBUG
  locked_ = false;
  //printf("BeforeUnlock %p\n", &mu_);
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

bool CondVar::TimedWaitAbsolute(uint64_t abs_time_us, const char* msg) {
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

bool CondVar::TimedWait(uint64_t timeout, const char* msg) {
  /*
   * pthread_cond_timedwait api use absolute API
   * so we need gettimeofday + timeout
   */
  struct timespec ts;
  struct timeval now;
  gettimeofday(&now, nullptr);
  int64_t usec = now.tv_usec + timeout * 1000LL;
  ts.tv_sec = now.tv_sec + usec / 1000000;
  ts.tv_nsec = (usec % 1000000) * 1000;
  mu_->BeforeUnlock();
  bool ret = PthreadTimeoutCall("timewait",
      pthread_cond_timedwait(&cv_, &mu_->mu_, &ts));
  mu_->AfterLock(msg);
  return ret;
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

ConditionVariable::ConditionVariable() {
    pthread_condattr_t cond_attr;
    pthread_condattr_init(&cond_attr);
    pthread_cond_init(&m_hCondition, &cond_attr);
    pthread_condattr_destroy(&cond_attr);
}

ConditionVariable::~ConditionVariable() {
    pthread_cond_destroy(&m_hCondition);
}

void ConditionVariable::CheckValid() const
{
    assert(m_hCondition.__data.__total_seq != -1ULL && "this cond has been destructed");
}

void ConditionVariable::Signal() {
    CheckValid();
    pthread_cond_signal(&m_hCondition);
}

void ConditionVariable::Broadcast() {
    CheckValid();
    pthread_cond_broadcast(&m_hCondition);
}

void ConditionVariable::Wait(Mutex* mutex) {
    CheckValid();
    pthread_cond_wait(&m_hCondition, mutex->GetMutex());
}

int ConditionVariable::TimedWait(Mutex* mutex, int timeout_in_ms) {
    // -1 wait forever
    if (timeout_in_ms < 0) {
        Wait(mutex);
        return 0;
    }

    timespec ts;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int64_t usec = tv.tv_usec + timeout_in_ms * 1000LL;
    ts.tv_sec = tv.tv_sec + usec / 1000000;
    ts.tv_nsec = (usec % 1000000) * 1000;

    return pthread_cond_timedwait(&m_hCondition, mutex->GetMutex(), &ts);
}

RWMutex::RWMutex() {
  PthreadCall("init rwmutex", pthread_rwlock_init(&mu_, nullptr));
}

RWMutex::~RWMutex() { PthreadCall("destroy rwmutex", pthread_rwlock_destroy(&mu_)); }

void RWMutex::ReadLock() { PthreadCall("read lock", pthread_rwlock_rdlock(&mu_)); }

void RWMutex::WriteLock() { PthreadCall("write lock", pthread_rwlock_wrlock(&mu_)); }

void RWMutex::ReadUnlock() { PthreadCall("read unlock", pthread_rwlock_unlock(&mu_)); }

void RWMutex::WriteUnlock() { PthreadCall("write unlock", pthread_rwlock_unlock(&mu_)); }
  
RefMutex::RefMutex() {
  refs_ = 0;
  PthreadCall("init refmutex", pthread_mutex_init(&mu_, nullptr));
}

RefMutex::~RefMutex() {
  PthreadCall("destroy refmutex", pthread_mutex_destroy(&mu_));
}

void RefMutex::Ref() {
  refs_++;
}
void RefMutex::Unref() {
  --refs_;
  if (refs_ == 0) {
    delete this;
  }
}

void RefMutex::Lock() {
  PthreadCall("lock refmutex", pthread_mutex_lock(&mu_));
}

void RefMutex::Unlock() {
  PthreadCall("unlock refmutex", pthread_mutex_unlock(&mu_));
}

RecordMutex::~RecordMutex() {
  mutex_.Lock();
  
  std::unordered_map<std::string, RefMutex *>::const_iterator it = records_.begin();
  for (; it != records_.end(); it++) {
    delete it->second;
  }
  mutex_.Unlock();
}


void RecordMutex::Lock(const std::string &key) {
  mutex_.Lock();
  std::unordered_map<std::string, RefMutex *>::const_iterator it = records_.find(key);

  if (it != records_.end()) {
    RefMutex *ref_mutex = it->second;
    ref_mutex->Ref();
    mutex_.Unlock();

    ref_mutex->Lock();
  } else {
    RefMutex *ref_mutex = new RefMutex();

    records_.insert(std::make_pair(key, ref_mutex));
    ref_mutex->Ref();
    mutex_.Unlock();

    ref_mutex->Lock();
  }
}

void RecordMutex::Unlock(const std::string &key) {
  mutex_.Lock();
  std::unordered_map<std::string, RefMutex *>::const_iterator it = records_.find(key);
  
  if (it != records_.end()) {
    RefMutex *ref_mutex = it->second;

    if (ref_mutex->IsLastRef()) {
      records_.erase(it);
    }
    ref_mutex->Unlock();
    ref_mutex->Unref();
  }

  mutex_.Unlock();
}

CondLock::CondLock() {
  PthreadCall("init condlock", pthread_mutex_init(&mutex_, NULL));
}

CondLock::~CondLock() {
  PthreadCall("destroy condlock", pthread_mutex_unlock(&mutex_));
}

void CondLock::Lock() {
  PthreadCall("lock condlock", pthread_mutex_lock(&mutex_));
}

void CondLock::Unlock() {
  PthreadCall("unlock condlock", pthread_mutex_unlock(&mutex_));
}

void CondLock::Wait() {
  PthreadCall("condlock wait", pthread_cond_wait(&cond_, &mutex_));
}

void CondLock::TimedWait(uint64_t timeout) {
  /*
   * pthread_cond_timedwait api use absolute API
   * so we need gettimeofday + timeout
   */
  struct timeval now;
  gettimeofday(&now, NULL);
  struct timespec tsp;

  int64_t usec = now.tv_usec + timeout * 1000LL;
  tsp.tv_sec = now.tv_sec + usec / 1000000;
  tsp.tv_nsec = (usec % 1000000) * 1000;

  pthread_cond_timedwait(&cond_, &mutex_, &tsp);
}

void CondLock::Signal() {
  PthreadCall("condlock signal", pthread_cond_signal(&cond_));
}

void CondLock::Broadcast() {
  PthreadCall("condlock broadcast", pthread_cond_broadcast(&cond_));
}
  
} // namespace port 
  
} // namespace bubblefs