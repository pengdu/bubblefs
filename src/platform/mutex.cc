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

static long get_micros() ALLOW_UNUSED; // use static and gcc attributes at declaration, inline defines in function implementation
long get_micros() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<long>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

static int32_t now_time_str(char* buf, int32_t len) ALLOW_UNUSED;
int32_t now_time_str(char* buf, int32_t len) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    const time_t seconds = tv.tv_sec;
    struct tm t;
    localtime_r(&seconds, &t);
    int32_t ret = 0;
    ret = snprintf(buf, len, "%02d/%02d %02d:%02d:%02d.%06d",
            t.tm_mon + 1,
            t.tm_mday,
            t.tm_hour,
            t.tm_min,
            t.tm_sec,
            static_cast<int>(tv.tv_usec));
    return ret;
}

static int PthreadCall(const char* label, int result) {
  if (result != 0) {
    fprintf(stderr, "pthreadcall %s: %s\n", label, strerror(result));
    abort();
  }
  return result;
}

void InitOnce(OnceType* once, void (*initializer)()) {
  PthreadCall("once", pthread_once(once, initializer));
}  

Mutex::Mutex() {
#ifdef MUTEX_DEBUG
  owner_ = 0;
  msg_ = 0;
  msg_threshold_ = 0;
  lock_time_ = 0;
#endif
  
  PthreadCall("init mutex default", pthread_mutex_init(&mu_, nullptr));
}

Mutex::Mutex(bool adaptive) {
#ifdef MUTEX_DEBUG
  owner_ = 0;
  msg_ = 0;
  msg_threshold_ = 0;
  lock_time_ = 0;
#endif
  
  if (!adaptive) {
    // prevent called by the same thread.
    pthread_mutexattr_t attr;
    PthreadCall("init mutexattr", pthread_mutexattr_init(&attr));
    PthreadCall("set mutexattr errorcheck", pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK));
    PthreadCall("init mutex errorcheck", pthread_mutex_init(&mu_, &attr));
    PthreadCall("destroy mutexattr errorcheck", pthread_mutexattr_destroy(&attr));
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
#ifdef MUTEX_DEBUG
  int64_t s = (msg) ? get_micros() : 0;
#endif
  
  PthreadCall("mutex lock", pthread_mutex_lock(&mu_));
  AfterLock(msg, msg_threshold);
  
#ifdef MUTEX_DEBUG
  if (msg && lock_time_ - s > msg_threshold) {
    char buf[32];
    now_time_str(buf, sizeof(buf));
    printf("%s [Mutex] %s wait lock %.3f ms\n", buf, msg, (lock_time_ -s) / 1000.0);
  }
#endif
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
  if (_millisecond < 0) {
    Lock();
    return true;
  }
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
#ifdef MUTEX_DEBUG
  if (0 == pthread_equal(owner_, pthread_self())) {
    fprintf(stderr, "mutex is held by two calling threads " PRIu64_FORMAT ":" PRIu64_FORMAT "\n",
            (uint64_t)owner_, (uint64_t)pthread_self());
    abort();
  }
#endif
}

void Mutex::AfterLock(const char* msg, int64_t msg_threshold) {
#ifdef MUTEX_DEBUG
  msg_ = msg;
  msg_threshold_ = msg_threshold;
  if (msg_) {
    lock_time_ = :get_micros();
  }
  (void)msg;
  (void)msg_threshold;
  owner_ = pthread_self();
#endif
}

void Mutex::BeforeUnlock(const char* msg) {
#ifdef MUTEX_DEBUG
  if (msg_ && :get_micros() - lock_time_ > msg_threshold_) {
    char buf[32];
    now_time_str(buf, sizeof(buf));
    printf("%s [Mutex] %s locked %.3f ms\n", 
           buf, msg_, (get_micros() - lock_time_) / 1000.0);
  }
  msg_ = NULL;
  owner_ = 0;
#endif
}

CondVar::CondVar(Mutex* mu)
    : mu_(mu) {
    PthreadCall("init cv", pthread_cond_init(&cv_, nullptr));
}

CondVar::~CondVar() { PthreadCall("destroy cv", pthread_cond_destroy(&cv_)); }

void CondVar::Wait() {
#ifdef MUTEX_DEBUG
  mu_->BeforeUnlock();
#endif
  
  PthreadCall("cv wait", pthread_cond_wait(&cv_, &mu_->mu_));
  
#ifdef MUTEX_DEBUG
  mu_->AfterLock();
#endif
}

bool CondVar::TimedWaitAbsolute(uint64_t abs_time_us) {
  struct timespec ts;
  ts.tv_sec = static_cast<time_t>(abs_time_us / 1000000);
  ts.tv_nsec = static_cast<suseconds_t>((abs_time_us % 1000000) * 1000);

#ifdef MUTEX_DEBUG
  mu_->BeforeUnlock();
#endif
  
  int err = pthread_cond_timedwait(&cv_, &mu_->mu_, &ts);
  
#ifdef MUTEX_DEBUG
  mu_->AfterLock();
#endif  

  if (err == ETIMEDOUT) {
    return true;
  }
  if (err != 0) {
    PthreadCall("cv timedwait", err);
  }
  return false;
}

// Returns true if the lock is acquired, false otherwise. abstime is the
// *absolute* time.
bool CondVar::TimedWaitAbsolute(const struct timespec& absolute_time) {
  int status = pthread_cond_timedwait(&cv_, &mu_->mu_, &absolute_time);
  if (status == ETIMEDOUT) {
    return false;
  }
  assert(status == 0);
  return true;
}

bool CondVar::TimedWait(uint64_t timeout) {
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
  
#ifdef MUTEX_DEBUG  
  mu_->BeforeUnlock();
#endif
  
  bool ret = pthread_cond_timedwait(&cv_, &mu_->mu_, &ts);
  
#ifdef MUTEX_DEBUG
  mu_->AfterLock();
#endif
  
  return (ret == 0);
}

// Calls timedwait with a relative, instead of an absolute, timeout.
bool CondVar::TimedwaitRelative(const struct timespec& relative_time) {
  struct timespec absolute;
  // clock_gettime would be more convenient, but that needs librt
  // int status = clock_gettime(CLOCK_REALTIME, &absolute);
  struct timeval tv;
  int status = gettimeofday(&tv, NULL);
  assert(status == 0);
  absolute.tv_sec = tv.tv_sec + relative_time.tv_sec;
  absolute.tv_nsec = tv.tv_usec * 1000 + relative_time.tv_nsec;

  return TimedWaitAbsolute(absolute);
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

/*!
    Attempts to lock for reading. If the lock was obtained, this
    function returns true, otherwise it returns false instead of
    waiting for the lock to become available, i.e. it does not block.
    The lock attempt will fail if another thread has locked for writing.
    If the lock was obtained, the lock must be unlocked with unlock()
    before another thread can successfully lock it.
    \sa unlock() lockForRead()
*/
bool RWMutex::TryReadLock() {
  int ret = pthread_rwlock_tryrdlock(&mu_);
  switch (ret) {
    case 0: return true;
    case EBUSY: return false;
    case EINVAL: abort();
    case EAGAIN: abort();
    case EDEADLK: abort();
    default: abort();
  }
  return false;
}

void RWMutex::WriteLock() { PthreadCall("write lock", pthread_rwlock_wrlock(&mu_)); }

/*!
    Attempts to lock for writing. If the lock was obtained, this
    function returns true; otherwise, it returns false immediately.
    The lock attempt will fail if another thread has locked for
    reading or writing.
    If the lock was obtained, the lock must be unlocked with unlock()
    before another thread can successfully lock it.
    \sa unlock() lockForWrite()
*/
bool RWMutex::TryWriteLock() {
  int ret = pthread_rwlock_trywrlock(&mu_);
  switch (ret) {
    case 0: return true;
    case EBUSY: return false;
    case EINVAL: abort();
    case EAGAIN: abort();
    case EDEADLK: abort();
    default: abort();
  }
  return false;
}

/*!
    Unlocks the lock.
    Attempting to unlock a lock that is not locked is an error, and will result
    in program termination.
    \sa lockForRead() lockForWrite() tryLockForRead() tryLockForWrite()
*/
void RWMutex::Unlock() {
  PthreadCall("unlock rwmutex", pthread_rwlock_unlock(&mu_));
}

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
  PthreadCall("init condlock", pthread_mutex_init(&mutex_, nullptr));
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