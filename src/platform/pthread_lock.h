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

// common/include/mutex.h
// Paddle/paddle/utils/arch/linux/Locks.h

#ifndef BUBBLEFS_PLATFORM_PTHREAD_LOCK_H_
#define BUBBLEFS_PLATFORM_PTHREAD_LOCK_H_

#include <sys/time.h>
#include <semaphore.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include "platform/base.h"
#include "platform/macros.h"
#include "platform/timer.h"

namespace bubblefs {
namespace locks { 
  
static inline void PthreadCall(const char* label, int result) {
    if (result != 0) {
        fprintf(stderr, "pthread %s: %s\n", label, strerror(result));
        abort();
    }
}

//#define MUTEX_DEBUG 

// A Mutex represents an exclusive lock.
class Mutex {
public:
    Mutex()
        : owner_(0), msg_(nullptr), msg_threshold_(0), lock_time_(0), magic_(reinterpret_cast<uintptr_t>(this)) {
        pthread_mutexattr_t attr;
        PthreadCall("init mutexattr", pthread_mutexattr_init(&attr));
        PthreadCall("set mutexattr", pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK));
        PthreadCall("init mutex", pthread_mutex_init(&mu_, &attr));
        PthreadCall("destroy mutexattr", pthread_mutexattr_destroy(&attr));
    }
    ~Mutex() {
        PthreadCall("destroy mutex", pthread_mutex_destroy(&mu_));
    }
    // Lock the mutex.
    // Will deadlock if the mutex is already locked by this thread.
    void Lock(const char* msg = nullptr, int64_t msg_threshold = 5000) {
        #ifdef MUTEX_DEBUG
        int64_t s = (msg) ? timer::get_micros() : 0;
        #endif
        PthreadCall("mutex lock", pthread_mutex_lock(&mu_));
        AfterLock(msg, msg_threshold);
        #ifdef MUTEX_DEBUG
        if (msg && lock_time_ - s > msg_threshold) {
            char buf[32];
            timer::now_time_str(buf, sizeof(buf));
            printf("%s [Mutex] %s wait lock %.3f ms\n", buf, msg, (lock_time_ -s) / 1000.0);
        }
        #endif
    }
    void TryLock() {
      int ret = pthread_mutex_trylock(&mu_);
      if (EBUSY == ret) return false;
      if (EINVAL == ret) abort();
      else if (EAGAIN == ret) abort();
      else if (EDEADLK == ret) abort();
      else if (0 != ret) abort();
      return 0 == ret;
    }
    bool TimedLock(long _millisecond) {
      struct timespec ts;
      timer::make_timeout(&ts, _millisecond);
      int ret =  pthread_mutex_timedlock(&mu_, &ts);
      switch (ret) {
        case 0: return true;
        case ETIMEDOUT: return false;
        case EAGAIN: abort();
        case EDEADLK: abort();
        case EINVAL: abort();
        default: abort();
      }
      return false;
    }
    // Unlock the mutex.
    void Unlock() {
        BeforeUnlock();
        PthreadCall("mutex unlock", pthread_mutex_unlock(&mu_));
    }
    bool IsLocked() {
        int ret = pthread_mutex_trylock(&mu_);
        if (0 == ret) Unlock();
        return 0 != ret;
    }
    // Crash if this thread does not hold this mutex.
    void AssertHeld() {
        if (0 == pthread_equal(owner_, pthread_self())) {
            abort();
        }
    }
private:
    void AfterLock(const char* msg, int64_t msg_threshold) {
        #ifdef MUTEX_DEBUG
        msg_ = msg;
        msg_threshold_ = msg_threshold;
        if (msg_) {
            lock_time_ = timer::get_micros();
        }
        #endif
        (void)msg;
        (void)msg_threshold;
        owner_ = pthread_self();
    }
    void BeforeUnlock() {
        #ifdef MUTEX_DEBUG
        if (msg_ && timer::get_micros() - lock_time_ > msg_threshold_) {
            char buf[32];
            timer::now_time_str(buf, sizeof(buf));
            printf("%s [Mutex] %s locked %.3f ms\n", 
                   buf, msg_, (timer::get_micros() - lock_time_) / 1000.0);
        }
        msg_ = NULL;
        #endif
        owner_ = 0;
    }
private:
    friend class CondVar;
    Mutex(const Mutex&);
    void operator=(const Mutex&);
    pthread_mutex_t mu_;
    pthread_t owner_;
    const char* msg_;
    int64_t msg_threshold_;
    int64_t lock_time_;
    uintptr_t    magic_;  // Dangling pointer will dead lock, so check it!!!
};

// Mutex lock guard
class MutexLock {
public:
    explicit MutexLock(Mutex *mu, const char* msg = NULL, int64_t msg_threshold = 5000)
      : mu_(mu) {
        mu_->Lock(msg, msg_threshold);
    }
    ~MutexLock() {
        mu_->Unlock();
    }
private:
    Mutex *const mu_;
    MutexLock(const MutexLock&);
    void operator=(const MutexLock&);
};

// Conditional variable
class CondVar {
public:
    explicit CondVar(Mutex* mu) : mu_(mu) {
        PthreadCall("init condvar", pthread_cond_init(&cond_, nullptr));
    }
    ~CondVar() {
        PthreadCall("destroy condvar", pthread_cond_destroy(&cond_));
    }
    void Wait(const char* msg = nullptr) {
        int64_t msg_threshold = mu_->msg_threshold_;
        mu_->BeforeUnlock();
        PthreadCall("condvar wait", pthread_cond_wait(&cond_, &mu_->mu_));
        mu_->AfterLock(msg, msg_threshold);
    }
    // Timed wait in ms, return true iff signalled
    bool TimedLock(int timeout, const char* msg = nullptr) {
        if (timeout < 0) {
          Wait(msg);
          return 0;
        }
        timespec ts;
        struct timeval tv;
        gettimeofday(&tv, nullptr);
        int64_t usec = tv.tv_usec + timeout * 1000LL;
        ts.tv_sec = tv.tv_sec + usec / 1000000;
        ts.tv_nsec = (usec % 1000000) * 1000;
        int64_t msg_threshold = mu_->msg_threshold_;
        mu_->BeforeUnlock();
        int ret = pthread_cond_timedwait(&cond_, &mu_->mu_, &ts);
        mu_->AfterLock(msg, msg_threshold);
        return (ret == 0);
    }
    void Signal() {
        PthreadCall("signal", pthread_cond_signal(&cond_));
    }
    void Broadcast() {
        PthreadCall("broadcast", pthread_cond_broadcast(&cond_));
    }
private:
    CondVar(const CondVar&);
    void operator=(const CondVar&);
    Mutex* mu_;
    pthread_cond_t cond_;
};

/**
 * A simple read-write lock.
 * The RWlock allows a number of readers or at most one writer
 * at any point in time.
 * The RWlock disable copy.
 *
 * Lock:
 *
 * Use lock() to lock on write mode, no other thread can get it
 * until unlock.
 *
 * Use lock_shared() to lock on read mode, other thread can get
 * it by using the same method lock_shared().
 *
 * Unlock:
 *
 * Use unlock() to unlock the lock.
 */
class RWLock {
public:
  RWLock() { pthread_rwlock_init(&rwlock_, nullptr); }
  ~RWLock() { pthread_rwlock_destroy(&rwlock_); }
  RWLock(const RWLock&) = delete;
  RWLock& operator=(const RWLock&) = delete;

  /**
   * @brief lock on write mode.
   * @note the method will block the thread, if failed to get the lock.
   */
  // std::mutex interface
  void Lock() { pthread_rwlock_wrlock(&rwlock_); }
  void ReadLock() { pthread_rwlock_rdlock(&rwlock_); }
  void WriteLock() { pthread_rwlock_wrlock(&rwlock_); }
  /**
   * @brief lock on read mode.
   * @note if another thread is writing, it can't get the lock,
   * and will block the thread.
   */
  void LockShared() { pthread_rwlock_rdlock(&rwlock_); }
  void Unlock() { pthread_rwlock_unlock(&rwlock_); }

protected:
  pthread_rwlock_t rwlock_;
};

/**
 * The ReadLockGuard is a read mode RWLock
 * using RAII management mechanism.
 */
class ReadLockGuard {
public:
  /**
   * @brief Construct Function. Lock on rwlock in read mode.
   */
  explicit ReadLockGuard(RWLock* rwlock) : rwlock_(rwlock) {
    if (nullptr != rwlock) {
      rwlock_->LockShared();      
    }
  }

  /**
   * @brief Destruct Function.
   * @note This method just unlock the read mode rwlock,
   * won't destroy the lock.
   */
  ~ReadLockGuard() { 
    if (nullptr != rwlock_) {
      rwlock_->Unlock();      
    } 
  }

protected:
  RWLock* rwlock_;
};

/**
 * The WriteLockGuard is a write mode RWLock
 * using RAII management mechanism.
 */
class WriteLockGuard {
public:
  /**
   * @brief Construct Function. Lock on rwlock in read mode.
   */
  explicit WriteLockGuard(RWLock* rwlock) : rwlock_(rwlock) {
    if (nullptr != rwlock) {
      rwlock_->WriteLock();      
    }
  }

  /**
   * @brief Destruct Function.
   * @note This method just unlock the read mode rwlock,
   * won't destroy the lock.
   */
  ~WriteLockGuard() { 
    if (nullptr != rwlock_) {
      rwlock_->Unlock();      
    } 
  }

protected:
  RWLock* rwlock_;
};

#ifdef TF_USE_PTHREAD_SPINLOCK
/**
 * A simple wrapper for spin lock.
 * The lock() method of SpinLock is busy-waiting
 * which means it will keep trying to lock until lock on successfully.
 * The SpinLock disable copy.
 */
class SpinLock {
public:
  inline SpinLock() { pthread_spin_init(&lock_, PTHREAD_PROCESS_PRIVATE); }
  inline ~SpinLock() { pthread_spin_destroy(&lock_); }

  inline void Lock() { pthread_spin_lock(&lock_); }
  inline void Unlock() { pthread_spin_unlock(&lock_); }
  inline int TryLock() { return pthread_spin_trylock(&lock_); }

  pthread_spinlock_t lock_;
  char padding_[64 - sizeof(pthread_spinlock_t)];
};
#else
#include <stddef.h>
class SpinLock {
public:
  inline void Lock() {
    while (lock_.test_and_set(std::memory_order_acquire)) {
    }
  }
  inline void Unlock() { lock_.clear(std::memory_order_release); }
  inline int TryLock() { return lock_.test_and_set(std::memory_order_acquire); }

  std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
  char padding_[64 - sizeof(lock_)];  // Padding to cache line size
};
#endif // TF_USE_PTHREAD_SPINLOCK

/**
 * The SpinLockGuard is a SpinLock
 * using RAII management mechanism.
 */
class SpinLockGuard {
public:
  /**
   * @brief Construct Function. Lock on spin_lock.
   */
  explicit SpinLockGuard(SpinLock* spin_lock) : spin_lock_(spin_lock) {
    if (nullptr != spin_lock_) {
      spin_lock_->Lock();      
    }
  }

  /**
   * @brief Destruct Function.
   * @note This method just unlock the spin_lock,
   * won't destroy the lock.
   */
  ~SpinLockGuard() {
    if (nullptr != spin_lock_) {
      spin_lock_->Unlock();      
    }
  }

protected:
  SpinLock* spin_lock_;
};

/**
 * A simple wapper of semaphore which can only be shared in the same process.
 */
class Semaphore {
public:
  /**
   * @brief Construct Function.
   * @param[in] initValue the initial value of the
   * semaphore, default 0.
   */
  explicit Semaphore(int initValue = 0) {
    sem_init(&sem, 0, initValue);
  }

  ~Semaphore() {
    sem_destroy(&sem);
  }

  /**
   * @brief The same as wait(), except if the decrement can not
   * be performed until ts return false install of blocking.
   * @param[in] ts an absolute timeout in seconds and nanoseconds
   * since the Epoch 1970-01-01 00:00:00 +0000(UTC).
   * @return ture if the decrement proceeds before ts,
   * else return false.
   */
  bool TimedWait(struct timespec* ts) {
    return (0 == sem_timedwait(&sem, ts));
  }

  /**
   * @brief decrement the semaphore. If the semaphore's value is 0, then call
   * blocks.
   */
  void Wait() { sem_wait(&sem); }

  /**
   * @brief increment the semaphore. If the semaphore's value
   * greater than 0, wake up a thread blocked in wait().
   */
  void Post() { sem_post(&sem); }

private:
  sem_t sem;
};

/**
 * A simple wrapper of thread barrier.
 * The ThreadBarrier disable copy.
 */
class ThreadBarrier {
public:
  pthread_barrier_t barrier_;

  inline explicit ThreadBarrier(int count) {
    pthread_barrier_init(&barrier_, nullptr, count);
  }

  inline ~ThreadBarrier() { pthread_barrier_destroy(&barrier_); }

  inline void Wait() { pthread_barrier_wait(&barrier_); }
  
  TF_DISALLOW_COPY_AND_ASSIGN(ThreadBarrier);
};

/**
 * A wrapper for condition variable with mutex.
 */
class LockedCondition : public std::condition_variable {
public:
  /**
   * @brief execute op and notify one thread which was blocked.
   * @param[in] op a thread can do something in op before notify.
   */
  template <class Op>
  void NotifyOne(Op op) {
    std::lock_guard<std::mutex> guard(mutex_);
    op();
    std::condition_variable::notify_one();
  }

  /**
   * @brief execute op and notify all the threads which were blocked.
   * @param[in] op a thread can do something in op before notify.
   */
  template <class Op>
  void NotifyAll(Op op) {
    std::lock_guard<std::mutex> guard(mutex_);
    op();
    std::condition_variable::notify_all();
  }

  /**
   * @brief wait until pred return ture.
   * @tparam Predicate c++ concepts, describes a function object
   * that takes a single iterator argument
   * that is dereferenced and used to
   * return a value testable as a bool.
   * @note pred shall not apply any non-constant function
   * through the dereferenced iterator.
   */
  template <class Predicate>
  void Wait(Predicate pred) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::condition_variable::wait(lock, pred);
  }

  /**
   * @brief get mutex.
   */
  std::mutex* mutex() { return &mutex_; }

protected:
  std::mutex mutex_;
};

}  // namespace locks
}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_PTHREAD_LOCK_H_