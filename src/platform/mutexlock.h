//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// rocksdb/util/mutexlock.h

#ifndef BUBBLEFS_PLATFORM_MUTEXLOCK_H_
#define BUBBLEFS_PLATFORM_MUTEXLOCK_H_

#include "platform/base.h"
#include <assert.h>
#include <semaphore.h>
#include <stddef.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include "platform/port_posix.h"

namespace bubblefs {

// Helper class that locks a mutex on construction and unlocks the mutex when
// the destructor of the MutexLock object is invoked.
//
// Typical usage:
//
//   void MyClass::MyMethod() {
//     MutexLock l(&mu_);       // mu_ is an instance variable
//     ... some complex code, possibly with multiple return paths ...
//   }

class MutexLock {
 public:
  explicit MutexLock(port::Mutex *mu) : mu_(mu) {
    this->mu_->Lock();
  }
  ~MutexLock() { this->mu_->Unlock(); }

 private:
  port::Mutex *const mu_;
  // No copying allowed
  MutexLock(const MutexLock&);
  void operator=(const MutexLock&);
};

//
// Acquire a ReadLock on the specified RWMutex.
// The Lock will be automatically released then the
// object goes out of scope.
//
class ReadLock {
 public:
  explicit ReadLock(port::RWMutex *mu) : mu_(mu) {
    this->mu_->ReadLock();
  }
  ~ReadLock() { this->mu_->ReadUnlock(); }

 private:
  port::RWMutex *const mu_;
  // No copying allowed
  ReadLock(const ReadLock&);
  void operator=(const ReadLock&);
};

//
// Automatically unlock a locked mutex when the object is destroyed
//
class ReadUnlock {
 public:
  explicit ReadUnlock(port::RWMutex *mu) : mu_(mu) { mu->AssertHeld(); }
  ~ReadUnlock() { mu_->ReadUnlock(); }

 private:
  port::RWMutex *const mu_;
  // No copying allowed
  ReadUnlock(const ReadUnlock &) = delete;
  ReadUnlock &operator=(const ReadUnlock &) = delete;
};

//
// Acquire a WriteLock on the specified RWMutex.
// The Lock will be automatically released then the
// object goes out of scope.
//
class WriteLock {
 public:
  explicit WriteLock(port::RWMutex *mu) : mu_(mu) {
    this->mu_->WriteLock();
  }
  ~WriteLock() { this->mu_->WriteUnlock(); }

 private:
  port::RWMutex *const mu_;
  // No copying allowed
  WriteLock(const WriteLock&);
  void operator=(const WriteLock&);
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

//
// SpinMutex has very low overhead for low-contention cases.  Method names
// are chosen so you can use std::unique_lock or std::lock_guard with it.
//
class SpinMutex {
 public:
  SpinMutex() : locked_(false) {}

  bool try_lock() {
    auto currently_locked = locked_.load(std::memory_order_relaxed);
    return !currently_locked &&
           locked_.compare_exchange_weak(currently_locked, true,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed);
  }

  void lock() {
    for (size_t tries = 0;; ++tries) {
      if (try_lock()) {
        // success
        break;
      }
      port::AsmVolatilePause();
      if (tries > 100) {
        std::this_thread::yield();
      }
    }
  }

  void unlock() { locked_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool> locked_;
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

}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_MUTEXLOCK_H_