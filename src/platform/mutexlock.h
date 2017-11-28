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

#include <assert.h>
#include <pthread.h>
#include <stddef.h>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include "platform/macros.h"
#include "platform/mutex.h"

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
  explicit MutexLock(port::Mutex *mu, const char* msg = nullptr, int64_t msg_threshold = 5000) : mu_(mu) {
    this->mu_->Lock();
  }
  ~MutexLock() { this->mu_->Unlock(); }

 private:
  port::Mutex *const mu_;
  DISALLOW_COPY_AND_ASSIGN(MutexLock);
};

/*
  // qt/src/corelib/thread/qorderedmutexlocker_p.h
  Locks 2 mutexes in a defined order, avoiding a recursive lock if
  we're trying to lock the same mutex twice.
  
class QOrderedMutexLocker
{
public:
    QOrderedMutexLocker(QMutex *m1, QMutex *m2)
        : mtx1((m1 == m2) ? m1 : (m1 < m2 ? m1 : m2)),
          mtx2((m1 == m2) ?  0 : (m1 < m2 ? m2 : m1)),
          locked(false)
    {
        relock();
    }
    ~QOrderedMutexLocker()
    {
        unlock();
    }
    void relock()
    {
        if (!locked) {
            if (mtx1) mtx1->lockInline();
            if (mtx2) mtx2->lockInline();
            locked = true;
        }
    }
    void unlock()
    {
        if (locked) {
            if (mtx1) mtx1->unlockInline();
            if (mtx2) mtx2->unlockInline();
            locked = false;
        }
    }
private:
    QMutex *mtx1, *mtx2;
    bool locked;
};
*/

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
  DISALLOW_COPY_AND_ASSIGN(ReadLock);
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
  DISALLOW_COPY_AND_ASSIGN(ReadUnlock);
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
  DISALLOW_COPY_AND_ASSIGN(WriteLock);
};

#ifdef TF_USE_PTHREAD_SPINLOCK
/**
 * A simple wrapper for spin lock.
 * The lock() method of SpinLock is busy-waiting
 * which means it will keep trying to lock until lock on successfully.
 * The SpinLock disable copy.
 */
class SpinLock final {
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
class SpinLock final {
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
      asm_volatile_pause();
      if (tries > 100) {
        std::this_thread::yield();
      }
    }
  }

  void unlock() { locked_.store(false, std::memory_order_release); }
  
private:
  void asm_volatile_pause() {
#if defined(__i386__) || defined(__x86_64__)
  asm volatile("pause");
#endif
  // it's okay for other platforms to be no-ops
  }

 private:
  std::atomic<bool> locked_;
};

class AutoSpinLock
{
public:
    explicit AutoSpinLock(SpinLock* spin_lock)
        :   _spin_lock(spin_lock)
    {
        if (NULL != _spin_lock)
        {
            _spin_lock->Lock();
        }
    }
    ~AutoSpinLock()
    {
        if (NULL != _spin_lock)
        {
            _spin_lock->Unlock();
        }
    }
private:
    SpinLock*   _spin_lock;
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
  
  DISALLOW_COPY_AND_ASSIGN(ThreadBarrier);
};

}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_MUTEXLOCK_H_