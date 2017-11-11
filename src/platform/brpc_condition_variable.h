// Copyright (c) 2011 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/synchronization/lock.h
// brpc/src/butil/synchronization/condition_variable.h

#ifndef BUBBLEFS_PLATFORM_BRPC_CONDITION_VARIABLE_H_
#define BUBBLEFS_PLATFORM_BRPC_CONDITION_VARIABLE_H_

#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include "platform/base_export.h"
#include "platform/macros.h"

#if defined(OS_WIN)
#include <windows.h>
#endif

namespace bubblefs {
namespace mybrpc {

// A convenient wrapper for an OS specific critical section.  
class BASE_EXPORT Mutex {
    DISALLOW_COPY_AND_ASSIGN(Mutex);
public:
#if defined(OS_WIN)
  typedef CRITICAL_SECTION NativeHandle;
#elif defined(OS_POSIX)
  typedef pthread_mutex_t NativeHandle;
#endif

public:
    Mutex() {
#if defined(OS_WIN)
    // The second parameter is the spin count, for short-held locks it avoid the
    // contending thread from going to sleep which helps performance greatly.
        ::InitializeCriticalSectionAndSpinCount(&_native_handle, 2000);
#elif defined(OS_POSIX)
        pthread_mutex_init(&_native_handle, NULL);
#endif
    }
    
    ~Mutex() {
#if defined(OS_WIN)
        ::DeleteCriticalSection(&_native_handle);
#elif defined(OS_POSIX)
        pthread_mutex_destroy(&_native_handle);
#endif
    }

    // Locks the mutex. If another thread has already locked the mutex, a call
    // to lock will block execution until the lock is acquired.
    void lock() {
#if defined(OS_WIN)
        ::EnterCriticalSection(&_native_handle);
#elif defined(OS_POSIX)
        pthread_mutex_lock(&_native_handle);
#endif
    }

    // Unlocks the mutex. The mutex must be locked by the current thread of
    // execution, otherwise, the behavior is undefined.
    void unlock() {
#if defined(OS_WIN)
        ::LeaveCriticalSection(&_native_handle);
#elif defined(OS_POSIX)
        pthread_mutex_unlock(&_native_handle);
#endif
    }
    
    // Tries to lock the mutex. Returns immediately.
    // On successful lock acquisition returns true, otherwise returns false.
    bool try_lock() {
#if defined(OS_WIN)
        return (::TryEnterCriticalSection(&_native_handle) != FALSE);
#elif defined(OS_POSIX)
        return pthread_mutex_trylock(&_native_handle) == 0;
#endif
    }

    // Returns the underlying implementation-defined native handle object.
    NativeHandle* native_handle() { return &_native_handle; }

private:
#if defined(OS_POSIX)
    // The posix implementation of ConditionVariable needs to be able
    // to see our lock and tweak our debugging counters, as it releases
    // and acquires locks inside of pthread_cond_{timed,}wait.
friend class ConditionVariable;
#elif defined(OS_WIN)
// The Windows Vista implementation of ConditionVariable needs the
// native handle of the critical section.
friend class WinVistaCondVar;
#endif

    NativeHandle _native_handle;
};

// TODO: Remove this type.
class BASE_EXPORT Lock : public Mutex {
    DISALLOW_COPY_AND_ASSIGN(Lock);
public:
    Lock() {}
    ~Lock() {}
    void Acquire() { lock(); }
    void Release() { unlock(); }
    bool Try() { return try_lock(); }
    void AssertAcquired() const {}
};

// A helper class that acquires the given Lock while the AutoLock is in scope.
class AutoLock {
public:
    struct AlreadyAcquired {};

    explicit AutoLock(Lock& lock) : lock_(lock) {
        lock_.Acquire();
    }

    AutoLock(Lock& lock, const AlreadyAcquired&) : lock_(lock) {
        lock_.AssertAcquired();
    }

    ~AutoLock() {
        lock_.AssertAcquired();
        lock_.Release();
    }

private:
    Lock& lock_;
    DISALLOW_COPY_AND_ASSIGN(AutoLock);
};

// AutoUnlock is a helper that will Release() the |lock| argument in the
// constructor, and re-Acquire() it in the destructor.
class AutoUnlock {
public:
    explicit AutoUnlock(Lock& lock) : lock_(lock) {
        // We require our caller to have the lock.
        lock_.AssertAcquired();
        lock_.Release();
    }

    ~AutoUnlock() {
        lock_.Acquire();
    }

private:
    Lock& lock_;
    DISALLOW_COPY_AND_ASSIGN(AutoUnlock);
};

// ConditionVariable wraps pthreads condition variable synchronization or, on
// Windows, simulates it.  This functionality is very helpful for having
// several threads wait for an event, as is common with a thread pool managed
// by a master.  The meaning of such an event in the (worker) thread pool
// scenario is that additional tasks are now available for processing.  It is
// used in Chrome in the DNS prefetching system to notify worker threads that
// a queue now has items (tasks) which need to be tended to.  A related use
// would have a pool manager waiting on a ConditionVariable, waiting for a
// thread in the pool to announce (signal) that there is now more room in a
// (bounded size) communications queue for the manager to deposit tasks, or,
// as a second example, that the queue of tasks is completely empty and all
// workers are waiting.
//
// USAGE NOTE 1: spurious signal events are possible with this and
// most implementations of condition variables.  As a result, be
// *sure* to retest your condition before proceeding.  The following
// is a good example of doing this correctly:
//
// while (!work_to_be_done()) Wait(...);
//
// In contrast do NOT do the following:
//
// if (!work_to_be_done()) Wait(...);  // Don't do this.
//
// Especially avoid the above if you are relying on some other thread only
// issuing a signal up *if* there is work-to-do.  There can/will
// be spurious signals.  Recheck state on waiting thread before
// assuming the signal was intentional. Caveat caller ;-).
//
// USAGE NOTE 2: Broadcast() frees up all waiting threads at once,
// which leads to contention for the locks they all held when they
// called Wait().  This results in POOR performance.  A much better
// approach to getting a lot of threads out of Wait() is to have each
// thread (upon exiting Wait()) call Signal() to free up another
// Wait'ing thread.  Look at condition_variable_unittest.cc for
// both examples.
//
// Broadcast() can be used nicely during teardown, as it gets the job
// done, and leaves no sleeping threads... and performance is less
// critical at that point.
//
// The semantics of Broadcast() are carefully crafted so that *all*
// threads that were waiting when the request was made will indeed
// get signaled.  Some implementations mess up, and don't signal them
// all, while others allow the wait to be effectively turned off (for
// a while while waiting threads come around).  This implementation
// appears correct, as it will not "lose" any signals, and will guarantee
// that all threads get signaled by Broadcast().
//
// This implementation offers support for "performance" in its selection of
// which thread to revive.  Performance, in direct contrast with "fairness,"
// assures that the thread that most recently began to Wait() is selected by
// Signal to revive.  Fairness would (if publicly supported) assure that the
// thread that has Wait()ed the longest is selected. The default policy
// may improve performance, as the selected thread may have a greater chance of
// having some of its stack data in various CPU caches.
//
// For a discussion of the many very subtle implementation details, see the FAQ
// at the end of condition_variable_win.cc.

class ConditionVarImpl;

class BASE_EXPORT ConditionVariable {
 public:
  // Construct a cv for use with ONLY one user lock.
  explicit ConditionVariable(Mutex* user_lock);

  ~ConditionVariable();

  // Wait() releases the caller's critical section atomically as it starts to
  // sleep, and the reacquires it when it is signaled.
  void Wait();
  void TimedWait(const int64_t max_time);
  // Broadcast() revives all waiting threads.
  void Broadcast();
  // Signal() revives one waiting thread.
  void Signal();

 private:

#if defined(OS_WIN)
  ConditionVarImpl* impl_;
#elif defined(OS_POSIX)
  pthread_cond_t condition_;
  pthread_mutex_t* user_mutex_;
#endif

  DISALLOW_COPY_AND_ASSIGN(ConditionVariable);
};

}  // namespace mybrpc
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_BRPC_CONDITION_VARIABLE_H_