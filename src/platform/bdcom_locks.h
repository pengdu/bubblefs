// Copyright (c) 2014 Baidu.com, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// sofa-pbrpc/src/sofa/pbrpc/mutex_lock.h
// sofa-pbrpc/src/sofa/pbrpc/condition_variable.h
// sofa-pbrpc/src/sofa/pbrpc/spin_lock.h
// sofa-pbrpc/src/sofa/pbrpc/rw_lock.h
// sofa-pbrpc/src/sofa/pbrpc/scoped_locker.h

#ifndef BUBBLEFS_PLATFORM_BDCOM_LOCKS_H_
#define BUBBLEFS_PLATFORM_BDCOM_LOCKS_H_

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>

namespace bubblefs {
namespace mybdcom {

class ConditionVariable;

class MutexLock
{
public:
    MutexLock()
    {
        pthread_mutex_init(&_lock, NULL);
    }
    ~MutexLock()
    {
        pthread_mutex_destroy(&_lock);
    }
    void lock()
    {
        pthread_mutex_lock(&_lock);
    }
    void unlock()
    {
        pthread_mutex_unlock(&_lock);
    }
private:
    friend class ConditionVariable;
    pthread_mutex_t _lock;
};  

class ConditionVariable
{
public:
    ConditionVariable()
    {
        pthread_cond_init(&_cond, NULL);
    }
    ~ConditionVariable()
    {
        pthread_cond_destroy(&_cond);
    }
    void wait(MutexLock& mutex)
    {
        assert(0 == pthread_cond_wait(&_cond, &mutex._lock));
    }
    bool wait(MutexLock& mutex, int64 timeout_in_ms)
    {
        if (timeout_in_ms < 0)
        {
            wait(mutex);
            return true;
        }
        timespec ts;
        calculate_expiration(timeout_in_ms, &ts);
        int error = pthread_cond_timedwait(&_cond, &mutex._lock, &ts);
        if (error == 0)
        {
            return true;
        }
        else if (error == ETIMEDOUT)
        {
            return false;
        }
        else
        {
            assert(false);
            return false;
        }
    }
    void signal()
    {
        assert(0 == pthread_cond_signal(&_cond));
    }
    void broadcast()
    {
        assert(0 == pthread_cond_broadcast(&_cond));
    }
private:
    void calculate_expiration(int64 timeout_in_ms, timespec* ts)
    {
        timeval tv;
        gettimeofday(&tv, NULL);
        int64 usec = tv.tv_usec + timeout_in_ms * 1000LL;
        ts->tv_sec = tv.tv_sec + usec / 1000000;
        ts->tv_nsec = (usec % 1000000) * 1000;
    }
private:
    pthread_cond_t _cond;
};
  
class SpinLock
{
public:
    SpinLock() { pthread_spin_init(&_lock, 0); }
    ~SpinLock() { pthread_spin_destroy(&_lock); }
    void lock() { pthread_spin_lock(&_lock); }
    bool try_lock() { return pthread_spin_trylock(&_lock) == 0; }
    void unlock() { pthread_spin_unlock(&_lock); }

private:
    pthread_spinlock_t _lock;
}; // class SpinLock


class RWLock
{
public:
    RWLock()
    {
        pthread_rwlock_init(&_lock, NULL);
    }
    ~RWLock()
    {
        pthread_rwlock_destroy(&_lock);
    }
    void lock()
    {
        pthread_rwlock_wrlock(&_lock);
    }
    void lock_shared()
    {
        pthread_rwlock_rdlock(&_lock);
    }
    void unlock()
    {
        pthread_rwlock_unlock(&_lock);
    }
private:
    pthread_rwlock_t _lock;
};

class ReadLocker
{
public:
    explicit ReadLocker(RWLock* lock) : _lock(lock)
    {
        _lock->lock_shared();
    }
    ~ReadLocker()
    {
        _lock->unlock();
    }
private:
    RWLock* _lock;
};
class WriteLocker
{
public:
    explicit WriteLocker(RWLock* lock) : _lock(lock)
    {
        _lock->lock();
    }
    ~WriteLocker()
    {
        _lock->unlock();
    }
private:
    RWLock* _lock;
};

template <typename LockType>
class ScopedLocker
{
public:
    explicit ScopedLocker(LockType& lock)
        : _lock(&lock)
    {
        _lock->lock();
    }

    explicit ScopedLocker(LockType* lock)
        : _lock(lock)
    {
        _lock->lock();
    }

    ~ScopedLocker()
    {
        _lock->unlock();
    }

private:
    LockType* _lock;
}; // class ScopedLocker

} // namespace mybdcom
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_BDCOM_LOCKS_H_