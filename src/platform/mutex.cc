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

#ifdef MUTEX_DEBUG
#include "platform/bdcom_timer.h" // use timer utils
#endif

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

void InitOnce(OnceType* once, void (*initializer)()) {
  PthreadCall("once", pthread_once(once, initializer));
}  

/*!
    \class QMutex
    \brief The QMutex class provides access serialization between threads.
    \threadsafe
    \ingroup thread
    The purpose of a QMutex is to protect an object, data structure or
    section of code so that only one thread can access it at a time
    (this is similar to the Java \c synchronized keyword). It is
    usually best to use a mutex with a QMutexLocker since this makes
    it easy to ensure that locking and unlocking are performed consistently.
    
    QMutex may implement like this:
    QMutex has members: QMutexData *d;
    QMutexData *d has members: QAtomicInt contenders; const uint recursive : 1; uint reserved : 31;
    QMutexPrivate : public QMutexData, has members: HANDLE owner; uint count; volatile bool wakeup; pthread_mutex_t mutex; pthread_cond_t cond;
    1. bool QMutexPrivate::wait(int timeout): 
         while (contenders.fetchAndStoreAcquire(2) > 0) { syscall(SYS_futex, &contenders._q_value, FUTEX_WAIT, 2, timeout);  xtimeout -= timer.nsecsElapsed(); }
    2. bool QMutexPrivate::wait(int timeout):
         contenders.fetchAndAddAcquire(1); pthread_mutex_lock(&mutex); 
         while (!wakeup) { pthread_cond_timedwait(&cond, &mutex, &ti); } 
         wakeup = false; pthread_mutex_unlock(&mutex); contenders.deref();
    3. void QMutexPrivate::wakeUp():
         contenders.fetchAndStoreRelease(0); syscall(SYS_futex, &contenders._q_value, FUTEX_WAKE, 1, timeout);
    4. void QMutexPrivate::wakeUp():
         pthread_mutex_lock(&mutex); wakeup = true; pthread_cond_signal(&cond); pthread_mutex_unlock(&mutex);
*/

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

/*!
    Locks the mutex. When you call lock() in a thread, other threads that try to call
    lock() in the same place will block until the thread that got the
    lock calls unlock(). A non-blocking alternative to lock() is tryLock().
    Calling this function multiple times on the same mutex from the
    same thread is allowed if this mutex is a recursive mutex. 
    If this mutex is a non-recursive mutex, this function will
    \e dead-lock when the mutex is locked recursively.
    \sa Unlock().
    
    implement code may like this:
    
    QMutexPrivate *d = static_cast<QMutexPrivate *>(this->d);
    Qt::HANDLE self;
    if (d->recursive) {
        self = QThread::currentThreadId();
        if (d->owner == self) {
            ++d->count;
            Q_ASSERT_X(d->count != 0, "QMutex::lock", "Overflow in recursion counter");
            return;
        }
        bool isLocked = d->contenders.testAndSetAcquire(0, 1);
        if (!isLocked) {
            // didn't get the lock, wait for it
            isLocked = d->wait();
            Q_ASSERT_X(isLocked, "QMutex::lock", "Internal error, infinite wait has timed out.");
        }
        d->owner = self;
        ++d->count;
        Q_ASSERT_X(d->count != 0, "QMutex::lock", "Overflow in recursion counter");
        return;
    }
    // non-recursive
    bool isLocked = d->contenders.testAndSetAcquire(0, 1);
    if (!isLocked) {
        // two choices:
        // 1. don't spin on single cpu machines, use: bool isLocked = d->wait();
        // 2. spinning more, use: do { do spin-time work; QThread::yieldCurrentThread(); } while (d->contenders != 0 || !d->contenders.testAndSetAcquire(0, 1));
        lockInternal();
    }
*/
void Mutex::Lock(const char* msg, int64_t msg_threshold) {
#ifdef MUTEX_DEBUG
  int64_t s = (msg) ? mybdcom::get_micros() : 0;
#endif
  
  PthreadCall("mutex lock", pthread_mutex_lock(&mu_));
  AfterLock(msg, msg_threshold);
  
#ifdef MUTEX_DEBUG
  if (msg && lock_time_ - s > msg_threshold) {
    char buf[32];
    mybdcom::now_time_str(buf, sizeof(buf));
    printf("%s [Mutex] %s wait lock %.3f ms\n", buf, msg, (lock_time_ -s) / 1000.0);
  }
#endif
}

/*!
    Attempts to lock the mutex. If the lock was obtained, this function
    returns true. If another thread has locked the mutex, this
    function returns false immediately.
    If the lock was obtained, the mutex must be unlocked with unlock()
    before another thread can successfully lock it.
    Calling this function multiple times on the same mutex from the
    same thread is allowed if this mutex is a recursive mutex.
    If this mutex is a non-recursive mutex, this function will 
    \e always return false when attempting to lock the mutex recursively.
    \sa lock(), unlock()
    
    implement code may like this:
    
    QMutexPrivate *d = static_cast<QMutexPrivate *>(this->d);
    Qt::HANDLE self;
    if (d->recursive) {
        self = QThread::currentThreadId();
        if (d->owner == self) {
            ++d->count;
            Q_ASSERT_X(d->count != 0, "QMutex::tryLock", "Overflow in recursion counter");
            return true;
        }
        bool isLocked = d->contenders.testAndSetAcquire(0, 1);
        if (!isLocked) {
            // some other thread has the mutex locked, or we tried to
            // recursively lock an non-recursive mutex
            return isLocked;
        }
        d->owner = self;
        ++d->count;
        Q_ASSERT_X(d->count != 0, "QMutex::tryLock", "Overflow in recursion counter");
        return isLocked;
    }
    // non-recursive
    return d->contenders.testAndSetAcquire(0, 1);
*/
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

/*!
    Attempts to lock the mutex. This function returns true if the lock
    was obtained; otherwise it returns false. If another thread has
    locked the mutex, this function will wait for at most \a timeout
    milliseconds for the mutex to become available.
    Note: Passing a negative number as the \a timeout is equivalent to
    calling lock(), i.e. this function will wait forever until mutex
    can be locked if \a timeout is negative.
    If the lock was obtained, the mutex must be unlocked with unlock()
    before another thread can successfully lock it.
    Calling this function multiple times on the same mutex from the
    same thread is allowed if this mutex is a recursive mutex. 
    If this mutex is a non-recursive mutex, this function will
    \e always return false when attempting to lock the mutex recursively.
    \sa lock(), unlock()
    
    implement code may like this:
    
    // non-recursive mutex
    ret = (d->contenders.testAndSetAcquire(0, 1)
          // didn't get the lock, wait for it
          || d->wait(timeout));
*/
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

/*!
    Unlocks the mutex. Attempting to unlock a mutex in a different
    thread to the one that locked it results in an error (segment fault). 
    Unlocking a mutex that is not locked results in undefined behavior.
    \sa lock()
    
    implement code may like this:
    
    QMutexPrivate *d = static_cast<QMutexPrivate *>(this->d);
    if (d->recursive) {
        if (!--d->count) {
            d->owner = 0;
            if (!d->contenders.testAndSetRelease(1, 0))
                d->wakeUp();
        }
    } else {
        if (!d->contenders.testAndSetRelease(1, 0))
            d->wakeUp();
    }
*/
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

/*
void Mutex::AssertHeld() {
  int r = pthread_mutex_trylock(&mu_);
  switch (r) {
    case EBUSY:
      // The mutex could not be acquired because it was already locked.
      return;  // OK
    case EDEADLK:
      // The current thread already owns the mutex.
      return;  // OK
    case EAGAIN:
      // The mutex could not be acquired because the maximum number of recursive
      // locks for mutex has been exceeded.
      break;
    case EPERM:
      // The current thread does not own the mutex.
      break;
    case 0:
      // Unexpectedly lock the mutex.
      r = EINVAL;
      break;
    default:
      // Other errors
      break;
  }

  // Abort the call
  PthreadCall("pthread_mutex_trylock", r);
}
*/

void Mutex::AfterLock(const char* msg, int64_t msg_threshold) {
#ifdef MUTEX_DEBUG
  msg_ = msg;
  msg_threshold_ = msg_threshold;
  if (msg_) {
    lock_time_ = mybdcom::get_micros();
  }
  (void)msg;
  (void)msg_threshold;
  owner_ = pthread_self();
#endif
}

void Mutex::BeforeUnlock(const char* msg) {
#ifdef MUTEX_DEBUG
  if (msg_ && mybdcom::get_micros() - lock_time_ > msg_threshold_) {
    char buf[32];
    mybdcom::now_time_str(buf, sizeof(buf));
    printf("%s [Mutex] %s locked %.3f ms\n", 
           buf, msg_, (mybdcom::get_micros() - lock_time_) / 1000.0);
  }
  msg_ = NULL;
  owner_ = 0;
#endif
}

/*!
    \class QMutexPool
    \brief The QMutexPool class provides a pool of QMutex objects.
    \internal
    \ingroup thread
    QMutexPool is a convenience class that provides access to a fixed
    number of QMutex objects.
    Typical use of a QMutexPool is in situations where it is not
    possible or feasible to use one QMutex for every protected object.
    The mutex pool will return a mutex based on the address of the
    object that needs protection.

// qt/src/corelib/thread/qmutexpool_p.h    
    
class Q_CORE_EXPORT QMutexPool
{
public:
    explicit QMutexPool(QMutex::RecursionMode recursionMode = QMutex::NonRecursive, int size = 131);
    ~QMutexPool();

    inline QMutex *get(const void *address) {
        int index = uint(quintptr(address)) % mutexes.count();
        QMutex *m = mutexes[index];
        if (m)
            return m;
        else
            return createMutex(index);
    }
    static QMutexPool *instance();
    static QMutex *globalInstanceGet(const void *address);

private:
    QMutex *createMutex(int index);
    QVarLengthArray<QAtomicPointer<QMutex>, 131> mutexes;
    QMutex::RecursionMode recursionMode;
};

extern Q_CORE_EXPORT QMutexPool *qt_global_mutexpool;

// qt/src/corelib/thread/qmutexpool.cpp

// qt_global_mutexpool is here for backwards compatibility only,
// use QMutexpool::instance() in new clode.
Q_CORE_EXPORT QMutexPool *qt_global_mutexpool = 0;
Q_GLOBAL_STATIC_WITH_ARGS(QMutexPool, globalMutexPool, (QMutex::Recursive))
    
QMutex *QMutexPool::createMutex(int index)
{
    // mutex not created, create one
    QMutex *newMutex = new QMutex(recursionMode);
    if (!mutexes[index].testAndSetOrdered(0, newMutex))
        delete newMutex;
    return mutexes[index];
}    
*/

/*!
    \class QWaitCondition
    \brief The QWaitCondition class provides a condition variable for
    synchronizing threads.
    \threadsafe
    \ingroup thread

    QWaitCondition allows a thread to tell other threads that some
    sort of condition has been met. One or many threads can block
    waiting for a QWaitCondition to set a condition with wakeOne() or
    wakeAll(). Use wakeOne() to wake one randomly selected condition or
    wakeAll() to wake them all.
    The mutex is necessary because the results of two threads
    attempting to change the value of the same variable
    simultaneously are unpredictable.
    Wait conditions are a powerful thread synchronization primitive.
    The \l{threads/waitconditions}{Wait Conditions} example shows how
    to use QWaitCondition as an alternative to QSemaphore for
    controlling access to a circular buffer shared by a producer
    thread and a consumer thread.
    
    the implement code may like this:
    QWaitCondition has members: QWaitConditionPrivate * d;
    QWaitConditionPrivate has members: 
      pthread_mutex_t mutex; pthread_cond_t cond; int waiters; int wakeups;
      bool wait(unsigned long time): forever { pthread_cond_timedwait(&cond, &mutex, &ti); if (wakeups == 0) {continue;} break; } --waiters; pthread_mutex_unlock(&mutex)
*/
CondVar::CondVar(Mutex* mu)
    : mu_(mu) {
    PthreadCall("init cv", pthread_cond_init(&cv_, nullptr));
}

CondVar::~CondVar() { PthreadCall("destroy cv", pthread_cond_destroy(&cv_)); }

/*!
    \fn bool QWaitCondition::wait(QMutex *mutex, unsigned long time)
    Releases the locked \a mutex and waits on the wait condition.  The
    \a mutex must be initially locked by the calling thread. If \a
    mutex is not in a locked state, this function returns
    immediately. If \a mutex is a recursive mutex, this function
    returns immediately. The \a mutex will be unlocked, and the
    calling thread will block until either of these conditions is met:
    \list
    \o Another thread signals it using wakeOne() or wakeAll(). This
       function will return true in this case.
    \o \a time milliseconds has elapsed. If \a time is \c ULONG_MAX
       (the default), then the wait will never timeout (the event
       must be signalled). This function will return false if the
       wait timed out.
    \endlist
    The mutex will be returned to the same locked state. This
    function is provided to allow the atomic transition from the
    locked state to the wait state.
    \sa wakeOne(), wakeAll()
    
    implement code may liek this:
    
    if (mutex->d->recursive) {
        qWarning("QWaitCondition: cannot wait on recursive mutexes");
        return false;
    }
    pthread_mutex_lock(&d->mutex) // or prelocked by outside
    ++d->waiters;
    mutex->unlock();
    bool returnValue = d->wait(time);
    mutex->lock();
*/
void CondVar::Wait(const char* msg) {
#ifdef MUTEX_DEBUG
  mu_->BeforeUnlock();
#endif
  
  PthreadCall("cv wait", pthread_cond_wait(&cv_, &mu_->mu_));
  
#ifdef MUTEX_DEBUG
  mu_->AfterLock();
#endif
}

bool CondVar::TimedWaitAbsolute(uint64_t abs_time_us, const char* msg) {
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
  
#ifdef MUTEX_DEBUG  
  mu_->BeforeUnlock();
#endif
  bool ret = pthread_cond_timedwait(&cv_, &mu_->mu_, &ts);
  
#ifdef MUTEX_DEBUG
  mu_->AfterLock(msg);
#endif
  
  return (ret == 0);
}

/*!
    \fn void QWaitCondition::wakeOne()
    Wakes one thread waiting on the wait condition. The thread that
    is woken up depends on the operating system's scheduling
    policies, and cannot be controlled or predicted.
    If you want to wake up a specific thread, the solution is
    typically to use different wait conditions and have different
    threads wait on different conditions.
    \sa wakeAll()
    
    implement code may like this:
    
    pthread_mutex_lock(&d->mutex);
    d->wakeups = qMin(d->wakeups + 1, d->waiters);
    pthread_cond_signal(&d->cond);
    pthread_mutex_unlock(&d->mutex);
*/
void CondVar::Signal() {
  PthreadCall("cv signal", pthread_cond_signal(&cv_));
}

/*!
    \fn void QWaitCondition::wakeAll()
    Wakes all threads waiting on the wait condition. The order in
    which the threads are woken up depends on the operating system's
    scheduling policies and cannot be controlled or predicted.
    \sa wakeOne()
    
    implement code may like this:
    
    pthread_mutex_lock(&d->mutex); // protect d->wakeups
    d->wakeups = d->waiters;
    pthread_cond_broadcast(&d->cond);
    pthread_mutex_unlock(&d->mutex);
*/
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

/*!
    \class QReadWriteLock
    \brief The QReadWriteLock class provides read-write locking.
    \threadsafe
    \ingroup thread
    A read-write lock is a synchronization tool for protecting
    resources that can be accessed for reading and writing. This type
    of lock is useful if you want to allow multiple threads to have
    simultaneous read-only access, but as soon as one thread wants to
    write to the resource, all other threads must be blocked until
    the writing is complete.
    In many cases, QReadWriteLock is a direct competitor to QMutex.
    QReadWriteLock is a good choice if there are many concurrent
    reads and writing occurs infrequently.
    
    QReadWriteLock mey implement like this:
    QReadWriteLock has members: QReadWriteLockPrivate *d;
    QReadWriteLockPrivate has members: 
      QMutex mutex; QWaitCondition readerWait; QWaitCondition writerWait;
      int accessCount; int waitingReaders; int waitingWriters; bool recursive;
      Qt::HANDLE currentWriter; QHash<Qt::HANDLE, int> currentReaders;
*/
RWMutex::RWMutex() {
  PthreadCall("init rwmutex", pthread_rwlock_init(&mu_, nullptr));
}

RWMutex::~RWMutex() { PthreadCall("destroy rwmutex", pthread_rwlock_destroy(&mu_)); }

/*!
    Locks the lock for reading. This function will block the current
    thread if any thread (including the current) has locked for writing.
    \sa unlock() lockForWrite() tryLockForRead()
    
    implement code like this:
    
    QMutexLocker lock(&d->mutex);
    Qt::HANDLE self = 0;
    if (d->recursive) {
        self = QThread::currentThreadId();
        QHash<Qt::HANDLE, int>::iterator it = d->currentReaders.find(self);
        if (it != d->currentReaders.end()) {
            ++it.value();
            ++d->accessCount;
            Q_ASSERT_X(d->accessCount > 0, "QReadWriteLock::lockForRead()", "Overflow in lock counter");
            return;
        }
    }
    while (d->accessCount < 0 || d->waitingWriters) { // no-op this for tryLockForRead()
        ++d->waitingReaders;
        d->readerWait.wait(&d->mutex, timeout < 0 ? ULONG_MAX : ulong(timeout));
        --d->waitingReaders;
    }
    if (d->recursive)
        d->currentReaders.insert(self, 1);
    ++d->accessCount;
*/
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

/*!
    Locks the lock for writing. This function will block the current
    thread if another thread has locked for reading or writing.
    \sa unlock() lockForRead() tryLockForWrite()
    
    implement code like this:
    
    QMutexLocker lock(&d->mutex);
    Qt::HANDLE self = 0;
    if (d->recursive) {
        self = QThread::currentThreadId();
        if (d->currentWriter == self) {
            --d->accessCount;
            Q_ASSERT_X(d->accessCount < 0, "QReadWriteLock::lockForWrite()", "Overflow in lock counter");
            return;
        }
    }
    while (d->accessCount != 0) { // no-ops for tryLockForWrite()
        ++d->waitingWriters;
        d->writerWait.wait(&d->mutex, timeout < 0 ? ULONG_MAX : ulong(timeout));
        --d->waitingWriters;
    }
    if (d->recursive)
        d->currentWriter = self;
    --d->accessCount;
    Q_ASSERT_X(d->accessCount < 0, "QReadWriteLock::lockForWrite()", "Overflow in lock counter");
*/
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