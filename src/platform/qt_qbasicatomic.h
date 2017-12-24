/****************************************************************************
**
** Copyright (C) 2015 The Qt Company Ltd.
** Contact: http://www.qt.io/licensing/
**
** This file is part of the QtCore module of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see http://www.qt.io/terms-conditions. For further
** information use the contact form at http://www.qt.io/contact-us.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 or version 3 as published by the Free
** Software Foundation and appearing in the file LICENSE.LGPLv21 and
** LICENSE.LGPLv3 included in the packaging of this file. Please review the
** following information to ensure the GNU Lesser General Public License
** requirements will be met: https://www.gnu.org/licenses/lgpl.html and
** http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** As a special exception, The Qt Company gives you certain additional
** rights. These rights are described in The Qt Company LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** GNU General Public License Usage
** Alternatively, this file may be used under the terms of the GNU
** General Public License version 3.0 as published by the Free Software
** Foundation and appearing in the file LICENSE.GPL included in the
** packaging of this file.  Please review the following information to
** ensure the GNU General Public License version 3.0 requirements will be
** met: http://www.gnu.org/copyleft/gpl.html.
**
** $QT_END_LICENSE$
**
****************************************************************************/

// qt/src/corelib/thread/qbasicatomic.h

#ifndef BUBBLEFS_PLATFORM_QT_QBASICATOMIC_H_
#define BUBBLEFS_PLATFORM_QT_QBASICATOMIC_H_

namespace bubblefs {
namespace myqt {
  
class QBasicAtomicInt
{
public:
#ifdef QT_ARCH_PARISC
    int _q_lock[4];
#endif
#if defined(QT_ARCH_WINDOWS) || defined(QT_ARCH_WINDOWSCE)
    union { // needed for Q_BASIC_ATOMIC_INITIALIZER
        volatile long _q_value;
    };
#else
    volatile int _q_value;
#endif

    // Non-atomic API
    inline bool operator==(int value) const
    {
        return _q_value == value;
    }

    inline bool operator!=(int value) const
    {
        return _q_value != value;
    }

    inline bool operator!() const
    {
        return _q_value == 0;
    }

    inline operator int() const
    {
        return _q_value;
    }

    inline QBasicAtomicInt &operator=(int value)
    {
#ifdef QT_ARCH_PARISC
        this->_q_lock[0] = this->_q_lock[1] = this->_q_lock[2] = this->_q_lock[3] = -1;
#endif
        _q_value = value;
        return *this;
    }

    // Atomic API, implemented in qatomic_XXX.h

    static bool isReferenceCountingNative();
    static bool isReferenceCountingWaitFree();

    bool ref();
    bool deref();

    static bool isTestAndSetNative();
    static bool isTestAndSetWaitFree();

    bool testAndSetRelaxed(int expectedValue, int newValue);
    bool testAndSetAcquire(int expectedValue, int newValue);
    bool testAndSetRelease(int expectedValue, int newValue);
    bool testAndSetOrdered(int expectedValue, int newValue);

    static bool isFetchAndStoreNative();
    static bool isFetchAndStoreWaitFree();

    int fetchAndStoreRelaxed(int newValue);
    int fetchAndStoreAcquire(int newValue);
    int fetchAndStoreRelease(int newValue);
    int fetchAndStoreOrdered(int newValue);

    static bool isFetchAndAddNative();
    static bool isFetchAndAddWaitFree();

    int fetchAndAddRelaxed(int valueToAdd);
    int fetchAndAddAcquire(int valueToAdd);
    int fetchAndAddRelease(int valueToAdd);
    int fetchAndAddOrdered(int valueToAdd);
};

template <typename T>
class QBasicAtomicPointer
{
public:
#ifdef QT_ARCH_PARISC
    int _q_lock[4];
#endif
#if defined(QT_ARCH_WINDOWS) || defined(QT_ARCH_WINDOWSCE)
    union {
        T * volatile _q_value;
#  if !defined(Q_OS_WINCE) && !defined(__i386__) && !defined(_M_IX86)
        qint64
#  else
        long
#  endif
        volatile _q_value_integral;
    };
#else
    T * volatile _q_value;
#endif

    // Non-atomic API
    inline bool operator==(T *value) const
    {
        return _q_value == value;
    }

    inline bool operator!=(T *value) const
    {
        return !operator==(value);
    }

    inline bool operator!() const
    {
        return operator==(0);
    }

    inline operator T *() const
    {
        return _q_value;
    }

    inline T *operator->() const
    {
        return _q_value;
    }

    inline QBasicAtomicPointer<T> &operator=(T *value)
    {
#ifdef QT_ARCH_PARISC
        this->_q_lock[0] = this->_q_lock[1] = this->_q_lock[2] = this->_q_lock[3] = -1;
#endif
        _q_value = value;
        return *this;
    }

    // Atomic API, implemented in qatomic_XXX.h

    static bool isTestAndSetNative();
    static bool isTestAndSetWaitFree();

    bool testAndSetRelaxed(T *expectedValue, T *newValue);
    bool testAndSetAcquire(T *expectedValue, T *newValue);
    bool testAndSetRelease(T *expectedValue, T *newValue);
    bool testAndSetOrdered(T *expectedValue, T *newValue);

    static bool isFetchAndStoreNative();
    static bool isFetchAndStoreWaitFree();

    T *fetchAndStoreRelaxed(T *newValue);
    T *fetchAndStoreAcquire(T *newValue);
    T *fetchAndStoreRelease(T *newValue);
    T *fetchAndStoreOrdered(T *newValue);

    static bool isFetchAndAddNative();
    static bool isFetchAndAddWaitFree();

    T *fetchAndAddRelaxed(qptrdiff valueToAdd);
    T *fetchAndAddAcquire(qptrdiff valueToAdd);
    T *fetchAndAddRelease(qptrdiff valueToAdd);
    T *fetchAndAddOrdered(qptrdiff valueToAdd);
};

#ifdef QT_ARCH_PARISC
#  define Q_BASIC_ATOMIC_INITIALIZER(a) {{-1,-1,-1,-1},(a)}
#elif defined(QT_ARCH_WINDOWS) || defined(QT_ARCH_WINDOWSCE)
#  define Q_BASIC_ATOMIC_INITIALIZER(a) { {(a)} }
#else
#  define Q_BASIC_ATOMIC_INITIALIZER(a) { (a) }
#endif

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

} // namespace myqt
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_QT_QBASICATOMIC_H_