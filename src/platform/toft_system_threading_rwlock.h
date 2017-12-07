// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/threading/rwlock.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_RWLOCK_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_RWLOCK_H_

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <string.h>

#include "platform/toft_system_check_error.h"
#include "platform/toft_system_threading_scoped_locker.h"
#include "utils/toft_base_uncopyable.h"

namespace bubblefs {
namespace mytoft {

// Reader/Writer lock
class RwLock
{
    DECLARE_UNCOPYABLE(RwLock);
public:
    typedef ScopedReaderLocker<RwLock> ReaderLocker;
    typedef ScopedWriterLocker<RwLock> WriterLocker;
    typedef ScopedTryReaderLocker<RwLock> TryReaderLocker;
    typedef ScopedTryWriterLocker<RwLock> TryWriterLocker;

    enum Kind {
        kKindPreferReader = PTHREAD_RWLOCK_PREFER_READER_NP,

        // Setting the value read-write lock kind to PTHREAD_RWLOCK_PREFER_WRITER_NP,
        // results in the same behavior as setting the value to PTHREAD_RWLOCK_PREFER_READER_NP.
        // As long as a reader thread holds the lock the thread holding a write lock will be
        // starved. Setting the kind value to PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP,
        // allows the writer to run. However, the writer may not be recursive as is implied by the
        // name.
        kKindPreferWriter = PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP,
        kKindDefault = PTHREAD_RWLOCK_DEFAULT_NP,
    };

public:
    RwLock();
    explicit RwLock(Kind kind);
    ~RwLock();

    void ReaderLock()
    {
        CheckValid();
        MYTOFT_CHECK_PTHREAD_ERROR(pthread_rwlock_rdlock(&m_lock));
    }

    bool TryReaderLock()
    {
        CheckValid();
        return MYTOFT_CHECK_PTHREAD_TRYLOCK_ERROR(pthread_rwlock_tryrdlock(&m_lock));
    }

    void WriterLock()
    {
        CheckValid();
        MYTOFT_CHECK_PTHREAD_ERROR(pthread_rwlock_wrlock(&m_lock));
    }

    bool TryWriterLock()
    {
        CheckValid();
        return MYTOFT_CHECK_PTHREAD_TRYLOCK_ERROR(pthread_rwlock_trywrlock(&m_lock));
    }

    void Unlock()
    {
        CheckValid();
        MYTOFT_CHECK_PTHREAD_ERROR(pthread_rwlock_unlock(&m_lock));
    }

    // Names for scoped lockers
    void ReaderUnlock()
    {
        Unlock();
    }

    void WriterUnlock()
    {
        Unlock();
    }

public: // Only for test and debug, should not be used in normal code logical
    void AssertReaderLocked() const
    {
        // Accessing pthread private data: nonportable but no other way
        assert(m_lock.__data.__nr_readers != 0);
    }

    void AssertWriterLocked() const
    {
        assert(m_lock.__data.__writer != 0);
    }

    void AssertLocked() const
    {
        assert(m_lock.__data.__nr_readers != 0 || m_lock.__data.__writer != 0);
    }

private:
    void CheckValid() const
    {
#ifdef __GLIBC__
        // If your program crashed here at runtime, maybe the rwlock object
        // has been destructed.
        if (memcmp(&m_lock.__data.__flags,  "\xFF\xFF\xFF\xFF", 4) == 0)
            MYTOFT_CHECK_ERRNO_ERROR(EINVAL);
#endif
    }

private:
    pthread_rwlock_t m_lock;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_THREADING_RWLOCK_H_