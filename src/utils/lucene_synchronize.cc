/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/src/core/util/Synchronize.cpp

#include "utils/lucene_synchronize.h"
#include <pthread.h>
#include <chrono>

namespace bubblefs {
namespace mylucene {

Synchronize::Synchronize() {
    lockThread = 0;
    recursionCount = 0;
}

Synchronize::~Synchronize() {
}

void Synchronize::createSync(SynchronizePtr& sync) {
    static std::mutex lockMutex;
    std::lock_guard<std::mutex> syncLock(lockMutex);
    if (!sync) {
        sync = newInstance<Synchronize>();
    }
}

void Synchronize::lock(int32_t timeout) {
    if (timeout > 0) {
        mutexSynchronize.try_lock_for(std::chrono::microseconds(timeout));
    } else {
        mutexSynchronize.lock();
    }
    lockThread = pthread_self();
    ++recursionCount;
}

void Synchronize::unlock() {
    if (--recursionCount == 0) {
        lockThread = 0;
    }
    mutexSynchronize.unlock();
}

int32_t Synchronize::unlockAll() {
    int32_t count = recursionCount;
    for (int32_t unlock = 0; unlock < count; ++unlock) {
        this->unlock();
    }
    return count;
}

bool Synchronize::holdsLock() {
    return (lockThread == pthread_self() && recursionCount > 0);
}

SyncLock::SyncLock(const SynchronizePtr& sync, int32_t timeout) {
    this->sync = sync;
    lock(timeout);
}

SyncLock::~SyncLock() {
    if (sync) {
        sync->unlock();
    }
}

void SyncLock::lock(int32_t timeout) {
    if (sync) {
        sync->lock(timeout);
    }
}

} // namespace mylucene
} // namespace bubblefs