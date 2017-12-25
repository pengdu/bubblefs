/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/src/core/util/LuceneSignal.cpp

#include "utils/lucene_signal.h"
#include <chrono>
#include "utils/lucene_synchronize.h" 

namespace bubblefs {
namespace mylucene {  
   
Signal::Signal(const SynchronizePtr& objectLock) {
    this->objectLock = objectLock;
}

Signal::~Signal() {
}

void Signal::createSignal(SignalPtr& signal, const SynchronizePtr& objectLock) {
    static std::mutex lockMutex;
    std::lock_guard<std::mutex> syncLock(lockMutex);
    if (!signal) {
        signal = newInstance<Signal>(objectLock);
    }
}

void Signal::wait(int32_t timeout) {
    int32_t relockCount = objectLock ? objectLock->unlockAll() : 0;
    std::unique_lock<std::mutex> waitLock(waitMutex);
    while (signalCondition.wait_for(waitLock, std::chrono::milliseconds(timeout)) != std::cv_status::timeout) {
        if (timeout != 0) {
            break;
        }
        if (signalCondition.wait_for(waitLock, std::chrono::milliseconds(10)) != std::cv_status::timeout) {
            break;
        }
    }
    for (int32_t relock = 0; relock < relockCount; ++relock) {
        objectLock->lock();
    }
}

void Signal::notifyAll() {
    signalCondition.notify_all();
}

} // namespace mylucene
} // namespace bubblefs