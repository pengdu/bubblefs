/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/src/core/util/LuceneSync.cpp

#include "utils/lucene_sync.h"
#include "utils/lucene_synchronize.h"
#include "utils/lucene_signal.h"

namespace bubblefs {
namespace mylucene {

ObjectSync::~ObjectSync() {
}

SynchronizePtr ObjectSync::getSync() {
    Synchronize::createSync(objectLock);
    return objectLock;
}

SignalPtr ObjectSync::getSignal() {
    Signal::createSignal(objectSignal, getSync());
    return objectSignal;
}

void ObjectSync::lock(int32_t timeout) {
    getSync()->lock();
}

void ObjectSync::unlock() {
    getSync()->unlock();
}

bool ObjectSync::holdsLock() {
    return getSync()->holdsLock();
}

void ObjectSync::wait(int32_t timeout) {
    getSignal()->wait(timeout);
}

void ObjectSync::notifyAll() {
    getSignal()->notifyAll();
}

} // namespace mylucene
} // namespace bubblefs