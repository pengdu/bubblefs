/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/include/LuceneSync.h

#ifndef BUBBLEFS_UTILS_LUCENE_SYNC_H_
#define BUBBLEFS_UTILS_LUCENE_SYNC_H_

#include "utils/lucene.h"

namespace bubblefs {
namespace mylucene {

/// Base class for all Lucene synchronised classes
class ObjectSync {
public:
    virtual ~ObjectSync();

protected:
    SynchronizePtr objectLock;
    SignalPtr objectSignal;

public:
    /// Return this object synchronize lock.
    virtual SynchronizePtr getSync();

    /// Return this object signal.
    virtual SignalPtr getSignal();

    /// Lock this object using an optional timeout.
    virtual void lock(int32_t timeout = 0);

    /// Unlock this object.
    virtual void unlock();

    /// Returns true if this object is currently locked by current thread.
    virtual bool holdsLock();

    /// Wait for signal using an optional timeout.
    virtual void wait(int32_t timeout = 0);

    /// Notify all threads waiting for signal.
    virtual void notifyAll();
};

} // namespace mylucene
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_LUCENE_SYNC_H_