/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/include/LuceneSignal.h

#ifndef BUBBLEFS_UTILS_LUCENE_SIGNAL_H_
#define BUBBLEFS_UTILS_LUCENE_SIGNAL_H_

#include <condition_variable>
#include <mutex>
#include "utils/lucene.h"
#include "utils/lucene_types.h"

namespace bubblefs {
namespace mylucene {
  
/// Utility class to support signaling notifications.
class Signal {
public:
    Signal(const SynchronizePtr& objectLock = SynchronizePtr());
    virtual ~Signal();

protected:
    std::mutex waitMutex;
    std::condition_variable signalCondition;
    SynchronizePtr objectLock;

public:
    /// create a new LuceneSignal instance atomically.
    static void createSignal(SignalPtr& signal, const SynchronizePtr& objectLock);

    /// Wait for signal using an optional timeout.
    void wait(int32_t timeout = 0);

    /// Notify all threads waiting for signal.
    void notifyAll();
};

} // namespace mylucene
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_LUCENE_SIGNAL_H_