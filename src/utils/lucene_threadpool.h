/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/include/ThreadPool.h

#ifndef BUBBLEFS_UTILS_LUCENE_THREADPOOL_H_
#define BUBBLEFS_UTILS_LUCENE_THREADPOOL_H_

#include "utils/lucene_object.h"
#include "utils/lucene_synchronize.h"

#include "boost/asio.hpp"
#include "boost/any.hpp"
#include "boost/thread/thread.hpp"

namespace bubblefs {
namespace mylucene {

typedef std::shared_ptr<boost::asio::io_service::work> workPtr;

/// A Future represents the result of an asynchronous computation. Methods are provided to check if the computation
/// is complete, to wait for its completion, and to retrieve the result of the computation. The result can only be
/// retrieved using method get when the computation has completed, blocking if necessary until it is ready.
class Future : public Object {
public:
    virtual ~Future();

protected:
    boost::any value;

public:
    void set(const boost::any& value) {
        SyncLock syncLock(this);
        this->value = value;
    }

    template <typename TYPE>
    TYPE get() {
        SyncLock syncLock(this);
        while (value.empty()) {
            wait(10);
        }
        return value.empty() ? TYPE() : boost::any_cast<TYPE>(value);
    }
};

/// Utility class to handle a pool of threads.
class ThreadPool : public Object {
public:
    ThreadPool();
    virtual ~ThreadPool();

    LUCENE_CLASS(ThreadPool)

protected:
    boost::asio::io_service io_service;
    workPtr work;
    boost::thread_group threadGroup;

    static const int32_t THREADPOOL_SIZE;

public:
    /// Get singleton thread pool instance.
    static ThreadPoolPtr getInstance();

    template <typename FUNC>
    FuturePtr scheduleTask(FUNC func) {
        FuturePtr future(newInstance<Future>());
        io_service.post(boost::bind(&ThreadPool::execute<FUNC>, this, func, future));
        return future;
    }

protected:
    // this will be executed when one of the threads is available
    template <typename FUNC>
    void execute(FUNC func, const FuturePtr& future) {
        future->set(func());
        future->notifyAll();
    }
};

} // namespace mylucene
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_LUCENE_THREADPOOL_H_