/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/src/core/util/LuceneThread.cpp

#include "utils/lucene_thread.h"
#include "utils/lucene_synchronize.h"

namespace bubblefs {
namespace mylucene { 
  
#if defined(_WIN32) || defined(_WIN64)
const int32_t Thread::MAX_THREAD_PRIORITY = THREAD_PRIORITY_HIGHEST;
const int32_t Thread::NORM_THREAD_PRIORITY = THREAD_PRIORITY_NORMAL;
const int32_t Thread::MIN_THREAD_PRIORITY = THREAD_PRIORITY_LOWEST;
#else
const int32_t Thread::MAX_THREAD_PRIORITY = 2;
const int32_t Thread::NORM_THREAD_PRIORITY = 0;
const int32_t Thread::MIN_THREAD_PRIORITY = -2;
#endif

Thread::Thread() {
    running = false;
}

Thread::~Thread() {
}

void Thread::start() {
    setRunning(false);
    thread = newInstance<boost::thread>(Thread::runThread, this);
    setRunning(true);
}

void Thread::runThread(Thread* thread) {
    ThreadPtr threadObject(thread->shared_from_this());
    try {
        threadObject->run();
    } catch (...) {
    }
    threadObject->setRunning(false);
    threadObject.reset();
}

void Thread::setRunning(bool running) {
    SyncLock syncLock(this);
    this->running = running;
}

bool Thread::isRunning() {
    SyncLock syncLock(this);
    return running;
}

bool Thread::isAlive() {
    return (thread && isRunning());
}

void Thread::setPriority(int32_t priority) {
#if defined(_WIN32) || defined(_WIN64)
    if (thread) {
        SetThreadPriority(thread->native_handle(), priority);
    }
#endif
}

int32_t Thread::getPriority() {
#if defined(_WIN32) || defined(_WIN64)
    return thread ? GetThreadPriority(thread->native_handle()) : NORM_THREAD_PRIORITY;
#else
    return NORM_THREAD_PRIORITY;
#endif
}

void Thread::yield() {
    if (thread) {
        thread->yield();
    }
}

bool Thread::join(int32_t timeout) {
    while (isAlive() && !thread->timed_join(boost::posix_time::milliseconds(timeout))) {
        if (timeout != 0) {
            return false;
        }
        if (thread->timed_join(boost::posix_time::milliseconds(10))) {
            return true;
        }
    }
    return true;
}

int64_t Thread::currentId() {
#if defined(_WIN32) || defined(_WIN64)
    return (int64_t)GetCurrentThreadId();
#else
    return (int64_t)pthread_self();
#endif
}

void Thread::threadSleep(int32_t time) {
    boost::this_thread::sleep(boost::posix_time::milliseconds(time));
}

void Thread::threadYield() {
    boost::this_thread::yield();
}

} // namespace mylucene
} // namespace bubblefs