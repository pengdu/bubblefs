// Copyright (c) 2009, The Toft Authors.
// All rights reserved.
// Author: CHEN Feng <chen3feng@gmail.com>

// // toft/container/skiplist.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_ATOMIC_POINTER_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_ATOMIC_POINTER_H_

#include "platform/toft_system_memory_barrier.h"

namespace bubblefs {
namespace mytoft {

class AtomicPointer {
private:
    void* rep_;

public:
    AtomicPointer() : rep_(NULL) {
    }
    explicit AtomicPointer(void* p) : rep_(p) {
    }

    inline void* NoBarrier_Load() const {
        return rep_;
    }
    inline void NoBarrier_Store(void* v) {
        rep_ = v;
    }
    inline void* Acquire_Load() const {
        void* result = rep_;
        MemoryBarrier();
        return result;
    }
    inline void Release_Store(void* v) {
        MemoryBarrier();
        rep_ = v;
    }
};
  
} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_ATOMIC_POINTER_H_