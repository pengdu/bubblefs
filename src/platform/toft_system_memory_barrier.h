// Copyright (c) 2012, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/memory/barrier.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_MEMORY_BARRIER_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_MEMORY_BARRIER_H_

namespace bubblefs {
namespace mytoft {

inline void CompilerBarrier() { __asm__ __volatile__("": : :"memory"); }
#if defined __i386__ || defined __x86_64__
inline void MemoryBarrier() { __asm__ __volatile__("mfence": : :"memory"); }
inline void MemoryReadBarrier() { __asm__ __volatile__("lfence" ::: "memory"); }
inline void MemoryWriteBarrier() { __asm__ __volatile__("sfence" ::: "memory"); }
#else
#error Unsupportted platform.
#endif

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_MEMORY_BARRIER_H_