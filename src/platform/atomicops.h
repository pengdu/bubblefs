// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// For atomic operations on reference counts, see atomic_refcount.h.
// For atomic operations on sequence numbers, see atomic_sequence_num.h.

// The routines exported by this module are subtle.  If you use them, even if
// you get the code right, it will depend on careful reasoning about atomicity
// and memory ordering; it will be less readable, and harder to maintain.  If
// you plan to use these routines, you should have a good reason, such as solid
// evidence that performance would otherwise suffer, or there being no
// alternative.  You should assume only properties explicitly guaranteed by the
// specifications in this file.  You are almost certainly _not_ writing code
// just for the x86; if you assume x86 semantics, x86 hardware bugs and
// implementations on other archtectures will cause your code to break.  If you
// do not know what you are doing, avoid these routines, and use a Mutex.
//
// It is incorrect to make direct assignments to/from an atomic variable.
// You should use one of the Load or Store routines.  The NoBarrier
// versions are provided when no barriers are needed:
//   NoBarrier_Store()
//   NoBarrier_Load()
// Although there are currently no compiler enforcement, you are encouraged
// to use these.
//

// brpc/src/butil/atomicops.h

#ifndef BUBBLEFS_PLATFORM_ATOMICOPS_H_
#define BUBBLEFS_PLATFORM_ATOMICOPS_H_

#include <stdint.h>
#include <atomic>
#include "platform/macros.h"

namespace bubblefs {
  
namespace base {

#if !defined(__i386__) && !defined(__x86_64__)
#error    "Arch not supprot asm atomic!"
#endif  
  
// For atomic operations on reference counts, see atomic_refcount.h.
// For atomic operations on sequence numbers, see atomic_sequence_num.h.

// The routines exported by this module are subtle.  If you use them, even if
// you get the code right, it will depend on careful reasoning about atomicity
// and memory ordering; it will be less readable, and harder to maintain.  If
// you plan to use these routines, you should have a good reason, such as solid
// evidence that performance would otherwise suffer, or there being no
// alternative.  You should assume only properties explicitly guaranteed by the
// specifications in this file.  You are almost certainly _not_ writing code
// just for the x86; if you assume x86 semantics, x86 hardware bugs and
// implementations on other archtectures will cause your code to break.  If you
// do not know what you are doing, avoid these routines, and use a Mutex.
//
// It is incorrect to make direct assignments to/from an atomic variable.
// You should use one of the Load or Store routines.  The NoBarrier
// versions are provided when no barriers are needed:
//   NoBarrier_Store()
//   NoBarrier_Load()
// Although there are currently no compiler enforcement, you are encouraged
// to use these.
//

#if defined(OS_WIN) && defined(ARCH_CPU_64_BITS)
// windows.h #defines this (only on x64). This causes problems because the
// public API also uses MemoryBarrier at the public name for this fence. So, on
// X64, undef it, and call its documented
// (http://msdn.microsoft.com/en-us/library/windows/desktop/ms684208.aspx)
// implementation directly.
#undef MemoryBarrier
#endif

typedef int32_t Atomic32;
#ifdef ARCH_CPU_64_BITS
// We need to be able to go between Atomic64 and AtomicWord implicitly.  This
// means Atomic64 and AtomicWord should be the same type on 64-bit.
#if defined(__ILP32__) || defined(OS_NACL)
// NaCl's intptr_t is not actually 64-bits on 64-bit!
// http://code.google.com/p/nativeclient/issues/detail?id=1162
typedef int64_t Atomic64;
#else
typedef intptr_t Atomic64;
#endif
#endif

// Use AtomicWord for a machine-sized pointer.  It will use the Atomic32 or
// Atomic64 routines below, depending on your architecture.
typedef intptr_t AtomicWord;

// Atomically execute:
//      result = *ptr;
//      if (*ptr == old_value)
//        *ptr = new_value;
//      return result;
//
// I.e., replace "*ptr" with "new_value" if "*ptr" used to be "old_value".
// Always return the old value of "*ptr"
//
// This routine implies no memory barriers.
Atomic32 NoBarrier_CompareAndSwap(volatile Atomic32* ptr,
                                  Atomic32 old_value,
                                  Atomic32 new_value);

// Atomically store new_value into *ptr, returning the previous value held in
// *ptr.  This routine implies no memory barriers.
Atomic32 NoBarrier_AtomicExchange(volatile Atomic32* ptr, Atomic32 new_value);

// Atomically increment *ptr by "increment".  Returns the new value of
// *ptr with the increment applied.  This routine implies no memory barriers.
Atomic32 NoBarrier_AtomicIncrement(volatile Atomic32* ptr, Atomic32 increment);

Atomic32 Barrier_AtomicIncrement(volatile Atomic32* ptr,
                                 Atomic32 increment);

// These following lower-level operations are typically useful only to people
// implementing higher-level synchronization operations like spinlocks,
// mutexes, and condition-variables.  They combine CompareAndSwap(), a load, or
// a store with appropriate memory-ordering instructions.  "Acquire" operations
// ensure that no later memory access can be reordered ahead of the operation.
// "Release" operations ensure that no previous memory access can be reordered
// after the operation.  "Barrier" operations have both "Acquire" and "Release"
// semantics.   A MemoryBarrier() has "Barrier" semantics, but does no memory
// access.
Atomic32 Acquire_CompareAndSwap(volatile Atomic32* ptr,
                                Atomic32 old_value,
                                Atomic32 new_value);
Atomic32 Release_CompareAndSwap(volatile Atomic32* ptr,
                                Atomic32 old_value,
                                Atomic32 new_value);

void MemoryBarrier();
void NoBarrier_Store(volatile Atomic32* ptr, Atomic32 value);
void Acquire_Store(volatile Atomic32* ptr, Atomic32 value);
void Release_Store(volatile Atomic32* ptr, Atomic32 value);

Atomic32 NoBarrier_Load(volatile const Atomic32* ptr);
Atomic32 Acquire_Load(volatile const Atomic32* ptr);
Atomic32 Release_Load(volatile const Atomic32* ptr);

// 64-bit atomic operations (only available on 64-bit processors).
#ifdef ARCH_CPU_64_BITS
Atomic64 NoBarrier_CompareAndSwap(volatile Atomic64* ptr,
                                  Atomic64 old_value,
                                  Atomic64 new_value);
Atomic64 NoBarrier_AtomicExchange(volatile Atomic64* ptr, Atomic64 new_value);
Atomic64 NoBarrier_AtomicIncrement(volatile Atomic64* ptr, Atomic64 increment);
Atomic64 Barrier_AtomicIncrement(volatile Atomic64* ptr, Atomic64 increment);

Atomic64 Acquire_CompareAndSwap(volatile Atomic64* ptr,
                                Atomic64 old_value,
                                Atomic64 new_value);
Atomic64 Release_CompareAndSwap(volatile Atomic64* ptr,
                                Atomic64 old_value,
                                Atomic64 new_value);
void NoBarrier_Store(volatile Atomic64* ptr, Atomic64 value);
void Acquire_Store(volatile Atomic64* ptr, Atomic64 value);
void Release_Store(volatile Atomic64* ptr, Atomic64 value);
Atomic64 NoBarrier_Load(volatile const Atomic64* ptr);
Atomic64 Acquire_Load(volatile const Atomic64* ptr);
Atomic64 Release_Load(volatile const Atomic64* ptr);
#endif  // ARCH_CPU_64_BITS

} // namespace base

} // namespace bubblefs

// Include our platform specific implementation.
#if defined(COMPILER_GCC) && defined(ARCH_CPU_X86_FAMILY)
#include "platform/atomicops_internals_x86_gcc.h"
#else
#error "Atomic operations are not supported on your platform"
#endif

// ========= Provide base::atomic<T> =========
namespace bubblefs {
namespace base {
// static_atomic<> is a work-around for C++03 to declare global atomics
// w/o constructing-order issues. It can also used in C++11 though.
// Example:
//   base::static_atomic<int> g_counter = BASE_STATIC_ATOMIC_INIT(0);
// Notice that to make static_atomic work for C++03, it cannot be
// initialized by a constructor. Following code is wrong:
//   base::static_atomic<int> g_counter(0); // Not compile

#define BASE_STATIC_ATOMIC_INIT(val) { (val) }

template <typename T> struct static_atomic {
    T val;

    // NOTE: the memory_order parameters must be present.
    T load(std::memory_order o) { return ref().load(o); }
    void store(T v, std::memory_order o) { return ref().store(v, o); }
    T exchange(T v, std::memory_order o) { return ref().exchange(v, o); }
    bool compare_exchange_weak(T& e, T d, std::memory_order o)
    { return ref().compare_exchange_weak(e, d, o); }
    bool compare_exchange_weak(T& e, T d, std::memory_order so, std::memory_order fo)
    { return ref().compare_exchange_weak(e, d, so, fo); }
    bool compare_exchange_strong(T& e, T d, std::memory_order o)
    { return ref().compare_exchange_strong(e, d, o); }
    bool compare_exchange_strong(T& e, T d, std::memory_order so, std::memory_order fo)
    { return ref().compare_exchange_strong(e, d, so, fo); }
    T fetch_add(T v, std::memory_order o) { return ref().fetch_add(v, o); }
    T fetch_sub(T v, std::memory_order o) { return ref().fetch_sub(v, o); }
    T fetch_and(T v, std::memory_order o) { return ref().fetch_and(v, o); }
    T fetch_or(T v, std::memory_order o) { return ref().fetch_or(v, o); }
    T fetch_xor(T v, std::memory_order o) { return ref().fetch_xor(v, o); }
    static_atomic& operator=(T v) {
        store(v, std::memory_order_seq_cst);
        return *this;
    }
private:
    DISALLOW_ASSIGN(static_atomic);
    COMPILE_ASSERT(sizeof(T) == sizeof(std::atomic<T>), size_must_match);
    std::atomic<T>& ref() {
        // Suppress strict-alias warnings.
        std::atomic<T>* p = reinterpret_cast<std::atomic<T>*>(&val);
        return *p;
    }
};

} // namespace base
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_ATOMICOPS_H_