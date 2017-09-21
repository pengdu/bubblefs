// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// baidu/common/include/atomic.h
// brpc/src/butil/atomicops.h

#ifndef BUBBLEFS_PLATFORM_ATOMICOPS_H_
#define BUBBLEFS_PLATFORM_ATOMICOPS_H_

#include <stdint.h>
#include <atomic>
#include "platform/macros.h"

namespace bubblefs {
  
namespace bdcommon {
  
/**
 * Note: use gcc.
 * asm (statement : out : in : dirty clobbered regs or memory).
 * if asm conflicts, use __asm__ instead.
 * a,b,c,d,S,D means eax,ebx,ecx,edx,esi,edi.
 * r means any register, m means memory, i means imediate, g means any, %%reg refs register
 * $, $0x means constants.
 * %0 means output ... %n-1 means input operand(expression), "=" specifies output operand.
 * 
**/   
  
/**
 * @brief 原子加,返回原值
 *
 * @param [in/out] mem 原子变量
 * @param [in] add              : 加数
 * @return  inline int
 * @author yanshiguang02
 * @date 2012/09/09 13:55:38
**/
static inline int atomic_add(volatile int *mem, int add)
{
    asm volatile(
            "lock xadd %0, (%1);"
            : "=a"(add)
            : "r"(mem), "a"(add)
            : "memory"
    );
    return add;
}
static inline long atomic_add64(volatile long* mem, long add)
{
    asm volatile (
            "lock xaddq %0, (%1)"
            : "=a" (add)
            : "r" (mem), "a" (add)
            : "memory"
    );  
    return add;
}

/**
 * @brief 原子自增
 *
 * @param [in/out] mem   : volatile int*
 * @return  inline void
 * @author yanshiguang02
 * @date 2012/09/09 13:56:46
**/
static inline void atomic_inc(volatile int *mem)
{
    asm volatile(
            "lock incl %0;"
            : "=m"(*mem)
            : "m"(*mem)
    );
}
static inline void atomic_inc64(volatile long *mem)
{
    asm volatile(
            "lock incq %0;"
            : "=m"(*mem)
            : "m"(*mem)
    );
}

/**
 * @brief 原子自减
 *
 * @param [in/out] mem   : volatile int*
 * @return  inline void
 * @author yanshiguang02
 * @date 2012/09/09 13:57:54
**/
static inline void atomic_dec(volatile int *mem)
{
    asm volatile(
            "lock decl %0;"
            : "=m"(*mem)
            : "m"(*mem)
    );
}
static inline void atomic_dec64(volatile long *mem)
{
    asm volatile(
            "lock decq %0;"
            : "=m"(*mem)
            : "m"(*mem)
    );
}

/**
 * @brief swap
 *
 * @param [in/out] lockword   : volatile void*
 * @param [in/out] value   : int
 * @return  inline int
 * @author yanshiguang02
 * @date 2012/09/09 13:55:25
**/
static inline int atomic_swap(volatile void *lockword, int value)
{
    asm volatile(
            "lock xchg %0, (%1);"
            : "=a"(value)
            : "r"(lockword), "a"(value)
            : "memory"
    );
    return value;
}
static inline long atomic_swap64(volatile void *lockword, long value)
{
    asm volatile(
            "lock xchg %0, (%1);"
            : "=a"(value)
            : "r"(lockword), "a"(value)
            : "memory"
    );
    return value;
}


/**
 * @brief if set
    if(*mem == cmp)
        *mem = xchg;
    else
        cmp = *mem;
    return cmp;
 *
 * @param [in/out] mem   : volatile void*
 * @param [in/out] xchg   : int
 * @param [in/out] cmp   : int
 * @return  inline int
 * @author yanshiguang02
 * @date 2012/09/09 13:54:54
**/
static inline int atomic_comp_swap(volatile void *mem, int xchg, int cmp)
{
    asm volatile(
            "lock cmpxchg %1, (%2)"
            :"=a"(cmp)
            :"d"(xchg), "r"(mem), "a"(cmp)
    );
    return cmp;
}

/**
 * @brief 64位 if set
 *
 * @param [in/out] mem   : volatile void*
 * @param [in/out] xchg   : long long
 * @param [in/out] cmp   : long long
 * @return  inline int
 * @author yanshiguang02
 * @date 2012/09/09 13:54:15
**/
static inline long atomic_comp_swap64(volatile void *mem, long long xchg, long long cmp)
{
    asm volatile(
            "lock cmpxchg %1, (%2)"
            :"=a"(cmp)
            :"d"(xchg), "r"(mem), "a"(cmp)
    );
    return cmp;
}  
  
} // namespace bdcommon
  
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
    BASE_CASSERT(sizeof(T) == sizeof(std::atomic<T>), size_must_match);
    std::atomic<T>& ref() {
        // Suppress strict-alias warnings.
        std::atomic<T>* p = reinterpret_cast<std::atomic<T>*>(&val);
        return *p;
    }
};

} // namespace base
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_ATOMICOPS_H_