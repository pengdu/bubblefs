// Modifications copyright (C) 2017, Baidu.com, Inc.
// Copyright 2017 The Apache Software Foundation

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// baidu/common/include/atomic.h
// baidu/ins/src/common/asm_atomic.h
// palo/be/src/common/atomic.h

#ifndef BUBBLEFS_PLATFORM_ATOMIC_H_
#define BUBBLEFS_PLATFORM_ATOMIC_H_

#include <stdint.h>
#include <algorithm>
#include "platform/macros.h"

#if !defined(__i386__) && !defined(__x86_64__)
#error    "Arch not supprot asm atomic!"
#endif

namespace bubblefs {
namespace atomics {

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
 * @brief atomic add
 * lock xadd guarantees atomic ops and memory fence for muliti processors;
 * xadd exchanges the first operand (destination operand) with the second operand (source operand), 
 * then loads the sum of the two values into the destination operand. 
 * The destination operand can be a register or a memory location; 
 * the source operand is a register.
 * @param [in/out] mem  atomic operand
 * @param [in] add              : add operand
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
 * @brief atomic increment
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
 * @brief atomic decrement
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
 * @brief 64bit if set
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

/*
template <typename T>
inline void asm_atomic_inc(volatile T* n)
{
    asm volatile ("lock; incl %0;":"+m"(*n)::"cc");
}
template <typename T>
inline void asm_atomic_dec(volatile T* n)
{
    asm volatile ("lock; decl %0;":"+m"(*n)::"cc");
}
template <typename T>
inline T asm_atomic_add_ret_old(volatile T* n, T v)
{
    asm volatile ("lock; xaddl %1, %0;":"+m"(*n),"+r"(v)::"cc");
    return v;
}
template <typename T>
inline T asm_atomic_inc_ret_old(volatile T* n)
{
    T r = 1;
    asm volatile ("lock; xaddl %1, %0;":"+m"(*n), "+r"(r)::"cc");
    return r;
}
template <typename T>
inline T asm_atomic_dec_ret_old(volatile T* n)
{
    T r = (T)-1;
    asm volatile ("lock; xaddl %1, %0;":"+m"(*n), "+r"(r)::"cc");
    return r;
}
template <typename T>
inline T asm_atomic_add_ret_old64(volatile T* n, T v)
{
    asm volatile ("lock; xaddq %1, %0;":"+m"(*n),"+r"(v)::"cc");
    return v;
}
template <typename T>
inline T asm_atomic_inc_ret_old64(volatile T* n)
{
    T r = 1;
    asm volatile ("lock; xaddq %1, %0;":"+m"(*n), "+r"(r)::"cc");
    return r;
}
template <typename T>
inline T asm_atomic_dec_ret_old64(volatile T* n)
{
    T r = (T)-1;
    asm volatile ("lock; xaddq %1, %0;":"+m"(*n), "+r"(r)::"cc");
    return r;
}
template <typename T>
inline void asm_atomic_add(volatile T* n, T v)
{
    asm volatile ("lock; addl %1, %0;":"+m"(*n):"r"(v):"cc");
}
template <typename T>
inline void asm_atomic_sub(volatile T* n, T v)
{
    asm volatile ("lock; subl %1, %0;":"+m"(*n):"r"(v):"cc");
}
template <typename T, typename C, typename D>
inline T asm_atomic_cmpxchg(volatile T* n, C cmp, D dest)
{
    asm volatile ("lock; cmpxchgl %1, %0":"+m"(*n), "+r"(dest), "+a"(cmp)::"cc");
    return cmp;
}
// return old value
template <typename T>
inline T asm_atomic_swap(volatile T* lockword, T value)
{
    asm volatile ("lock; xchg %0, %1;" : "+r"(value), "+m"(*lockword));
    return value;
}
template <typename T, typename E, typename C>
inline T asm_atomic_comp_swap(volatile T* lockword, E exchange, C comperand)
{
    return asm_atomic_cmpxchg(lockword, comperand, exchange);
}
*/

class AtomicUtil {
public:
    // Issues instruction to have the CPU wait, this is less busy (bus traffic
    // etc) than just spinning.
    // For example:
    //  while (1);
    // should be:
    //  while (1) CpuWait();
    static TF_ATTRIBUTE_ALWAYS_INLINE void cpu_wait() {
        asm volatile("pause\n": : :"memory");
    }

    /// Provides "barrier" semantics (see below) without a memory access.
    static TF_ATTRIBUTE_ALWAYS_INLINE void memory_barrier() {
        TF_SYNC_SYNCHRONIZE
    }

    /// Provides a compiler barrier. The compiler is not allowed to reorder memory
    /// accesses across this (but the CPU can).  This generates no instructions.
    static TF_ATTRIBUTE_ALWAYS_INLINE void compiler_barrier() {
        __asm__ __volatile__("" : : : "memory");
    }
};

} // namespace atomics
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_ATOMIC_H_