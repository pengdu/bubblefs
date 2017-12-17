// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

// mars/mars/comm/thread/spinlock.h

#ifndef BUBBLEFS_PLATFORM_MARS_SPINLOCK_H_
#define BUBBLEFS_PLATFORM_MARS_SPINLOCK_H_

#include <sched.h>
#include "platform/mars_atomic_oper.h"

namespace bubblefs {
namespace mymars {

/*
 * Improves the performance of spin-wait loops. 
 * When executing a “spin-wait loop,” a Pentium 4 or Intel Xeon processor 
 * suffers a severe performance penalty when exiting the loop 
 * because it detects a possible memory order violation. 
 * The PAUSE instruction provides a hint to the processor 
 * that the code sequence is a spin-wait loop. 
 * The processor uses this hint to avoid the memory order violation in most situations,
 * which greatly improves processor performance. 
 * For this reason, it is recommended that a PAUSE instruction 
 * be placed in all spin-wait loops.
 * An additional fucntion of the PAUSE instruction 
 * is to reduce the power consumed by a Pentium 4 processor 
 * while executing a spin loop.
*/

static inline void cpu_relax() {

#if defined(__arc__) || defined(__mips__) || defined(__arm__) || defined(__powerpc__)
        asm volatile("" ::: "memory");
#elif defined(__i386__) || defined(__x86_64__)
        asm volatile("rep; nop" ::: "memory");
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#elif defined(__ia64__)
        asm volatile ("hint @pause" ::: "memory");

#elif defined(_WIN32)
#if defined(_MSC_VER) && _MSC_VER >= 1310 && ( defined(_M_ARM) )
        YieldProcessor();
#else
        _mm_pause();
#endif
#endif

}  
  
class SpinLock
{
public:
     typedef uint32_t handle_type;
     
private:
     enum state
     {
         initial_pause = 2,
         max_pause = 16
     };

     uint32_t state_;

public:
     SpinLock() : state_(0) {}

     bool trylock()
     {
         return (atomic_cas32((volatile uint32_t *)&state_, 1, 0) == 0);
     }

     bool lock()
     {
         /*register*/ unsigned int pause_count = initial_pause; //'register' storage class specifier is deprecated and incompatible with C++1z
         while (!trylock())
         {
             if (pause_count < max_pause)
             {
                 for (/*register*/ unsigned int i = 0; i < pause_count; ++i) //'register' storage class specifier is deprecated and incompatible with C++1z
                 {
                     cpu_relax();
                 }
                 pause_count += pause_count;
             } else {
                 pause_count = initial_pause;
                 sched_yield();
             }
         }
         return true;
     }

     bool unlock()
     {
         atomic_write32((volatile uint32_t *)&state_, 0);
         return true;
     }
     
    uint32_t* internal() { return &state_; }

private:
     SpinLock(const SpinLock&);
     SpinLock& operator = (const SpinLock&);
};

} // namespace mymars
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_MARS_SPINLOCK_H_