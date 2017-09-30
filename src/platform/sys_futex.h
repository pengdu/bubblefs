// bthread - A M:N threading library to make applications more concurrent.
// Copyright (c) 2012 Baidu, Inc.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Ge,Jun (gejun@baidu.com)
// Date: Tue Jul 10 17:40:58 CST 2012

// brpc/src/bthread/sys_futex.h

#ifndef BUBBLEFS_SYS_FUTEX_H_
#define BUBBLEFS_SYS_FUTEX_H_

#include <linux/futex.h>                // FUTEX_WAIT, FUTEX_WAKE
#include <syscall.h>                    // SYS_futex
#include <time.h>                       // timespec
#include <unistd.h>                     // syscall

namespace bubblefs {
namespace port {

extern const int futex_private_flag;

inline int futex_wait_private(
    void* addr1, int expected, const timespec* timeout) {
    return syscall(SYS_futex, addr1, (FUTEX_WAIT | futex_private_flag),
                   expected, timeout, NULL, 0);
}

inline int futex_wake_private(void* addr1, int nwake) {
    return syscall(SYS_futex, addr1, (FUTEX_WAKE | futex_private_flag),
                   nwake, NULL, NULL, 0);
}

inline int futex_requeue_private(void* addr1, int nwake, void* addr2) {
    return syscall(SYS_futex, addr1, (FUTEX_REQUEUE | futex_private_flag),
                   nwake, NULL, addr2, 0);
}

}  // namespace port
}  // namespace bubblefs

#endif // BUBBLEFS_SYS_FUTEX_H_