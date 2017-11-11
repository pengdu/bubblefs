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

// brpc/src/bthread/sys_futex.cpp

#include "platform/brpc_sys_futex.h"

namespace bubblefs {
namespace mybrpc {

const int SYS_FUTEX_PRIVATE_FLAG = 128;

static int get_futex_private_flag() {
    static int dummy = 0;
    if (!syscall(SYS_futex, &dummy, (FUTEX_WAKE | SYS_FUTEX_PRIVATE_FLAG),
                 1, NULL, NULL, 0)) {
        return SYS_FUTEX_PRIVATE_FLAG;
    }
    return 0;
}

extern const int futex_private_flag = get_futex_private_flag();

}  // namespace mybrpc
}  // namespace bubblefs