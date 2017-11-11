// Copyright (c) 2011 Baidu, Inc.
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

// brpc/src/butil/fd_utility.h

// Utility functions on file descriptor.

#ifndef BUBBLEFS_PLATFORM_BRPC_FD_UTILITY_H_
#define BUBBLEFS_PLATFORM_BRPC_FD_UTILITY_H_

namespace bubblefs {
namespace mybrpc {

// Make file descriptor |fd| non-blocking
// Returns 0 on success, -1 otherwise and errno is set (by fcntl)
int make_non_blocking(int fd);

// Make file descriptor |fd| blocking
// Returns 0 on success, -1 otherwise and errno is set (by fcntl)
int make_blocking(int fd);

// Make file descriptor |fd| automatically closed during exec()
// Returns 0 on success, -1 when error and errno is set (by fcntl)
int make_close_on_exec(int fd);

// Disable nagling on file descriptor |socket|.
// Returns 0 on success, -1 when error and errno is set (by setsockopt)
int make_no_delay(int socket);

}  // namespace mybrpc
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_BRPC_FD_UTILITY_H_