// Copyright (c) 2014 Baidu, Inc.
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

// brpc/src/butil/unix_socket.h

#ifndef BUBBLEFS_PLATFORM_BRPC_UNIX_SOCKET_H_
#define BUBBLEFS_PLATFORM_BRPC_UNIX_SOCKET_H_

#include <sys/epoll.h>
#include <list>
#include <string>

namespace bubblefs {
namespace mybrpc {

// Wrappers of unix domain sockets, mainly for unit-test of network stuff.  
  
// Create an unix domain socket at `sockname' and listen to it.
// If remove_previous_file is true or absent, remove previous file before
// creating the socket.
// Returns the file descriptor on success, -1 otherwise and errno is set.
int unix_socket_listen(const char* sockname, bool remove_previous_file);
int unix_socket_listen(const char* sockname);

// Create an unix domain socket and connect it to another listening unix domain
// socket at `sockname'.
// Returns the file descriptor on success, -1 otherwise and errno is set.
int unix_socket_connect(const char* sockname);

}  // namespace mybrpc
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_BRPC_UNIX_SOCKET_H_