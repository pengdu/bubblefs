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

// brpc/src/butil/unix_socket.cpp

#include "platform/brpc_unix_socket.h"
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/un.h>  
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "platform/base_error.h"
#include "utils/brpc_fd_guard.h"

namespace bubblefs {
namespace mybrpc {
  
int unix_socket_listen(const char* sockname, bool remove_previous_file) {
    struct sockaddr_un addr;
    addr.sun_family = AF_LOCAL;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sockname);

    fd_guard fd(socket(AF_LOCAL, SOCK_STREAM, 0));
    if (fd < 0) {
        PRINTF_ERROR("Fail to create unix socket\n");
        return -1;
    }
    if (remove_previous_file) {
        remove(sockname);
    }
    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        PRINTF_ERROR("Fail to bind sockfd=%d as unix socket=%s\n", fd.getfd(), sockname);
        return -1;
    }
    if (listen(fd, SOMAXCONN) != 0) {
        PRINTF_ERROR("Fail to listen to sockfd=%d\n", fd.getfd());
        return -1;
    }
    return fd.release();
}

int unix_socket_listen(const char* sockname) {
    return unix_socket_listen(sockname, true);
}

int unix_socket_connect(const char* sockname) {
    struct sockaddr_un addr;
    addr.sun_family = AF_LOCAL;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", sockname);

    fd_guard fd(socket(AF_LOCAL, SOCK_STREAM, 0));
    if (fd < 0) {
        PRINTF_ERROR("Fail to create unix socket\n");
        return -1;
    }
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        PRINTF_ERROR("Fail to connect to sockfd=%d as unix socket=%s\n", fd.getfd(), sockname);
        return -1;
    }
    return fd.release();
}

}  // namespace mybrpc
}  // namespace bubblefs