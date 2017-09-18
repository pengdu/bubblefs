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

// Author: Jiang,Rujie(jiangrujie@baidu.com)
// Date: Mon. Jan 27  23:08:35 CST 2014

// brpc/src/butil/unix_socket.h
// Pebble/src/common/net_util.h

// Wrappers of unix domain sockets, mainly for unit-test of network stuff.

#ifndef BUBBLEFS_PLATFORM_UNIX_SOCKET_H_
#define BUBBLEFS_PLATFORM_UNIX_SOCKET_H_

#include <string>
#include <sys/epoll.h>

namespace bubblefs {
namespace base {

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

int GetIpByIf(const char* if_name, std::string* ip);

typedef uint64_t NetAddr;
static const NetAddr INVAILD_NETADDR = UINT64_MAX;
    
uint64_t NetAddressToUIN(const std::string& ip, uint16_t port);

void UINToNetAddress(uint64_t net_uin, std::string* ip, uint16_t* port);

class Epoll
{
public:
    Epoll();
    ~Epoll();

    int32_t Init(uint32_t max_event);

    /// @brief 等待事件触发
    /// @return >=0 触发的事件数
    /// @return -1 错误，原因见errno
    int32_t Wait(int32_t timeout_ms);

    int32_t AddFd(int32_t fd, uint32_t events, uint64_t data);

    int32_t DelFd(int32_t fd);

    int32_t ModFd(int32_t fd, uint32_t events, uint64_t data);

    /// @brief 获取事件
    /// @return 0 获取成功
    /// @return -1 获取失败，没有事件信息
    int32_t GetEvent(uint32_t *events, uint64_t *data);

    const char* GetLastError() const {
        return m_last_error;
    }

private:
    char    m_last_error[256];
    int32_t m_epoll_fd;
    int32_t m_max_event;
    int32_t m_event_num;
    struct epoll_event* m_events;
    //NetIO*  m_bind_net_io;
};

}  // namespace base
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_UNIX_SOCKET_H_