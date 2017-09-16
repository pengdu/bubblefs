/*
 * Tencent is pleased to support the open source community by making Pebble available.
 * Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.
 * Licensed under the MIT License (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 * http://opensource.org/licenses/MIT
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */

// Pebble/src/common/net_util.h

#ifndef BUBBLEFS_PLATFORM_NET_H_
#define BUBBLEFS_PLATFORM_NET_H_

#include <string>
#include <sys/epoll.h>

namespace bubblefs {
namespace netutil {
  
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
  
} // namespace netutil  
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_NET_H_