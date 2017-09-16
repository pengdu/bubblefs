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

// Pebble/src/common/net_util.cpp

#include "platform/net.h"
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

namespace bubblefs {
namespace netutil {
  
int GetIpByIf(const char* if_name, std::string* ip)
{
    if (nullptr == if_name) return -1;
    if (INADDR_NONE != inet_addr(if_name)) {
        ip->assign(if_name);
        return 0;
    }
    int fd, if_num;
    struct ifreq buf[32];
    struct ifconf ifc;
    if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        return -1;
    }

    ifc.ifc_len = sizeof(buf);
    ifc.ifc_buf = (caddr_t) buf;
    if (ioctl(fd, SIOCGIFCONF, (char*) &ifc)) { // NOLINT
        close(fd);
        return -2;
    }

    if_num = ifc.ifc_len / sizeof(struct ifreq);
    for (int if_idx = 0 ; if_idx < if_num ; ++if_idx) {
        if (0 != strcmp(buf[if_idx].ifr_name, if_name)) {
            continue;
        }
        if (0 == ioctl(fd, SIOCGIFADDR, (char *) &buf[if_idx])) { // NOLINT
            ip->assign(inet_ntoa((*((struct sockaddr_in *)(&buf[if_idx].ifr_addr))).sin_addr));
            close(fd);
            return 0;
        }
    }
    close(fd);
    return -1;
}

uint64_t NetAddressToUIN(const std::string& ip, uint16_t port)
{
    uint64_t ret = (static_cast<uint64_t>(inet_addr(ip.c_str())) << 32);
    ret |= htons(port);
    return ret;
}

void UINToNetAddress(uint64_t net_uin, std::string* ip, uint16_t* port)
{
    char tmp[32];
    uint32_t uip = static_cast<uint32_t>(net_uin >> 32);
    snprintf(tmp, sizeof(tmp), "%u.%u.%u.%u",
        (uip & 0xFF), ((uip >> 8) & 0xFF), ((uip >> 16) & 0xFF), ((uip >> 24) & 0xFF));
    ip->assign(tmp);
    *port = ntohs(static_cast<uint16_t>(net_uin & 0xFFFF));
}


Epoll::Epoll()
    :   m_epoll_fd(-1), m_max_event(1000), m_event_num(0), m_events(nullptr)
{
}

Epoll::~Epoll()
{
    if (m_epoll_fd >= 0) {
        close(m_epoll_fd);
    }
    if (nullptr != m_events) {
        m_event_num = 0;
        delete [] m_events;
    }
}

int32_t Epoll::Init(uint32_t max_event)
{
    m_last_error[0] = 0;
    m_max_event = static_cast<int32_t>(max_event);
    m_epoll_fd = epoll_create(m_max_event);
    m_events = new struct epoll_event[max_event];
    if (m_epoll_fd < 0) {
        return -1;
    }
    return 0;
}

int32_t Epoll::Wait(int32_t timeout)
{
    m_event_num = epoll_wait(m_epoll_fd, m_events, m_max_event, timeout);
    return m_event_num;
}

int32_t Epoll::AddFd(int32_t fd, uint32_t events, uint64_t data)
{
    struct epoll_event eve;
    eve.events = events;
    eve.data.u64 = data;
    return epoll_ctl(m_epoll_fd, EPOLL_CTL_ADD, fd, &eve);
}

int32_t Epoll::DelFd(int32_t fd)
{
    return epoll_ctl(m_epoll_fd, EPOLL_CTL_DEL, fd, nullptr);
}

int32_t Epoll::ModFd(int32_t fd, uint32_t events, uint64_t data)
{
    struct epoll_event eve;
    eve.events = events;
    eve.data.u64 = data;
    return epoll_ctl(m_epoll_fd, EPOLL_CTL_MOD, fd, &eve);
}

int32_t Epoll::GetEvent(uint32_t *events, uint64_t *data)
{
    if (m_event_num > 0) {
        --m_event_num;
        *events = m_events[m_event_num].events;
        *data = m_events[m_event_num].data.u64;
        //m_bind_net_io->OnEvent(*data, *events);
        return 0;
    }
    return -1;
}
  
} // namespace netutil
} // namespace bubblefs
