// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

// mars/mars/comm/socket/unix_socket.h

#ifndef BUBBLEFS_PLATFORM_MARS_UNIX_SOCKET_H_
#define BUBBLEFS_PLATFORM_MARS_UNIX_SOCKET_H_

#include <sys/socket.h>
#include <sys/select.h>
#include <sys/uio.h>
#include <netinet/tcp.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <net/if.h>
#include <unistd.h>

namespace bubblefs {
namespace mymars {

#define SOCKET_ERRNO(error) error

#define socket_errno errno
#define socket_strerror strerror

#define socket_close close

#define socket_inet_ntop inet_ntop
#define socket_inet_pton inet_pton

#define SOCKET int
#define INVALID_SOCKET -1

#define IS_NOBLOCK_CONNECT_ERRNO(err) ((err) == SOCKET_ERRNO(EINPROGRESS))

#define IS_NOBLOCK_SEND_ERRNO(err) IS_NOBLOCK_WRITE_ERRNO(err)
#define IS_NOBLOCK_RECV_ERRNO(err) IS_NOBLOCK_READ_ERRNO(err)

#define IS_NOBLOCK_WRITE_ERRNO(err) ((err) == SOCKET_ERRNO(EAGAIN) || (err) == SOCKET_ERRNO(EWOULDBLOCK))
#define IS_NOBLOCK_READ_ERRNO(err)  ((err) == SOCKET_ERRNO(EAGAIN) || (err) == SOCKET_ERRNO(EWOULDBLOCK))


int socket_set_nobio(SOCKET fd);
int socket_set_tcp_mss(SOCKET sockfd, int size);
int socket_get_tcp_mss(SOCKET sockfd, int* size);
int socket_fix_tcp_mss(SOCKET sockfd);    // make mss=mss-40
int socket_disable_nagle(SOCKET sock, int nagle);
int socket_error(SOCKET sock);
int socket_reuseaddr(SOCKET sock, int optval);

int socket_get_nwrite(SOCKET _sock, int* _nwriteLen);
int socket_get_nread(SOCKET _sock, int* _nreadLen);
int socket_nwrite(SOCKET _sock);
int socket_nread(SOCKET _sock);
/*
 https://msdn.microsoft.com/zh-cn/library/windows/desktop/bb513665(v=vs.85).aspx
 Dual-Stack Sockets for IPv6 Winsock Applications
 By default, an IPv6 socket created on Windows Vista and later only operates over the IPv6 protocol. In order to make an IPv6 socket into a dual-stack socket, the setsockopt function must be called with the IPV6_V6ONLY socket option to set this value to zero before the socket is bound to an IP address. When the IPV6_V6ONLY socket option is set to zero, a socket created for the AF_INET6 address family can be used to send and receive packets to and from an IPv6 address or an IPv4 mapped address.
 */
int socket_ipv6only(SOCKET _sock, int _only);

int socket_isnonetwork(int error);

} // namespace mymars
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_MARS_UNIX_SOCKET_H_