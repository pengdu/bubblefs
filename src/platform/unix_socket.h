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

#ifndef BUBBLEFS_PLATFORM_UNIX_SOCKET_H_
#define BUBBLEFS_PLATFORM_UNIX_SOCKET_H_

#include <sys/epoll.h>
#include <list>
#include <string>

/*
#include <netinet/in.h>

All pointers to socket address structures are often cast to pointers
to this type before use in various functions and system calls:

struct sockaddr {
  unsigned short    sa_family;    // address family, AF_xxx
  char              sa_data[14];  // 14 bytes of protocol address
};


IPv4 AF_INET sockets:

struct sockaddr_in {
    short            sin_family;   // e.g. AF_INET, AF_INET6
    unsigned short   sin_port;     // e.g. htons(3490)
    struct in_addr   sin_addr;     // see struct in_addr, below
    char             sin_zero[8];  // zero this if you want to
};

struct in_addr {
    unsigned long s_addr;          // load with inet_pton()
};


IPv6 AF_INET6 sockets:

struct sockaddr_in6 {
    u_int16_t       sin6_family;   // address family, AF_INET6
    u_int16_t       sin6_port;     // port number, Network Byte Order
    u_int32_t       sin6_flowinfo; // IPv6 flow information
    struct in6_addr sin6_addr;     // IPv6 address
    u_int32_t       sin6_scope_id; // Scope ID
};

struct in6_addr {
    unsigned char   s6_addr[16];   // load with inet_pton()
};


General socket address holding structure, big enough to hold either
struct sockaddr_in or struct sockaddr_in6 data:

struct sockaddr_storage {
    sa_family_t  ss_family;     // address family

    // all this is padding, implementation specific, ignore it:
    char      __ss_pad1[_SS_PAD1SIZE];
    int64_t   __ss_align;
    char      __ss_pad2[_SS_PAD2SIZE];
};

Example:
IPv4:

struct sockaddr_in ip4addr;
int s;

ip4addr.sin_family = AF_INET;
ip4addr.sin_port = htons(3490);
inet_pton(AF_INET, "10.0.0.1", &ip4addr.sin_addr);

s = socket(PF_INET, SOCK_STREAM, 0);
bind(s, (struct sockaddr*)&ip4addr, sizeof ip4addr);


IPv6:

struct sockaddr_in6 ip6addr;
int s;

ip6addr.sin6_family = AF_INET6;
ip6addr.sin6_port = htons(4950);
inet_pton(AF_INET6, "2001:db8:8714:3a90::12", &ip6addr.sin6_addr);

s = socket(PF_INET6, SOCK_STREAM, 0);
bind(s, (struct sockaddr*)&ip6addr, sizeof ip6addr);


struct addrinfo {
  int     ai_flags;
  int     ai_family;
  int     ai_socktype;
  int     ai_protocol;
  size_t  ai_addrlen;
  struct  sockaddr* ai_addr;
  char*   ai_canonname; 
  struct  addrinfo* ai_next;
};


#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

int getaddrinfo(const char *hostname, // e.g. "www.example.com" or IP
                const char *service,  // e.g. "http" or port number
                const struct addrinfo *hints,
                struct addrinfo **res);

hostname:一个主机名或者地址串(IPv4的点分十进制串或者IPv6的16进制串)
service：服务名可以是十进制的端口号，也可以是已定义的服务名称，如ftp、http等
hints：可以是一个空指针，也可以是一个指向某个addrinfo结构体的指针，调用者在
       这个结构中填入关于期望返回的信息类型的暗示。举例来说：如果指定的服务
       既支持TCP也支持UDP，那么调用者可以把hints结构中的ai_socktype成员设置
       成SOCK_DGRAM使得返回的仅仅是适用于数据报套接口的信息。
result：本函数通过result指针参数返回一个指向addrinfo结构体链表的指针。
返回值：0——成功，非0——出错


#include <sys/socket.h>
#include <netdb.h>

int getnameinfo(const struct sockaddr* sa, socklen_t salen,
                char* host, size_t hostlen,
                char* serv, size_t servlen,
                int flags);

#include <sys/socket.h>
#include <netdb.h>

void freeaddrinfo(struct addrinfo *ai);
ai参数应指向由getaddrinfo返回的第一个addrinfo结构。这个连表中的所有结构以及
它们指向的任何动态存储空间都被释放掉。


int status;
struct addrinfo hints;
struct addrinfo *servinfo;  // will point to the results

memset(&hints, 0, sizeof hints); // make sure the struct is empty
hints.ai_family = AF_UNSPEC;     // don't care IPv4 or IPv6
hints.ai_socktype = SOCK_STREAM; // TCP stream sockets
hints.ai_flags = AI_PASSIVE;     // fill in my IP for me

#include<netdb.h>
const char *gai_strerror( int error );
该函数以getaddrinfo返回的非0错误值的名字和含义为他的唯一参数，返回一个指向对应的出错信息串的指针。


if ((status = getaddrinfo(nullptr, "3490", &hints, &servinfo)) != 0) {
    fprintf(stderr, "getaddrinfo error: %s\n", gai_strerror(status));
    exit(1);
}

// servinfo now points to a linked list of 1 or more struct addrinfos

// ... do everything until you don't need servinfo anymore ....

freeaddrinfo(servinfo); // free the linked-list


#define _BSD_SOURCE  
#include <endian.h>

uint16_t htobe16(uint16_t host_16bits);
uint16_t htole16(uint16_t host_16bits);
uint16_t be16toh(uint16_t big_endian_16bits);
uint16_t le16toh(uint16_t little_endian_16bits);

uint32_t htobe32(uint32_t host_32bits);
uint32_t htole32(uint32_t host_32bits);
uint32_t be32toh(uint32_t big_endian_32bits);
uint32_t le32toh(uint32_t little_endian_32bits);

uint64_t htobe64(uint64_t host_64bits);
uint64_t htole64(uint64_t host_64bits);
uint64_t be64toh(uint64_t big_endian_64bits);
 */

namespace bubblefs {
namespace port {

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

int GetIpByIf(const char* if_name, std::string* ip);

typedef uint64_t NetAddr;
static const NetAddr INVAILD_NETADDR = UINT64_MAX;

#define NETADDR_IP_PRINT_FMT   "%u.%u.%u.%u:%u"
#define NETADDR_IP_PRINT_CTX(socket_info) \
    (socket_info->_ip & 0xFFU), ((socket_info->_ip >> 8) & 0xFFU), \
    ((socket_info->_ip >> 16) & 0xFFU), ((socket_info->_ip >> 24) & 0xFFU), \
    (((socket_info->_port & 0xFF) << 8) | ((socket_info->_port >> 8) & 0xFF))
    
static const uint8_t TCP_PROTOCOL = 0x80;
static const uint8_t UDP_PROTOCOL = 0x40;
static const uint8_t IN_BLOCKED = 0x10;
static const uint8_t ADDR_TYPE = 0x07;
static const uint8_t CONNECT_ADDR = 0x04;
static const uint8_t LISTEN_ADDR = 0x02;
static const uint8_t ACCEPT_ADDR = 0x01;
    
uint64_t NetAddressToUIN(const std::string& ip, uint16_t port);

void UINToNetAddress(uint64_t net_uin, std::string* ip, uint16_t* port);

struct SocketInfo
{
    void Reset();

    uint8_t GetProtocol() const { return (_state & 0xC0); }
    uint8_t GetAddrType() const { return (_state & 0x7); }

    int32_t _socket_fd;
    uint32_t _addr_info;  ///< _addr_info accept时本地监听的地址信息，TCP的主动连接时为可自动重连次数
    uint32_t _ip;
    uint16_t _port;
    uint8_t _state;
    uint8_t _uin;        ///< 复用时的标记
};

class NetIO;

class Epoll
{
  friend class NetIO;
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
    NetIO*  m_bind_net_io;
};

/// @brief 网络IO处理类，管理socket、封装网络操作
/// @note 使用地址忽略socket，上层不用考虑重连
class NetIO
{
    friend class Epoll;
public:
    NetIO();
    ~NetIO();

    /// @brief 初始化
    int32_t Init(Epoll* epoll);

    /// @brief 打开监听
    /// @note 主动创建的连接异常时会自动尝试恢复
    NetAddr Listen(const std::string& ip, uint16_t port);

    /// @brief 接受服务连接
    /// @note 被动打开的连接在连接异常时会自动释放
    NetAddr Accept(NetAddr listen_addr);

    /// @brief 连接地址
    /// @note 主动创建的连接异常时会自动尝试恢复
    NetAddr ConnectPeer(const std::string& ip, uint16_t port);

    /// @brief 发送数据
    /// @note 只负责发送，不保存任何发送残渣数据
    int32_t Send(NetAddr dst_addr, const char* data, uint32_t data_len);

    /// @brief 发送数据
    /// @note 只负责发送，不保存任何发送残渣数据
    int32_t SendV(NetAddr dst_addr, uint32_t data_num,
                  const char* data[], uint32_t data_len[], uint32_t offset = 0);

    /// @brief 发送数据
    /// @note only for udp server send rsp
    int32_t SendTo(NetAddr local_addr, NetAddr remote_addr, const char* data, uint32_t data_len);

    /// @brief 接收数据
    /// @return -1 读取失败或连接关闭了，错误见errno
    /// @return >= 0 读取的内容长度，可能为空
    /// @note 只负责发送，不保存任何接收的残渣数据
    int32_t Recv(NetAddr dst_addr, char* buff, uint32_t buff_len);

    /// @brief 接收数据
    /// @note only for udp listen
    int32_t RecvFrom(NetAddr local_addr, NetAddr* remote_addr, char* buff, uint32_t buff_len);

    /// @brief 关闭连接
    /// @return -1 关闭后返回错误，错误原因见errno
    /// @return 0  连接关闭了
    /// @note 失败的意义同close
    int32_t Close(NetAddr dst_addr);

    /// @brief 重置连接(先关后连，暂时只针对CONNECT_ADDR类型)
    /// @return -1 关闭后返回错误，错误原因见errno
    /// @return 0  重置成功
    int32_t Reset(NetAddr dst_addr); // TODO: -> ReConnect

    /// @brief 关闭所有连接
    void CloseAll();

    /// @brief 获取地址的socket相关信息
    /// @return 非NULL对象指针
    /// @note 地址不存在时返回的Info中数据全为0
    const SocketInfo* GetSocketInfo(NetAddr dst_addr) const;

    /// @brief 获取监听地址的socket相关信息
    /// @return 非NULL对象指针
    /// @note 地址不存在或地址不是被动连接的地址时，返回的Info中数据全为0
    /// @note 用于被动连接的地址获取关联的本地的监听地址信息
    const SocketInfo* GetLocalListenSocketInfo(NetAddr dst_addr) const;

    /// @brief 获取监听地址的句柄信息
    NetAddr GetLocalListenAddr(NetAddr dst_addr);

    const char* GetLastError() const {
        return m_last_error;
    }

    // 配置项
    static bool NON_BLOCK;              ///< NON_BLOCK 是否为非阻塞读写，默认为true
    static bool ADDR_REUSE;             ///< ADDR_REUSE 是否打开地址复用，默认为true
    static bool KEEP_ALIVE;             ///< KEEP_ALIVE 是否打开连接定时活性检测，默认为true
    static bool USE_NAGLE;              ///< USE_NAGLE 是否使用nagle算法合并小包，默认为false
    static bool USE_LINGER;             ///< USE_LINGER 是否使用linger延时关闭连接，默认为false
    static int32_t LINGER_TIME;         ///< LINGER_TIME 使用linget时延时时间设置，默认为0
    static int32_t LISTEN_BACKLOG;      ///< LISTEN_BACKLOG 监听的backlog队列长度，默认为10240
    static uint32_t MAX_SOCKET_NUM;     ///< MAX_SOCKET_NUM 最大的连接数，默认为1000000
    static uint8_t AUTO_RECONNECT;      ///< AUTO_RECONNECT 对TCP的主动连接自动重连，默认值为3

    // 常数
    static const uint32_t MAX_SENDV_DATA_NUM = 32;   ///< SendV接口最大发送的数据段数量

private:
    NetAddr AllocNetAddr();

    void FreeNetAddr(NetAddr net_addr);

    SocketInfo* RawGetSocketInfo(NetAddr net_addr);

    int32_t InitSocketInfo(const std::string& ip, uint16_t port, SocketInfo* socket_info);

    int32_t OnEvent(NetAddr net_addr, uint32_t events);

    int32_t RawListen(NetAddr net_addr, SocketInfo* socket_info);

    int32_t RawConnect(NetAddr net_addr, SocketInfo* socket_info);

    int32_t RawClose(SocketInfo* socket_info);

    char            m_last_error[256];

    Epoll           *m_epoll;

    NetAddr         m_used_id;
    SocketInfo      *m_sockets;
    std::list<NetAddr>  m_free_sockets;
};

}  // namespace port
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_UNIX_SOCKET_H_