// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

// mars/mars/comm/socket/tcpserver.h

#ifndef BUBBLEFS_PLATFORM_MARS_TCPSERVER_H_
#define BUBBLEFS_PLATFORM_MARS_TCPSERVER_H_

#include "platform/mars_condition.h"
#include "platform/mars_lock.h"
#include "platform/mars_socketselect.h"
#include "platform/mars_thread.h"
#include "platform/mars_unix_socket.h"
#include "utils/mars_autobuffer.h"

namespace bubblefs {
namespace mymars {

class XLogger;
class SocketSelect;
class TcpServer;

class MTcpServer {
  public:
    virtual ~MTcpServer() {}
    virtual void OnCreate(TcpServer* _server) = 0;
    virtual void OnAccept(TcpServer* _server, SOCKET _sock, const sockaddr_in& _addr) = 0;
    virtual void OnError(TcpServer* _server, int _error) = 0;
};

class TcpServer {
  public:
    TcpServer(const char* _ip, uint16_t _port, MTcpServer& _observer, int _backlog = 256);
    TcpServer(uint16_t _port, MTcpServer& _observer, int _backlog = 256);
    TcpServer(const sockaddr_in& _bindaddr, MTcpServer& _observer, int _backlog = 256);
    ~TcpServer();

    SOCKET Socket() const;
    const sockaddr_in& Address() const;

    bool StartAndWait(bool* _newone = NULL);
    void StopAndWait();

  private:
    TcpServer(const TcpServer&);
    TcpServer& operator=(const TcpServer&);

  private:
    void __ListenThread();

  protected:
    MTcpServer&         observer_;
    Thread              thread_;
    Mutex                  mutex_;
    Condition            cond_;

    SOCKET                 listen_sock_;
    sockaddr_in         bind_addr_;
    const int             backlog_;

    SocketBreaker breaker_;
};

} // namespace mymars
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_MARS_TCPSERVER_H_