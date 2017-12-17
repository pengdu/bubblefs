// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

// mars/mars/comm/socket/udpclient.h

#ifndef BUBBLEFS_PLATFORM_MARS_UDPCLIENT_H_
#define BUBBLEFS_PLATFORM_MARS_UDPCLIENT_H_

#include <string>
#include <list>

#include "platform/mars_unix_socket.h"
#include "platform/mars_socketselect.h"
#include "platform/mars_thread.h"
#include "platform/mars_mutex.h"
#include "utils/mars_autobuffer.h"

namespace bubblefs {
namespace mymars {

#define IPV4_BROADCAST_IP "255.255.255.255"

struct UdpSendData;
class UdpClient;

class IAsyncUdpClientEvent {
  public:
    virtual ~IAsyncUdpClientEvent() {}
    virtual void OnError(UdpClient* _this, int _errno) = 0;
    virtual void OnDataGramRead(UdpClient* _this, void* _buf, size_t _len) = 0;
    virtual void OnDataSent(UdpClient* _this) = 0;
};

class UdpClient {
  public:
    UdpClient(const std::string& _ip, int _port);
    UdpClient(const std::string& _ip, int _port, IAsyncUdpClientEvent* _event);
    ~UdpClient();

    /*
     * return -2 break, -1 error, 0 timeout, else handle size
     */
    int SendBlock(void* _buf, size_t _len);
    int ReadBlock(void* _buf, size_t _len, int _timeOutMs = -1);

    void Break() { breaker_.Break(); }

    bool HasBuuferToSend();
    void SendAsync(void* _buf, size_t _len);

    void SetIpPort(const std::string& _ip, int _port);

  private:
    void __InitSocket(const std::string& _ip, int _port);
    int __DoSelect(bool _bReadSet, bool _bWriteSet, void* _buf, size_t _len, int& _errno, int _timeoutMs);
    void __RunLoop();

  private:
    SOCKET fd_socket_;
    struct sockaddr_in addr_;
    IAsyncUdpClientEvent* event_;

        SocketBreaker breaker_;
    SocketSelect selector_;
    Thread* thread_;

    std::list<UdpSendData> list_buffer_;
    Mutex mutex_;
};

} // namespace mymars
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_MARS_UDPCLIENT_H_