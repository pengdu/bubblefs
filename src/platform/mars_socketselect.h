/*  Copyright (c) 2013-2015 Tencent. All rights reserved.  */

// mars/mars/comm/unix/socket/socketselect.h

#ifndef BUBBLEFS_PALTFORM_MARS_SOCKETSELECT_H_
#define BUBBLEFS_PALTFORM_MARS_SOCKETSELECT_H_

#include "platform/mars_socketpoll.h"

namespace bubblefs {
namespace mymars {

class SocketSelect {
  public:
    SocketSelect(SocketBreaker& _breaker, bool _autoclear = false);
    virtual ~SocketSelect();

    void PreSelect();
    void Read_FD_SET(SOCKET _socket);
    void Write_FD_SET(SOCKET _socket);
    void Exception_FD_SET(SOCKET _socket);
    
    virtual int Select();
    virtual int Select(int _msec);

    int  Ret() const;
    int  Errno() const;

    int  Read_FD_ISSET(SOCKET _socket) const;
    int  Write_FD_ISSET(SOCKET _socket) const;
    int  Exception_FD_ISSET(SOCKET _socket) const;

    bool IsBreak() const;
    bool IsException() const;

    SocketBreaker& Breaker();

    SocketPoll&  Poll();
    
  private:
    SocketSelect(const SocketSelect&);
    SocketSelect& operator=(const SocketSelect&);

  protected:
    SocketPoll  socket_poll_;
};

} // namespace mymars
} // namespace bubblefs

#endif // BUBBLEFS_PALTFORM_MARS_SOCKETSELECT_H_