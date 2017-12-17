/*  Copyright (c) 2013-2015 Tencent. All rights reserved.  */

// mars/mars/comm/unix/socket/socketselect.cc

#include "platform/mars_socketselect.h"

#include <poll.h>
#include <algorithm>
#include "platform/base_error.h"

namespace bubblefs {
namespace mymars {

SocketSelect::SocketSelect(SocketBreaker& _breaker, bool _autoclear)
: socket_poll_(_breaker, _autoclear)
{}

SocketSelect::~SocketSelect() {}

void SocketSelect::PreSelect() { socket_poll_.ClearEvent(); }
int  SocketSelect::Select() { return Select(-1); }
int  SocketSelect::Select(int _msec) { return socket_poll_.Poll(_msec); }

void SocketSelect::Read_FD_SET(SOCKET _socket) { socket_poll_.ReadEvent(_socket, true); }
void SocketSelect::Write_FD_SET(SOCKET _socket) { socket_poll_.WriteEvent(_socket, true); }
void SocketSelect::Exception_FD_SET(SOCKET _socket) { socket_poll_.NullEvent(_socket); }

int SocketSelect::Read_FD_ISSET(SOCKET _socket) const {
    const std::vector<PollEvent>& events = socket_poll_.TriggeredEvents();
    auto find_it = std::find_if(events.begin(), events.end(), [_socket](const PollEvent& _v){ return _v.FD() == _socket; });
    if (find_it == events.end()) return 0;
    return find_it->Readable() || find_it->HangUp();
}

int SocketSelect::Write_FD_ISSET(SOCKET _socket) const {
    const std::vector<PollEvent>& events = socket_poll_.TriggeredEvents();
    auto find_it = std::find_if(events.begin(), events.end(), [_socket](const PollEvent& _v){ return _v.FD() == _socket; });
    if (find_it == events.end()) {
        return 0;
    }
    return find_it->Writealbe();
}

int SocketSelect::Exception_FD_ISSET(SOCKET _socket) const {
    const std::vector<PollEvent>& events = socket_poll_.TriggeredEvents();
    auto find_it = std::find_if(events.begin(), events.end(), [_socket](const PollEvent& _v){ return _v.FD() == _socket; });
    if (find_it == events.end()) return 0;
    return find_it->Error() || find_it->Invalid();
}

int  SocketSelect::Ret() const { return socket_poll_.Ret(); }
int  SocketSelect::Errno() const { return socket_poll_.Errno(); }
bool SocketSelect::IsException() const { return socket_poll_.BreakerIsError(); }
bool SocketSelect::IsBreak() const { return socket_poll_.BreakerIsBreak(); }

SocketBreaker& SocketSelect::Breaker() { return socket_poll_.Breaker(); }
SocketPoll& SocketSelect::Poll() { return socket_poll_; }

} // namespace mymars
} // namespace bubblefs