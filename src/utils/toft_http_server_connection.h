// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/net/http/server/connection.h

#ifndef BUBBLFS_UTILS_TOFT_HTTP_SERVER_CONNECTION_H_
#define BUBBLFS_UTILS_TOFT_HTTP_SERVER_CONNECTION_H_

#include <list>
#include <string>
#include <vector>
#include "utils/toft_base_string_string_piece.h"
#include "platform/toft_system_event_dispatcher_event_dispatcher.h"
#include "platform/toft_system_net_socket.h"

namespace bubblefs {
namespace mytoft {

class HttpConnection {
    DECLARE_UNCOPYABLE(HttpConnection);

public:
    HttpConnection(EventDispatcher* dispatcher, int fd);
    void Send(const StringPiece& data);
    void Close();

private:
    void OnIoEvents(int events);
    void OnError();
    bool OnReadable();
    bool OnWriteable();
    void OnClosed();

private:
    StreamSocket m_socket;
    IoEventWatcher m_watcher;
    std::string m_receive_buffer;
    std::list<std::string> m_send_queue;
    size_t m_sent_size;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLFS_UTILS_TOFT_HTTP_SERVER_CONNECTION_H_