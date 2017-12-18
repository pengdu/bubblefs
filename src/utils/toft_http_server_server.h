// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/net/http/server/server.h

#ifndef BUBBLEFS_UTILS_TOFT_HTTP_SERVER_SERVER_H_
#define BUBBLEFS_UTILS_TOFT_HTTP_SERVER_SERVER_H_

#include <map>
#include <string>
#include "utils/toft_base_scoped_ptr.h"
#include "utils/toft_base_uncopyable.h"
#include "platform/toft_system_net_socket_address.h"

namespace bubblefs {
namespace mytoft {

class HttpHandler;

class HttpServer {
    DECLARE_UNCOPYABLE(HttpServer);

public:
    HttpServer();
    virtual ~HttpServer();
    bool RegisterHttpHandler(const std::string& path, HttpHandler* handler);
    bool Bind(const SocketAddress& address, SocketAddress* real_address = NULL);
    bool Start();
    void Close();
    void Run();

private:
    struct Impl;
    scoped_ptr<Impl> m_impl;
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_TOFT_HTTP_SERVER_SERVER_H_