// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/net/http/server/handler.h

#ifndef BUBBLEFS_UTILS_TOFT_HTTP_SERVER_HANDLER_H_
#define BUBBLEFS_UTILS_TOFT_HTTP_SERVER_HANDLER_H_

#include "utils/toft_http_request.h"
#include "utils/toft_http_response.h"

namespace bubblefs {
namespace mytoft {

// Abstract base class of all concrete HttpHandler classes
class HttpHandler {
protected:
    HttpHandler();
public:
    virtual ~HttpHandler() {}
    virtual void HandleRequest(const HttpRequest* req, HttpResponse* resp);
    virtual void HandleGet(const HttpRequest* req, HttpResponse* resp);
    virtual void HandlePost(const HttpRequest* req, HttpResponse* resp);
    virtual void HandlePut(const HttpRequest* req, HttpResponse* resp);
    virtual void HandleHead(const HttpRequest* req, HttpResponse* resp);
    virtual void HandleDelete(const HttpRequest* req, HttpResponse* resp);
    virtual void HandleOptions(const HttpRequest* req, HttpResponse* resp);
    virtual void HandleTrace(const HttpRequest* req, HttpResponse* resp);
    virtual void HandleConnect(const HttpRequest* req, HttpResponse* resp);
protected:
    void MethodNotAllowed(const HttpRequest* req, HttpResponse* resp);
private:
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_TOFT_HTTP_SERVER_HANDLER_H_