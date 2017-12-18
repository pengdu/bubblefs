// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/http/http_request_parser.h

#ifndef BUBBLEFS_UTILS_VOYAGER_HTTP_REQUEST_PARSER_H_
#define BUBBLEFS_UTILS_VOYAGER_HTTP_REQUEST_PARSER_H_

#include "utils/voyager_http_request.h"

namespace bubblefs {
namespace myvoyager {

class Buffer;

class HttpRequestParser {
 public:
  HttpRequestParser();

  bool ParseBuffer(Buffer* buf);
  bool FinishParse() const { return state_ == kEnd; }

  HttpRequestPtr GetRequest() const { return request_; }

  void Reset();

 private:
  enum ParserState { kLine, kHeaders, kBody, kEnd };

  bool ParseRequestLine(const char* begin, const char* end);
  bool ParseRequestBody(Buffer* buf);

  ParserState state_;
  HttpRequestPtr request_;

  // No copying allowed
  HttpRequestParser(const HttpRequestParser&);
  void operator=(const HttpRequestParser&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_HTTP_REQUEST_PARSER_H_