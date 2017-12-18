// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/http/http_response_parser.h

#ifndef BUBBLEFS_UTILS_VOYAGER_HTTP_RESPONSE_PARSER_H_
#define BUBBLEFS_UTILS_VOYAGER_HTTP_RESPONSE_PARSER_H_

#include "utils/voyager_http_response.h"

namespace bubblefs {
namespace myvoyager {

class Buffer;

class HttpResponseParser {
 public:
  HttpResponseParser();

  bool ParseBuffer(Buffer* buf);
  bool FinishParse() const { return state_ == kEnd; }

  HttpResponsePtr GetResponse() const { return response_; }

  void Reset();

 private:
  enum ParserState { kLine, kHeaders, kBody, kEnd };

  bool ParseResponseLine(const char* begin, const char* end);
  bool ParseResponseBody(Buffer* buf);

  ParserState state_;
  HttpResponsePtr response_;

  // No copying allowed
  HttpResponseParser(const HttpResponseParser&);
  void operator=(const HttpResponseParser&);
};

}  // namespace myvoyager
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_VOYAGER_HTTP_RESPONSE_PARSER_H_