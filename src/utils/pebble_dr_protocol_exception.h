/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

// Pebble/src/framework/dr/protocol/protocol_exception.h

#ifndef BUBBLEFS_UTILS_PEBBLE_DR_PROTOCOL_EXCEPTION_H_
#define BUBBLEFS_UTILS_PEBBLE_DR_PROTOCOL_EXCEPTION_H_

#include <string>
#include "utils/pebble_dr_common.h"

namespace bubblefs {
namespace mypebble { namespace dr { namespace protocol {

/**
 * Class to encapsulate all the possible types of protocol errors that may
 * occur in various protocol systems. This provides a sort of generic
 * wrapper around the vague UNIX E_ error codes that lets a common code
 * base of error handling to be used for various types of protocols, i.e.
 * pipes etc.
 *
 */
class TProtocolException : public mypebble::TException {
 public:

  /**
   * Error codes for the various types of exceptions.
   */
  enum TProtocolExceptionType
  { UNKNOWN = 0
  , INVALID_DATA = 1
  , NEGATIVE_SIZE = 2
  , SIZE_LIMIT = 3
  , BAD_VERSION = 4
  , NOT_IMPLEMENTED = 5
  , DEPTH_LIMIT = 6
  , NULL_PROTOCOL = 7
  };

  TProtocolException() :
    TException(),
    type_(UNKNOWN) {}

  TProtocolException(TProtocolExceptionType type) :
    TException(),
    type_(type) {}

  TProtocolException(const std::string& message) :
    TException(message),
    type_(UNKNOWN) {}

  TProtocolException(TProtocolExceptionType type, const std::string& message) :
    TException(message),
    type_(type) {}

  virtual ~TProtocolException() throw() {}

  /**
   * Returns an error code that provides information about the type of error
   * that has occurred.
   *
   * @return Error code
   */
  TProtocolExceptionType getType() {
    return type_;
  }

  virtual const char* what() const throw() {
    if (message_.empty()) {
      switch (type_) {
        case UNKNOWN         : return "TProtocolException: Unknown protocol exception";
        case INVALID_DATA    : return "TProtocolException: Invalid data";
        case NEGATIVE_SIZE   : return "TProtocolException: Negative size";
        case SIZE_LIMIT      : return "TProtocolException: Exceeded size limit";
        case BAD_VERSION     : return "TProtocolException: Invalid version";
        case NOT_IMPLEMENTED : return "TProtocolException: Not implemented";
        case NULL_PROTOCOL   : return "TProtocolException: Group protocol is empty";
        default              : return "TProtocolException: (Invalid exception type)";
      }
    } else {
      return message_.c_str();
    }
  }

 protected:
  /**
   * Error code
   */
  TProtocolExceptionType type_;

};

}}}} // bubblefs::mypebble::dr::protocol

#endif // BUBBLEFS_UTILS_PEBBLE_DR_PROTOCOL_EXCEPTION_H_