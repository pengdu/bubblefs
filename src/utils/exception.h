/**
 * Tencent is pleased to support the open source community by making Tars available.
 *
 * Copyright (C) 2016THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this file except 
 * in compliance with the License. You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software distributed 
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR 
 * CONDITIONS OF ANY KIND, either express or implied. See the License for the 
 * specific language governing permissions and limitations under the License.
 */

// protobuf/src/google/protobuf/stubs/common.h

#ifndef BUBBLEFS_UTILS_EXCEPTION_H_
#define BUBBLEFS_UTILS_EXCEPTION_H_

#include <exception>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bubblefs {
  
class FatalException : public std::exception {
 public:
  FatalException(const char* filename, int line, const std::string& message)
    : filename_(filename), line_(line) { 
    std::stringstream ss;
    ss << "FatalException ";
    ss << "[" << filename_ << ":" << line_ << "] ";
    ss << message;
    message_ = ss.str();    
  }
  virtual ~FatalException() throw();

  virtual const char* what() const throw() {
    return message_.c_str();
  }

  const char* filename() const { return filename_; }
  int line() const { return line_; }
  const std::string& message() const { return message_; }

 private:
  const char* filename_;
  const int line_;
  const std::string message_;
};

#define throw_fatal_exception(...) { char buffer[1000]; sprintf(buffer, __VA_ARGS__); std::string detail{buffer}; \
  throw FatalException(detail.c_str(), __FILE__, __LINE__, "Default FatalException"); }

} // namespace bubblefs

#endif BUBBLEFS_UTILS_EXCEPTION_H_