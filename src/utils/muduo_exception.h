// Use of this source code is governed by a BSD-style license
// that can be found in the License file.
//
// Author: Shuo Chen (chenshuo at chenshuo dot com)

// muduo/muduo/base/Exception.h

#ifndef BUBBLEFS_UTILS_MUDUO_EXCEPTION_H_
#define BUBBLEFS_UTILS_MUDUO_EXCEPTION_H_

#include <exception>
#include <string>

namespace bubblefs {
namespace mymuduo
{

class Exception : public std::exception
{
 public:
  explicit Exception(const char* what);
  explicit Exception(const std::string& what);
  virtual ~Exception() throw();
  virtual const char* what() const throw();
  const char* stackTrace() const throw();

 private:
  void fillStackTrace();

  std::string message_;
  std::string stack_;
};

} // namespace mymuduo
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_MUDUO_EXCEPTION_H_