//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// exception.h
//
// Identification: src/include/common/exception.h
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// protobuf/src/google/protobuf/stubs/common.h
// peloton/src/include/common/exception.h

#ifndef BUBBLEFS_UTILS_EXCEPTION_H_
#define BUBBLEFS_UTILS_EXCEPTION_H_

#include <cxxabi.h>
#include <errno.h>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bubblefs {
  
enum ExceptionType {
  EXCEPTION_TYPE_INVALID = 0,            // invalid type
};

class Exception : public std::runtime_error {
 public:
  Exception(std::string message)
      : std::runtime_error(message), type(EXCEPTION_TYPE_INVALID) {
    std::string exception_message = "Message :: " + message + "\n";
    std::cerr << exception_message;
  }

  Exception(ExceptionType exception_type, std::string message)
      : std::runtime_error(message), type(exception_type) {
    std::string exception_message = "Exception Type :: " +
                                    ExpectionTypeToString(exception_type) +
                                    "\nMessage :: " + message + "\n";
    std::cerr << exception_message;
  }
  
  std::string ExpectionTypeToString(ExceptionType type) {
    switch (type) {
      case EXCEPTION_TYPE_INVALID:
        return "Invalid";
      default:
        return "Unknown";
    }
  }
  
  // Based on :: http://panthema.net/2008/0901-stacktrace-demangled/
  static void PrintStackTrace(FILE *out = stderr,
                              unsigned int max_frames = 63) {
    fprintf(out, "Stack Trace:\n");

    /// storage array for stack trace address data
    void *addrlist[max_frames + 1];

    /// retrieve current stack addresses
    int addrlen = backtrace(addrlist, max_frames + 1);

    if (addrlen == 0) {
      fprintf(out, "  <empty, possibly corrupt>\n");
      return;
    }

    /// resolve addresses into strings containing "filename(function+address)",
    char **symbol_list = backtrace_symbols(addrlist, addrlen);

    /// allocate string which will be filled with the demangled function name
    size_t func_name_size = 1024;
    std::unique_ptr<char> func_name(new char[func_name_size]);

    /// iterate over the returned symbol lines. skip the first, it is the
    /// address of this function.
    for (int i = 1; i < addrlen; i++) {
      char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

      /// find parentheses and +address offset surrounding the mangled name:
      /// ./module(function+0x15c) [0x8048a6d]
      for (char *p = symbol_list[i]; *p; ++p) {
        if (*p == '(')
          begin_name = p;
        else if (*p == '+')
          begin_offset = p;
        else if (*p == ')' && begin_offset) {
          end_offset = p;
          break;
        }
      }

      if (begin_name && begin_offset && end_offset &&
          begin_name < begin_offset) {
        *begin_name++ = '\0';
        *begin_offset++ = '\0';
        *end_offset = '\0';

        /// mangled name is now in [begin_name, begin_offset) and caller
        /// offset in [begin_offset, end_offset). now apply  __cxa_demangle():
        int status;
        char *ret = abi::__cxa_demangle(begin_name, func_name.get(),
                                        &func_name_size, &status);
        if (status == 0) {
          func_name.reset(ret);  // use possibly realloc()-ed string
          fprintf(out, "  %s : %s+%s\n", symbol_list[i], func_name.get(),
                    begin_offset);
        } else {
          /// demangling failed. Output function name as a C function with
          /// no arguments.
          fprintf(out, "  %s : %s()+%s\n", symbol_list[i], begin_name,
                    begin_offset);
        }
      } else {
        /// couldn't parse the line ? print the whole line.
        fprintf(out, "  %s\n", symbol_list[i]);
      }
    }
  }

 private:
  // type
  ExceptionType type;
};
  
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