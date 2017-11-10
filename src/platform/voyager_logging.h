// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// voyager/voyager/util/logging.h

#ifndef BUBBLEFS_PLATFORM_VOYAGER_LOGGING_H_
#define BUBBLEFS_PLATFORM_VOYAGER_LOGGING_H_

#include <string>
#include <thread>
#include "utils/status.h"
#include "utils/stringpiece.h"

namespace bubblefs {
namespace voyager {

enum LogLevel {
  LOGLEVEL_DEBUG,
  LOGLEVEL_INFO,
  LOGLEVEL_WARN,
  LOGLEVEL_ERROR,
  LOGLEVEL_FATAL
};

class LogFinisher;
using Slice = StringPiece;

class Logger {
 public:
  Logger(LogLevel level, const char* filename, int line);
  ~Logger() {}

  Logger& operator<<(char value);
  Logger& operator<<(short value);
  Logger& operator<<(unsigned short value);
  Logger& operator<<(int value);
  Logger& operator<<(unsigned int value);
  Logger& operator<<(long value);
  Logger& operator<<(unsigned long value);
  Logger& operator<<(long long value);
  Logger& operator<<(unsigned long long value);
  Logger& operator<<(double value);
  Logger& operator<<(void* value);
  Logger& operator<<(const void* value);
  Logger& operator<<(const char* value);
  Logger& operator<<(const Slice& value);
  Logger& operator<<(const std::string& value);
  Logger& operator<<(std::string&& value);
  Logger& operator<<(const Status& value);
  Logger& operator<<(const std::thread::id& value);

 private:
  friend class LogFinisher;
  void Finish();

  LogLevel level_;
  const char* filename_;
  int line_;
  std::string message_;
};

class LogFinisher {
 public:
  void operator=(Logger& logger);
};

#define VOYAGER_LOG(LEVEL)   \
  ::bubblefs::voyager::LogFinisher() = \
      ::bubblefs::voyager::Logger(::bubblefs::voyager::LOGLEVEL_##LEVEL, __FILE__, __LINE__)

template <typename T>
T* CheckNotNull(const char* /* filename */, int /* line */,
                const char* logmessage, T* ptr) {
  if (ptr == nullptr) {
    VOYAGER_LOG(FATAL) << logmessage;
  }
  return ptr;
}

#define VOYAGER_CHECK_NOTNULL(value)                  \
  ::bubblefs::voyager::CheckNotNull(__FILE__, __LINE__, \
                                    "'" #value "' Must not be nullptr", (value))

typedef void LogHandler(LogLevel level, const char* filename, int line,
                        const std::string& message);

extern void DefaultLogHandler(LogLevel level, const char* filename, int line,
                              const std::string& message);

extern void NullLogHandler(LogLevel /* level */, const char* /* filename */,
                           int /* line */, const std::string& /* message */);

extern LogHandler* SetLogHandler(LogHandler* new_func);

extern LogLevel SetLogLevel(LogLevel new_level);

}  // namespace voyager
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_VOYAGER_LOGGING_H_