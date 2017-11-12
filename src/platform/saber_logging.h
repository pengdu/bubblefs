// Copyright (c) 2017 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// saber/saber/util/logging.h

#ifndef BUBBLEFS_PLATFORM_SABER_LOGGING_H_
#define BUBBLEFS_PLATFORM_SABER_LOGGING_H_

#include <inttypes.h>
#include <stdarg.h>

namespace bubblefs {
namespace mysaber {

enum LogLevel {
  LOGLEVEL_DEBUG,
  LOGLEVEL_INFO,
  LOGLEVEL_WARN,
  LOGLEVEL_ERROR,
  LOGLEVEL_FATAL
};

extern void Log(LogLevel level, const char* filename, int line,
                const char* format, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((__format__(__printf__, 4, 5)))
#endif
    ;

#define SABER_LOG_DEBUG(format, ...)                                      \
  ::bubblefs::mysaber::Log(::bubblefs::mysaber::LOGLEVEL_DEBUG, __FILE__, __LINE__, format, \
                         ##__VA_ARGS__)

#define SABER_LOG_INFO(format, ...)                                      \
  ::bubblefs::mysaber::Log(::bubblefs::mysaber::LOGLEVEL_INFO, __FILE__, __LINE__, format, \
                         ##__VA_ARGS__)

#define SABER_LOG_WARN(format, ...)                                      \
  ::bubblefs::mysaber::Log(::bubblefs::mysaber::LOGLEVEL_WARN, __FILE__, __LINE__, format, \
                         ##__VA_ARGS__)

#define SABER_LOG_ERROR(format, ...)                                      \
  ::bubblefs::mysaber::Log(::bubblefs::mysaber::LOGLEVEL_ERROR, __FILE__, __LINE__, format, \
                         ##__VA_ARGS__)

#define SABER_LOG_FATAL(format, ...)                                      \
  ::bubblefs::mysaber::Log(::bubblefs::mysaber::LOGLEVEL_FATAL, __FILE__, __LINE__, format, \
                         ##__VA_ARGS__)

extern void DefaultLogHandler(LogLevel level, const char* filename, int line,
                              const char* format, va_list ap);

typedef void LogHandler(LogLevel level, const char* filename, int line,
                        const char* format, va_list ap);

extern LogHandler* SetLogHandler(LogHandler* new_handler);

extern LogLevel SetLogLevel(LogLevel new_level);

}  // namespace mysaber
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_SABER_LOGGING_H_