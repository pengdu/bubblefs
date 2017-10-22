/*
 * Copyright (c) 2011 The LevelDB Authors.
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// pdlfs-common/include/pdlfs-common/env.h

#ifndef BUBBLEFS_PLATFORM_LOGGER_H_
#define BUBBLEFS_PLATFORM_LOGGER_H_

#include <stdarg.h>

namespace bubblefs {

// An interface for writing log messages.
class Logger {
 public:
  Logger() {}
  virtual ~Logger();

  // Return the global logger that is shared by the entire process.
  // The result of Default() belongs to the system and cannot be deleted.
  static Logger* Default();

  // Write an entry to the log file with the specified format.
  virtual void Logv(const char* file, int line, int severity, int verbose,
                    const char* format, va_list ap) = 0;

 private:
  // No copying allowed
  void operator=(const Logger&);
  Logger(const Logger&);
};

} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_LOGGER_H_