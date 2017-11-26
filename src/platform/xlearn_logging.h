//------------------------------------------------------------------------------
// Copyright (c) 2016 by contributors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

// xlearn/src/base/logging.h

/*
Author: Chao Ma (mctt90@gmail.com)
This file provides logging facilities treating log messages by
their severities.
*/

#ifndef BUBBLEFS_PLATFORM_XLEARN_LOGGING_H_
#define BUBBLEFS_PLATFORM_XLEARN_LOGGING_H_

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace bubblefs {
namespace myxlearn {

//------------------------------------------------------------------------------
// If function |InitializeLogger| was invoked and was able to open
// files specified by the parameters, log messages of various severity
// will be written into corresponding files. Otherwise, all log messages
// will be written to stderr. For example:
//
//   int main() {
//     InitializeLogger("/tmp/info.log", "/tmp/warn.log", "/tmp/erro.log");
//     LOG(INFO)    << "An info message going into /tmp/info.log";
//     LOG(WARNING) << "An warn message going into /tmp/warn.log";
//     LOG(ERROR)   << "An erro message going into /tmp/erro.log";
//     LOG(FATAL)   << "An fatal message going into /tmp/erro.log, "
//                  << "and kills current process by a segmentation fault.";
//     return 0;
//   }
//------------------------------------------------------------------------------
  
void InitializeLogger(const std::string& info_log_filename,
                      const std::string& warn_log_filename,
                      const std::string& erro_log_filename);

enum LogSeverity { INFO, WARNING, ERROR, FATAL };

class Logger {
  friend void InitializeLogger(const std::string& info_log_filename,
                               const std::string& warn_log_filename,
                               const std::string& erro_log_filename);
 public:
  Logger(LogSeverity s) : severity_(s) {}
  ~Logger();

  static std::ostream& GetStream(LogSeverity severity);
  static std::ostream& Start(LogSeverity severity,
                             const std::string& file,
                             int line,
                             const std::string& function);

 private:
  static std::ofstream info_log_file_;
  static std::ofstream warn_log_file_;
  static std::ofstream erro_log_file_;
  LogSeverity severity_;
};

//-----------------------------------------------------------------------------
// The basic mechanism of logging.{h,cc} is as follows:
//  - LOG(severity) defines a Logger instance, which records the severity.
//  - LOG(severity) then invokes Logger::Start(), which invokes Logger::Stream
//    to choose an output stream, outputs a message head into the stream and
//    flush.
//  - The std::ostream reference returned by LoggerStart() is then passed to
//    user-specific output operators (<<), which writes the log message body.
//  - When the Logger instance is destructed, the destructor appends flush.
//    If severity is FATAL, the destructor causes SEGFAULT and core dump.
//
// It is important to flush in Logger::Start() after outputing message
// head.  This is because that the time when the destructor is invoked
// depends on how/where the caller code defines the Logger instance.
// If the caller code crashes before the Logger instance is properly
// destructed, the destructor might not have the chance to append its
// flush flags.  Without flush in Logger::Start(), this may cause the
// lose of the last few messages.  However, given flush in Start(),
// program crashing between invocations to Logger::Start() and
// destructor only causes the lose of the last message body; while the
// message head will be there.
//-----------------------------------------------------------------------------
#define LOG(severity)                                                       \
  bubblefs::myxlearn::Logger(severity).Start(severity, __FILE__, __LINE__, __FUNCTION__)
  
//------------------------------------------------------------------------------
// In cases when the program must quit immediately (e.g., due to
// severe bugs), CHECK_xxxx macros invoke abort() to cause a core
// dump.  To ensure the generation of the core dump, you might want to
// set the following shell option:
//
//        ulimit -c unlimited
//
// Once the core dump is generated, we can check the check failure
// using a debugger, for example, GDB:
//
//        gdb program_file core
//
// The GDB command 'where' will show you the stack trace.
//------------------------------------------------------------------------------

#define CHECK(a) if (!(a)) {                            \
    LOG(ERROR) << "CHECK failed "                       \
               << __FILE__ << ":" << __LINE__ << "\n"   \
               << #a << " = " << (a) << "\n";           \
    abort();                                            \
  }                                                     \

#define CHECK_NOTNULL(a) if ((a) == NULL) {             \
    LOG(ERROR) << "CHECK failed "                       \
               << __FILE__ << ":" << __LINE__ << "\n"   \
               << #a << " == NULL \n";                  \
    abort();                                            \
  }                                                     \

#define CHECK_NULL(a) if ((a) != NULL) {                \
    LOG(ERROR) << "CHECK failed "                       \
               << __FILE__ << ":" << __LINE__ << "\n"   \
               << #a << " = " << (a) << "\n";           \
    abort();                                            \
  }                                                     \

#define CHECK_EQ(a, b) if (!((a) == (b))) {             \
    LOG(ERROR) << "CHECK_EQ failed "                    \
               << __FILE__ << ":" << __LINE__ << "\n"   \
               << #a << " = " << (a) << "\n"            \
               << #b << " = " << (b) << "\n";           \
    abort();                                            \
  }                                                     \

#define CHECK_NE(a, b) if (!((a) != (b))) {             \
    LOG(ERROR) << "CHECK_NE failed "                    \
               << __FILE__ << ":" << __LINE__ << "\n"   \
               << #a << " = " << (a) << "\n"            \
               << #b << " = " << (b) << "\n";           \
    abort();                                            \
  }                                                     \

#define CHECK_GT(a, b) if (!((a) > (b))) {              \
    LOG(ERROR) << "CHECK_GT failed "                    \
               << __FILE__ << ":" << __LINE__ << "\n"   \
               << #a << " = " << (a) << "\n"            \
               << #b << " = " << (b) << "\n";           \
    abort();                                            \
  }                                                     \

#define CHECK_LT(a, b) if (!((a) < (b))) {              \
    LOG(ERROR) << "CHECK_LT failed "                    \
               << __FILE__ << ":" << __LINE__ << "\n"   \
               << #a << " = " << (a) << "\n"            \
               << #b << " = " << (b) << "\n";           \
    abort();                                            \
  }                                                     \

#define CHECK_GE(a, b) if (!((a) >= (b))) {             \
    LOG(ERROR) << "CHECK_GE failed "                    \
               << __FILE__ << ":" << __LINE__ << "\n"   \
               << #a << " = " << (a) << "\n"            \
               << #b << " = " << (b) << "\n";           \
    abort();                                            \
  }                                                     \

#define CHECK_LE(a, b) if (!((a) <= (b))) {             \
    LOG(ERROR) << "CHECK_LE failed "                    \
               << __FILE__ << ":" << __LINE__ << "\n"   \
               << #a << " = " << (a) << "\n"            \
               << #b << " = " << (b) << "\n";           \
    abort();                                            \
  }                                                     \
                                                        \
// Copied from glog.h
#define CHECK_DOUBLE_EQ(a, b)                           \
  do {                                                  \
    CHECK_LE((a), (b)+0.000000000000001L);              \
    CHECK_GE((a), (b)-0.000000000000001L);              \
  } while (0)

#define CHECK_NEAR(a, b, margin)                        \
  do {                                                  \
    CHECK_LE((a), (b)+(margin));                        \
    CHECK_GE((a), (b)-(margin));                        \
  } while (0)
  
} // namespace myxlearn
} // namespace bubblefs

#endif   // BUBBLEFS_PLATFORM_XLEARN_LOGGING_H_