// Copyright 2010, Shuo Chen.  All rights reserved.
// http://code.google.com/p/muduo/
//
// Use of this source code is governed by a BSD-style license
// that can be found in the License file.

// Author: Shuo Chen (chenshuo at chenshuo dot com)
//
// This is a public header file, it must only include public header files.

// muduo/muduo/base/ProcessInfo.h

#ifndef BUBBLEFS_UTILS_MUDUO_PROCESSINFO_H_
#define BUBBLEFS_UTILS_MUDUO_PROCESSINFO_H_

#include "platform/muduo_types.h"
#include "platform/muduo_timestamp.h"
#include "utils/muduo_stringpiece.h"
#include <vector>
#include <sys/types.h>

namespace bubblefs {
namespace mymuduo {
namespace ProcessInfo {
  
  pid_t pid();
  string pidString();
  uid_t uid();
  string username();
  uid_t euid();
  Timestamp startTime();
  int clockTicksPerSecond();
  int pageSize();
  bool isDebugBuild();  // constexpr

  string hostname();
  string procname();
  StringPiece procname(const string& stat);

  /// read /proc/self/status
  string procStatus();

  /// read /proc/self/stat
  string procStat();

  /// read /proc/self/task/tid/stat
  string threadStat();

  /// readlink /proc/self/exe
  string exePath();

  int openedFiles();
  int maxOpenFiles();

  struct CpuTime
  {
    double userSeconds;
    double systemSeconds;

    CpuTime() : userSeconds(0.0), systemSeconds(0.0) { }
  };
  CpuTime cpuTime();

  int numThreads();
  std::vector<pid_t> threads();
  
} // namespace ProcessInfo
} // namespace mymuduo
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_MUDUO_PROCESSINFO_H_