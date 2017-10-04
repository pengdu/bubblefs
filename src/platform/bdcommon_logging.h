// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// baidu/common/logging.h

#ifndef BUBBLEFS_PLATFORM_BDCOMMON_LOGGING_H_
#define BUBBLEFS_PLATFORM_BDCOMMON_LOGGING_H_

#include <sstream>

namespace bubblefs {
  
namespace bdcommon {
  
enum LogLevel {
    DEBUG = 2,
    INFO = 4,
    WARNING = 8,
    ERROR = 16,
    FATAL = 32,
};

void SetLogLevel(int level);
bool SetLogFile(const char* path, bool append = false);
bool SetWarningFile(const char* path, bool append = false);
bool SetLogSize(int size); // in MB
bool SetLogCount(int count);
bool SetLogSizeLimit(int size); // in MB

void LogC(int level, const char* fmt, ...);

class LogStream {
public:
    LogStream(int level);
    template<class T>
    LogStream& operator<<(const T& t) {
        oss_ << t;
        return *this;
    }
    ~LogStream();
private:
    int level_;
    std::ostringstream oss_;
};

} // namespace bdcommon

using bdcommon::DEBUG;
using bdcommon::INFO;
using bdcommon::WARNING;
using bdcommon::ERROR;
using bdcommon::FATAL;

#define Log(level, fmt, args...) ::bubblefs::bdcommon::LogC(level, "[%s:%d] " fmt, __FILE__, __LINE__, ##args)
#define LogS(level) ::bubblefs::bdcommon::LogStream(level)

} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_BDCOMMON_LOGGING_H_