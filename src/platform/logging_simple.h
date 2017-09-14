// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: yanshiguang02@baidu.com

// baidu/common/logging.h in c++11

#ifndef BUBBLEFS_PLATFORM_LOGGING_SIMPLE_H_
#define BUBBLEFS_PLATFORM_LOGGING_SIMPLE_H_

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

void Log(int level, const char* fmt, ...);

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

#define BDLOG(level, fmt, args...) ::bubblefs::bdcommon::Log(level, "[%s:%d] " fmt, __FILE__, __LINE__, ##args)
#define BDLOGS(level) ::bubblefs::bdcommon::LogStream(level)

} // namespace bdcommon
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_LOGGING_SIMPLE_H_