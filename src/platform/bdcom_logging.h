// Copyright (c) 2014, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// baidu/common/logging.h

#ifndef BUBBLEFS_PLATFORM_BDCOM_LOGGING_H_
#define BUBBLEFS_PLATFORM_BDCOM_LOGGING_H_

#include <sstream>

namespace bubblefs {
  
namespace mybdcom {
  
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

} // namespace mybdcom

using mybdcom::DEBUG;
using mybdcom::INFO;
using mybdcom::WARNING;
using mybdcom::ERROR;
using mybdcom::FATAL;

#define LOG(level, fmt, args...) ::bubblefs::mybdcom::LogC(level, "[%s:%d] " fmt, __FILE__, __LINE__, ##args)
#define LOGS(level) ::bubblefs::mybdcom::LogStream(level)

} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_BDCOM_LOGGING_H_