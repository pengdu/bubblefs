// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Chen Feng <chen3feng@gmail.com>

// toft/system/process/this_process.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_PROCESS_THIS_PROCESS_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_PROCESS_THIS_PROCESS_H_

//#pragma once

#include <time.h>
#include <unistd.h>
#include <string>

namespace bubblefs {
namespace mytoft {

class ThisProcess
{
    ThisProcess();
    ~ThisProcess();
public:
    // Process id of current process.
    static pid_t GetId();

    // Obtain executable file path.
    static std::string BinaryPath();

    // Obtain executable file name.
    static std::string BinaryName();

    // Obtain directory of binary executable file.
    static std::string BinaryDirectory();

    // Obtain start time in seconds since Epoch.
    static time_t StartTime();

    // Obtain elapsed seconds since start.
    static time_t ElapsedTime();
};

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_PROCESS_THIS_PROCESS_H_