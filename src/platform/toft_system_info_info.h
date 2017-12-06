// Copyright (c) 2010, The Toft Authors. All rights reserved.
// Author: Chen Feng <chen3feng@gmail.com>

// toft/system/info/info.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_INFO_INFO_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_INFO_INFO_H_

// GLOBAL_NOLINT(runtime/int)

#include <string>

namespace bubblefs {
namespace mytoft {

unsigned int GetLogicalCpuNumber();
unsigned long long GetPhysicalMemorySize();
unsigned long long GetTotalPhysicalMemorySize();
bool GetOsKernelInfo(std::string* kernerl_info);
bool GetMachineArchitecture(std::string* arch_info);
std::string GetUserName();

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_INFO_INFO_H_