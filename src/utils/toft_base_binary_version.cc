// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>
// Created: 05/23/11

// toft/base/binary_version.cpp
// toft/base/setup_binary_version.cpp

#include "utils/toft_base_binary_version.h"
#include <sstream>

#include "gflags/gflags.h"

extern "C" {
namespace binary_version {
__attribute__((weak)) extern const int kSvnInfoCount = 0;
__attribute__((weak)) extern const char * const kSvnInfo[] = {0};
__attribute__((weak)) extern const char kBuildType[] = "Unknown";
__attribute__((weak)) extern const char kBuildTime[] = "Unknown";
__attribute__((weak)) extern const char kBuilderName[] = "Unknown";
__attribute__((weak)) extern const char kHostName[] = "Unknown";
__attribute__((weak)) extern const char kCompiler[] = "Unknown";
} // namespace binary_version
}

namespace bubblefs {
namespace mytoft {

// This function can't be in binary_version.cpp,
// otherwise, compiler optmizer will use kSvnInfoCount as 0 at compile time.
// and then generate empty version info.
static std::string MakeVersionInfo()
{
    using namespace binary_version;

    std::ostringstream oss;
    oss << "\n"; // Open a new line in gflags --version output.

    if (kSvnInfoCount > 0)
    {
        oss << "Sources:\n"
            << "----------------------------------------------------------\n";
        for (int i = 0; i < kSvnInfoCount; ++i)
            oss << kSvnInfo[i];
        oss << "----------------------------------------------------------\n";
    }

    oss << "BuildTime: " << kBuildTime << "\n"
        << "BuildType: " << kBuildType << "\n"
        << "BuilderName: " << kBuilderName << "\n"
        << "HostName: " << kHostName << "\n"
        << "Compiler: " << kCompiler << "\n";

    return oss.str();
}

void SetupBinaryVersion()
{
    google::SetVersionString(MakeVersionInfo());
}

} // namespace mytoft
} // namespace bubblefs
