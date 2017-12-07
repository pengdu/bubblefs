// Copyright (c) 2010, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/text/wildcard.cpp

#include "utils/toft_text_wildcard.h"

#include <fnmatch.h>

namespace bubblefs {
namespace mytoft {

bool Wildcard::Match(const char* pattern, const char* string, int flags)
{
    // fnmatch was defined by ISO/IEC 9945-2: 1993 (POSIX.2)
    return ::fnmatch(pattern, string, flags) == 0;
}

} // namespace toft
} // namespace bubblefs