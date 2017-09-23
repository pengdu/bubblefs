// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/guid.h

#ifndef BUBBLEFS_PLATFORM_GUID_H_
#define BUBBLEFS_PLATFORM_GUID_H_

#include <string>
#include "platform/base_export.h"
#include "platform/macros.h"
#include "platform/types.h"

namespace bubblefs {
namespace random {

// Generate a 128-bit random GUID of the form: "%08X-%04X-%04X-%04X-%012llX".
// If GUID generation fails an empty string is returned.
// The POSIX implementation uses psuedo random number generation to create
// the GUID.  The Windows implementation uses system services.
BASE_EXPORT std::string GenerateGUID();

// Returns true if the input string conforms to the GUID format.
BASE_EXPORT bool IsValidGUID(const std::string& guid);

#if defined(OS_POSIX)
// For unit testing purposes only.  Do not use outside of tests.
BASE_EXPORT std::string RandomDataToGUIDString(const uint64_t bytes[2]);
#endif

}  // namespace random
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_GUID_H_