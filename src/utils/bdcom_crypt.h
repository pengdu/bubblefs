// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/utils/crypt.h

#ifndef BUBBLEFS_UTILS_BDCOM_CRYPT_H
#define BUBBLEFS_UTILS_BDCOM_CRYPT_H

#include <stdint.h>

#include <cstdio>
#include <string>

namespace buubblefs {
namespace mybdcom {

// return 0: all is ok, result(hash number) stored at the location given by @result;
// otherwise: invalid arguments.
int32_t GetHashNumber(const std::string& str, uint32_t seed, uint32_t* result);

// #define HASH_STRING_LEN 8
int32_t GetHashString(const std::string& str, uint32_t seed, std::string* result);

}  // namespace mybdcom
}  // namespace buubblefs

#endif // BUBBLEFS_UTILS_BDCOM_CRYPT_H