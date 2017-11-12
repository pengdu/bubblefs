// Copyright (c) 2016 Mirants Lu. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// saber/saber/util/crc32c.h

#ifndef BUBBLEFS_UTILS_SABER_CRC32C_H_
#define BUBBLEFS_UTILS_SABER_CRC32C_H_

#include <stddef.h>
#include <stdint.h>

namespace bubblefs {
namespace mysaber {
namespace crc {

extern uint32_t crc32(uint32_t crc, const char* buf, size_t size);

}  // namespace crc
}  // namespace mysaber
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_SABER_CRC32C_H_