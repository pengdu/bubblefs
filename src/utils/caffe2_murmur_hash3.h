//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.

// caffe2/caffe2/utils/murmur_hash3.h

#ifndef BUBBLEFS_UTILS_CAFFE2_MURMUR_HASH3_H_
#define BUBBLEFS_UTILS_CAFFE2_MURMUR_HASH3_H_

#include <stdint.h>

namespace bubblefs {
namespace caffe2 {

void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out);

void MurmurHash3_x86_128(const void* key, int len, uint32_t seed, void* out);

void MurmurHash3_x64_128(const void* key, int len, uint32_t seed, void* out);

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_MURMUR_HASH3_H_