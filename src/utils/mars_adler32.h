/* adler32.h -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2004 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

// mars/mars/comm/adler32.h

#ifndef BUBBLEFS_UTILS_MARS_ADLER32_H_
#define BUBBLEFS_UTILS_MARS_ADLER32_H_

#include <sys/types.h>

namespace bubblefs {
namespace mymars {
  
unsigned long adler32(unsigned long adler, const unsigned char* buf, unsigned int len);
unsigned long adler32_combine(unsigned long adler1, unsigned long adler2, unsigned long len2);

} // namespace mymars
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_MARS_ADLER32_H_