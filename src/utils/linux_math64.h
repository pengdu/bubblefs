/*-
 * Copyright (c) 2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2014-2015 Mellanox Technologies, Ltd. All rights reserved.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice unmodified, this list of conditions, and the following
 *    disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * $FreeBSD$
 */

// freebsd/sys/compat/linuxkpi/common/include/linux/math64.h

#ifndef BUBBLEFS_UTILS_LINUX_MATH64_H_
#define BUBBLEFS_UTILS_LINUX_MATH64_H_

#include <stdint.h>

namespace bubblefs {
namespace mylinux {
  
constexpr int MYLINUX_DIV_ROUND_UP(int x, int y) { return (((x)+((y)-1)) / (y)); }  
  
#define mylinux_do_div(n, base) ({                      \
        uint32_t __base = (base);               \
        uint32_t __rem;                         \
        __rem = ((uint64_t)(n)) % __base;       \
        (n) = ((uint64_t)(n)) / __base;         \
        __rem;                                  \
})

inline uint64_t
div64_u64_rem(uint64_t dividend, uint64_t divisor, uint64_t *remainder)
{

        *remainder = dividend % divisor;
        return (dividend / divisor);
}

inline int64_t
div64_s64(int64_t dividend, int64_t divisor)
{

        return (dividend / divisor);
}

inline uint64_t
div64_u64(uint64_t dividend, uint64_t divisor)
{

        return (dividend / divisor);
}

inline uint64_t
div_u64_rem(uint64_t dividend, uint32_t divisor, uint32_t *remainder)
{

        *remainder = dividend % divisor;
        return (dividend / divisor);
}

inline int64_t
div_s64(int64_t dividend, int32_t divisor)
{

        return (dividend / divisor);
}

inline uint64_t
div_u64(uint64_t dividend, uint32_t divisor)
{

        return (dividend / divisor);
}

}  // namespace mylinux
}  // namespace bubblefs

#endif // BUBBLEFS_UTILS_LINUX_MATH64_H_