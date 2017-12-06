// Copyright (c) 2010, The TOFT Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/memory/unaligned/generic.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_MEMORY_UNALIGNED_GENERIC_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_MEMORY_UNALIGNED_GENERIC_H_

// generic solution, using memcpy

#include <string.h>

#include "platform/toft_system_memory_unaligned_check_direct_include.h"
#include "utils/toft_base_type_cast.h"

namespace bubblefs {
namespace mytoft {

template <typename T>
T GetUnaligned(const void* p)
{
    T t;
    memcpy(&t, p, sizeof(t));
    return t;
}

template <typename T, typename U>
void PutUnaligned(void* p, const U& value)
{
    T t = implicit_cast<T>(value);
    memcpy(p, &t, sizeof(t));
}

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_MEMORY_UNALIGNED_GENERIC_H_