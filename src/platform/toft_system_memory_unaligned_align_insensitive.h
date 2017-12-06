// Copyright (c) 2010, The TOFT Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/memory/unaligned/align_insensitive.h

#ifndef BUBBLESF_PLATFORM_TOFT_SYSTEM_MEMORY_UNALIGNED_ALIGN_INSENSITIVE_H_
#define BUBBLESF_PLATFORM_TOFT_SYSTEM_MEMORY_UNALIGNED_ALIGN_INSENSITIVE_H_

// internal header, no inclusion guard needed

#include "platform/toft_system_memory_unaligned_check_direct_include.h"
#include "utils/toft_base_type_cast.h"

namespace bubblefs {
namespace mytoft {

// align insensitive archs

template <typename T>
T GetUnaligned(const void* p)
{
    return *static_cast<const T*>(p);
}

// introduce U make T must be given explicitly
template <typename T, typename U>
void PutUnaligned(void* p, const U& value)
{
    *static_cast<T*>(p) = implicit_cast<T>(value);
}

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLESF_PLATFORM_TOFT_SYSTEM_MEMORY_UNALIGNED_ALIGN_INSENSITIVE_H_