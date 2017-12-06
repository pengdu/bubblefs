// Copyright (c) 2012, The TOFT Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/memory/unaligned.h

#ifndef BUBBLEFS_PLATFORM_TOFT_SYSTEM_MEMORY_UNALIGNED_H_
#define BUBBLEFS_PLATFORM_TOFT_SYSTEM_MEMORY_UNALIGNED_H_

#include <stddef.h>

namespace bubblefs {
namespace mytoft {

//////////////////////////////////////////////////////////////////////////////
// interface declaration

/// @brief get value from unaligned address
/// @tparam T type to get
/// @param p pointer to get value from
/// @return get result
/// @details usage: uint32_t n = GetUnaligned<uint32_t>(p);
template <typename T>
T GetUnaligned(const void* p);

/// @brief put value into unaligned address
/// @tparam T type to get
/// @tparam U introduce U make T must be given explicitly
/// @param p pointer to get value
/// @param value value to put into p
/// @details usage: PutUnaligned<uint32_t>(p, 100);
template <typename T, typename U>
void PutUnaligned(void* p, const U& value);

} // namespace toft
} // namespace bubblefs

//////////////////////////////////////////////////////////////////////////////
// implementation

/// known alignment insensitive platforms
#if defined(__i386__) || \
    defined(__x86_64__) || \
    defined(_M_IX86) || \
    defined(_M_X64)
#define MYTOFT_ALIGNMENT_INSENSITIVE_PLATFORM 1
#endif

#if defined MYTOFT_ALIGNMENT_INSENSITIVE_PLATFORM
# include "platform/toft_system_memory_unaligned_align_insensitive.h"
#else
#  include "toft/system/memory/unaligned/generic.h"
#endif // arch detect

namespace bubblefs {
namespace mytoft {

/// @brief round up pointer to next nearest aligned address
/// @param p the pointer
/// @param align alignment, must be power if 2
template <typename T>
T* RoundUpPtr(T* p, size_t align)
{
    size_t address = reinterpret_cast<size_t>(p);
    return reinterpret_cast<T*>((address + align - 1) & ~(align - 1U));
}

/// @brief round down pointer to previous nearest aligned address
/// @param p the pointer
/// @param align alignment, must be power if 2
template <typename T>
T* RoundDownPtr(T* p, size_t align)
{
    size_t address = reinterpret_cast<size_t>(p);
    return reinterpret_cast<T*>(address & ~(align - 1U));
}

} // namespace mytoft
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_TOFT_SYSTEM_MEMORY_UNALIGNED_H_