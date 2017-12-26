/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/include/LuceneAllocator.h

#ifndef BUBBLEFS_UTILS_LUCENE_ALLOCATOR_H_
#define BUBBLEFS_UTILS_LUCENE_ALLOCATOR_H_

#include <stddef.h>

namespace bubblefs {
namespace mylucene {

/// Allocate block of memory.
void* AllocMemory(size_t size);

/// Reallocate a given block of memory.
void* ReallocMemory(void* memory, size_t size);

/// Release a given block of memory.
void FreeMemory(void* memory);

} // namespace mylucene
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_LUCENE_ALLOCATOR_H_