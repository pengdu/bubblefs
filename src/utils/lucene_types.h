/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/include/LuceneTypes.h

#ifndef BUBBLEFS_UTILS_LUCENE_TYPES_H_
#define BUBBLEFS_UTILS_LUCENE_TYPES_H_

#include <memory>
#include "platform/types.h"

#define DECLARE_SHARED_PTR(Type) \
    class Type; \
    typedef std::shared_ptr<Type> Type##Ptr; \
    typedef std::weak_ptr<Type> Type##WeakPtr;
    
namespace bubblefs {
namespace mylucene {

DECLARE_SHARED_PTR(Object)
DECLARE_SHARED_PTR(Signal)
DECLARE_SHARED_PTR(Synchronize)

} // namespace mylucene
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_LUCENE_TYPES_H_