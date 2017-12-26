/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/include/LuceneObject.h
// LucenePlusPlus/include/LuceneSync.h

#ifndef BUBBLEFS_UTILS_LUCENE_OBJECT_H_
#define BUBBLEFS_UTILS_LUCENE_OBJECT_H_

#include <memory>
#include <string>
#include "utils/lucene.h"
#include "utils/lucene_sync.h"

#define LUCENE_INTERFACE(Name) \
    static std::string _getClassName() { return #Name; } \
    virtual std::string getClassName() { return #Name; }

#define LUCENE_CLASS(Name) \
    LUCENE_INTERFACE(Name); \
    std::shared_ptr<Name> shared_from_this() { return std::static_pointer_cast<Name>(Object::shared_from_this()); }

namespace bubblefs {
namespace mylucene {
  
/// Base class for all Lucene classes
class Object : public ObjectSync, public std::enable_shared_from_this<Object> {
public:
    virtual ~Object();

protected:
    Object();

public:
    /// Called directly after instantiation to create objects that depend on this object being
    /// fully constructed.
    virtual void initialize();

    /// Return clone of this object
    /// @param other clone reference - null when called initially, then set in top virtual override.
    virtual ObjectPtr clone(const ObjectPtr& other = ObjectPtr());

    /// Return hash code for this object.
    virtual int32_t hashCode();

    /// Return whether two objects are equal
    virtual bool equals(const ObjectPtr& other);

    /// Compare two objects
    virtual int32_t compareTo(const ObjectPtr& other);

    /// Returns a string representation of the object
    virtual string toString();
};

} // namespace mylucene
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_LUCENE_OBJECT_H_