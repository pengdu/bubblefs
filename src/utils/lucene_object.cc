/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2009-2014 Alan Wright. All rights reserved.
// Distributable under the terms of either the Apache License (Version 2.0)
// or the GNU Lesser General Public License.
/////////////////////////////////////////////////////////////////////////////

// LucenePlusPlus/src/core/util/LuceneObject.cpp

#include "utils/lucene_object.h"
#include <sstream>

namespace bubblefs {
namespace mylucene {

/// Convert any given type to a {@link String}.
template <class TYPE>
static string toString(const TYPE& value) {
 std::stringstream os;
  os << value;
  return os.str();
}  
  
Object::Object() {
}

Object::~Object() {
}

void Object::initialize() {
    // override
}

ObjectPtr Object::clone(const ObjectPtr& other) {
    if (!other) {
        return nullptr;
    }
    return other;
}

int32_t Object::hashCode() {
    return (int32_t)(int64_t)this;
}

bool Object::equals(const ObjectPtr& other) {
    return (other && this == other.get());
}

int32_t Object::compareTo(const ObjectPtr& other) {
    return (int32_t)(this - other.get());
}

string Object::toString() {
    return "object@" + mylucene::toString(hashCode());
}

} // namespace mylucene
} // namespace bubblefs