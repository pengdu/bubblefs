// Modifications copyright (C) 2017, Baidu.com, Inc.
// Copyright 2017 The Apache Software Foundation

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

// be/src/common/object_pool.h

#ifndef BUBBLEFS_UTILS_BDCOMMON_OBJECT_POOL_H_
#define BUBBLEFS_UTILS_BDCOMMON_OBJECT_POOL_H_

#include <functional>
#include <mutex>
#include <vector>
#include "platform/mutexlock.h"

namespace bubblefs {
namespace bdcommon {
  
// An ObjectPool maintains a list of C++ objects which are deallocated
// by destroying the pool.
// Thread-safe.
class ObjectPool {
public:
    ObjectPool(): _objects() {}

    ~ObjectPool() {
        clear();
    }

    template <class T>
    T* add(T* t) {
        // Create the object to be pushed to the shared vector outside the critical section.
        // TODO: Consider using a lock-free structure.
        SpecificElement<T>* obj = new SpecificElement<T>(t);
        std::lock_guard<SpinMutex> l(_lock);
        _objects.push_back(obj);
        return t;
    }

    void clear() {
        std::lock_guard<SpinMutex> l(_lock);
        for (auto i = _objects.rbegin(); i != _objects.rend(); ++i) {
            delete *i;
        }
        _objects.clear();
    }

private:
    struct GenericElement {
        virtual ~GenericElement() {}
    };

    template <class T>
    struct SpecificElement : GenericElement {
        SpecificElement(T* t): t(t) {}
        ~SpecificElement() {
            delete t;
        }

        T* t;
    };

    typedef std::vector<GenericElement*> ElementVector;
    ElementVector _objects;
    SpinMutex _lock;
};

} // namespce bdcommon
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOMMON_OBJECT_POOL_H_