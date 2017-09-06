// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// A Cache is an interface that maps keys to values.  It has internal
// synchronization and may be safely accessed concurrently from
// multiple threads.  It may automatically evict entries to make room
// for new entries.  Values have a specified charge against the cache
// capacity.  For example, a cache where the values are variable
// length strings, may use the length of the string as the charge for
// the string.
//
// A builtin cache implementation with a least-recently-used eviction
// policy is provided.  Clients may use their own implementations if
// they want something more sophisticated (like scan-resistance, a
// custom eviction policy, variable cache sizing, etc.)

/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Paddle/paddle/utils/Util.h
// baidu/common/include/cache.h

#ifndef BUBBLEFS_UTILS_CACHE_H_
#define BUBBLEFS_UTILS_CACHE_H_

#include <assert.h>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "utils/stringpiece.h"

namespace bubblefs {
namespace core {

/**
 * Key-Value Cache Helper.
 *
 * It store a object instance global. User can invoke get method by key and a
 * object creator callback. If there is a instance stored in cache, then it will
 * return a shared_ptr of it, otherwise, it will invoke creator callback, create
 * a new instance store global, and return it.
 *
 * The cache instance will release when nobody hold a reference to it.
 *
 * The KType is the key type.
 * The VType is the value type.
 * The Hash is the key hasher object.
 */
template <typename KType, typename VType, typename Hash>
class WeakKVCache {
public:
  WeakKVCache() {}

  std::shared_ptr<VType> get(const KType& key,
                             const std::function<VType*()>& creator) {
    std::lock_guard<std::mutex> guard(this->lock_);
    auto it = this->storage_.find(key);
    if (it != this->storage_.end()) {
      auto& val = it->second;
      auto retVal = val.lock();
      if (retVal != nullptr) {
        return retVal;
      }  // else fall trough. Because it is WeakPtr Cache.
    }
    auto rawPtr = creator();
    assert(rawPtr != nullptr);
    std::shared_ptr<VType> retVal(rawPtr);
    this->storage_[key] = retVal;
    return retVal;
  }

private:
  std::mutex lock_;
  std::unordered_map<KType, std::weak_ptr<VType>, Hash> storage_;
};

class Cache;

// Create a new cache with a fixed size capacity.  This implementation
// of Cache uses a least-recently-used eviction policy.
extern Cache* NewLRUCache(size_t capacity);

class Cache {
public:
    Cache() { }

    // Destroys all existing entries by calling the "deleter"
    // function that was passed to the constructor.
    virtual ~Cache();

    // Opaque handle to an entry stored in the cache.
    struct Handle { };

    // Insert a mapping from key->value into the cache and assign it
    // the specified charge against the total cache capacity.
    //
    // Returns a handle that corresponds to the mapping.  The caller
    // must call this->Release(handle) when the returned mapping is no
    // longer needed.
    //
    // When the inserted entry is no longer needed, the key and
    // value will be passed to "deleter".
    virtual Handle* Insert(const StringPiece& key, void* value, size_t charge,
                           void (*deleter)(const StringPiece& key, void* value)) = 0;

    // If the cache has no mapping for "key", returns NULL.
    //
    // Else return a handle that corresponds to the mapping.  The caller
    // must call this->Release(handle) when the returned mapping is no
    // longer needed.
    virtual Handle* Lookup(const StringPiece& key) = 0;

    // Release a mapping returned by a previous Lookup().
    // REQUIRES: handle must not have been released yet.
    // REQUIRES: handle must have been returned by a method on *this.
    virtual void Release(Handle* handle) = 0;

    // Return the value encapsulated in a handle returned by a
    // successful Lookup().
    // REQUIRES: handle must not have been released yet.
    // REQUIRES: handle must have been returned by a method on *this.
    virtual void* Value(Handle* handle) = 0;

    // If the cache contains entry for key, erase it.  Note that the
    // underlying entry will be kept around until all existing handles
    // to it have been released.
    virtual void Erase(const StringPiece& key) = 0;

    // Return a new numeric id.  May be used by multiple clients who are
    // sharing the same cache to partition the key space.  Typically the
    // client will allocate a new id at startup and prepend the id to
    // its cache keys.
    virtual uint64_t NewId() = 0;

private:
    void LRU_Remove(Handle* e);
    void LRU_Append(Handle* e);
    void Unref(Handle* e);

    struct Rep;
    Rep* rep_;

    // No copying allowed
    Cache(const Cache&);
    void operator=(const Cache&);
};

} // namespace core
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CACHE_H_