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

#ifndef BUBBLEFS_UTILS_WEAK_CACHE_H_
#define BUBBLEFS_UTILS_WEAK_CACHE_H_

#include <assert.h>
#include <stdint.h>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "platform/macros.h"

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

// palo/be/src/util/lru_cache.hpp
template <typename Key, typename Value>
class WeakLruCache {
public:
    typedef typename std::pair<Key, Value> KeyValuePair;
    typedef typename std::list<KeyValuePair>::iterator ListIterator;

    class Iterator : public std::iterator<std::input_iterator_tag, KeyValuePair> {
    public:
        Iterator(typename std::unordered_map<Key, ListIterator>::iterator it) : _it(it) { }

        Iterator& operator++() {
            ++_it;
            return *this;
        }

        bool operator==(const Iterator& rhs) const {
            return _it == rhs._it;
        }

        bool operator!=(const Iterator& rhs) const {
            return _it != rhs._it;
        }

        KeyValuePair* operator->() {
            return _it->second.operator->();
        }

        KeyValuePair& operator*() {
            return *_it->second;
        }

    private:
        typename std::unordered_map<Key, ListIterator>::iterator _it;
    };

    WeakLruCache(size_t max_size) : _max_size(max_size) { }

    void put(const Key& key, const Value& value) {
        auto it = _cache_items_map.find(key);
        if (it != _cache_items_map.end()) {
            _cache_items_list.erase(it->second);
            _cache_items_map.erase(it);
        }

        _cache_items_list.push_front(KeyValuePair(key, value));
        _cache_items_map[key] = _cache_items_list.begin();

        if (_cache_items_map.size() > _max_size) {
            auto last = _cache_items_list.end();
            last--;
            _cache_items_map.erase(last->first);
            _cache_items_list.pop_back();
        }
    }

    void erase(const Key& key) {
        auto it = _cache_items_map.find(key);
        if (it != _cache_items_map.end()) {
            _cache_items_list.erase(it->second);
            _cache_items_map.erase(it);
        }
    }

    // Must copy value, because value maybe relased when caller used
    bool get(const Key& key, Value* value) {
        auto it = _cache_items_map.find(key);
        if (it == _cache_items_map.end()) {
            return false;
        }
        _cache_items_list.splice(_cache_items_list.begin(), _cache_items_list, it->second);
        *value = it->second->second;
        return true;
    }

    bool exists(const Key& key) const {
        return _cache_items_map.find(key) != _cache_items_map.end();
    }

    size_t size() const {
        return _cache_items_map.size();
    }

    Iterator begin() {
        return Iterator(_cache_items_map.begin());
    }

    Iterator end() {
        return Iterator(_cache_items_map.end());
    }

private:
    std::list<KeyValuePair> _cache_items_list;
    std::unordered_map<Key, ListIterator> _cache_items_map;
    size_t _max_size;
};  
  
} // namespace core
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_WEAK_CACHE_H_