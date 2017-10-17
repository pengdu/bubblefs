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
/*
 * Ceph - scalable distributed file system
 *
 * Copyright (C) 2004-2006 Sage Weil <sage@newdream.net>
 *
 * This is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1, as published by the Free Software 
 * Foundation.  See file COPYING.
 * 
 */
/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

// Paddle/paddle/utils/Util.h
// ceph/src/common/simple_cache.hpp
// apollo/modules/common/util/lru_cache.h

#ifndef BUBBLEFS_UTILS_WEAK_CACHE_H_
#define BUBBLEFS_UTILS_WEAK_CACHE_H_

#include <assert.h>
#include <stdint.h>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
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

template <class K, class V, class C = std::less<K>, class H = std::hash<K> >
class SimpleLRU {
  std::mutex lock;
  size_t max_size;
  std::unordered_map<K, typename std::list<std::pair<K, V> >::iterator, H> contents;
  std::list<std::pair<K, V> > lru;
  std::map<K, V, C> pinned;

  void trim_cache() {
    while (lru.size() > max_size) {
      contents.erase(lru.back().first);
      lru.pop_back();
    }
  }

  void _add(K key, V&& value) {
    lru.emplace_front(key, std::move(value)); // can't move key because we access it below
    contents[key] = lru.begin();
    trim_cache();
  }

public:
  SimpleLRU(size_t max_size) : lock(), max_size(max_size) {
    contents.rehash(max_size);
  }

  void pin(K key, V val) {
    std::lock_guard<std::mutex> l(lock);
    pinned.emplace(std::move(key), std::move(val));
  }

  void clear_pinned(K e) {
    std::lock_guard<std::mutex> l(lock);
    for (typename std::map<K, V, C>::iterator i = pinned.begin();
         i != pinned.end() && i->first <= e;
         pinned.erase(i++)) {
      typename std::unordered_map<K, typename std::list<std::pair<K, V> >::iterator, H>::iterator iter =
        contents.find(i->first);
      if (iter == contents.end())
        _add(i->first, std::move(i->second));
      else
        lru.splice(lru.begin(), lru, iter->second);
    }
  }

  void clear(K key) {
    std::lock_guard<std::mutex> l(lock);
    typename std::unordered_map<K, typename std::list<std::pair<K, V> >::iterator, H>::iterator i =
      contents.find(key);
    if (i == contents.end())
      return;
    lru.erase(i->second);
    contents.erase(i);
  }

  void set_size(size_t new_size) {
    std::lock_guard<std::mutex> l(lock);
    max_size = new_size;
    trim_cache();
  }

  bool lookup(K key, V *out) {
    std::lock_guard<std::mutex> l(lock);
    typename std::unordered_map<K, typename std::list<std::pair<K, V> >::iterator, H>::iterator i =
      contents.find(key);
    if (i != contents.end()) {
      *out = i->second->second;
      lru.splice(lru.begin(), lru, i->second);
      return true;
    }
    typename std::map<K, V, C>::iterator i_pinned = pinned.find(key);
    if (i_pinned != pinned.end()) {
      *out = i_pinned->second;
      return true;
    }
    return false;
  }

  void add(K key, V value) {
    std::lock_guard<std::mutex> l(lock);
    _add(std::move(key), std::move(value));
  }
};

// apollo/modules/common/util/lru_cache.h
template <class K, class V>
struct KvNode {
  K key;
  V val;
  KvNode* prev;
  KvNode* next;
  KvNode() : prev(nullptr), next(nullptr) {
    key = {};
    val = {};
  }

  template <typename VV>
  KvNode(const K& key, VV&& val)
      : key(key), val(std::forward<VV>(val)), prev(nullptr), next(nullptr) {}
};

template <class K, class V>
class SimpleLRUCache {
 public:
  SimpleLRUCache() : capacity_(kDefaultCapacity), map_(0), head_(), tail_() {
    Init();
  }

  explicit SimpleLRUCache(const size_t capacity)
      : capacity_(capacity), map_(0), head_(), tail_() {
    Init();
  }

  ~SimpleLRUCache() { Clear(); }

  void GetCache(std::unordered_map<K, V>* cache) {
    for (auto it = map_.begin(); it != map_.end(); ++it) {
      cache->operator[](it->first) = it->second.val;
    }
  }

  V& operator[](const K& key) {
    if (!Contains(key)) {
      K obsolete;
      GetObsolete(&obsolete);
    }
    return map_[key].val;
  }

  /*
   * Silently get all as vector
   */
  void GetAllSilently(std::vector<V*>* ret) {
    for (auto it = map_.begin(); it != map_.end(); ++it) {
      ret->push_back(&it->second.val);
    }
  }

  /*
   * for both add & update purposes
   */
  template <typename VV>
  bool Put(const K& key, VV&& val) {
    K tmp;
    return Update(key, std::forward<VV>(val), &tmp, false, false);
  }

  /*
   * update existing elements only
   */
  template <typename VV>
  bool Update(const K& key, VV&& val) {
    if (!Contains(key)) {
      return false;
    }
    K tmp;
    return Update(key, std::forward<VV>(val), &tmp, true, false);
  }

  /*
   * silently update existing elements only
   */
  template <typename VV>
  bool UpdateSilently(const K& key, VV* val) {
    if (!Contains(key)) {
      return false;
    }
    K tmp;
    return Update(key, std::forward<VV>(*val), &tmp, true, true);
  }

  /*
   * add new elements only
   */
  template <typename VV>
  bool Add(const K& key, VV* val) {
    K tmp;
    return Update(key, std::forward<VV>(*val), &tmp, true, false);
  }

  template <typename VV>
  bool PutAndGetObsolete(const K& key, VV* val, K* obs) {
    return Update(key, std::forward<VV>(*val), obs, false, false);
  }

  template <typename VV>
  bool AddAndGetObsolete(const K& key, VV* val, K* obs) {
    return Update(key, std::forward<VV>(*val), obs, true, false);
  }

  V* GetSilently(const K& key) { return Get(key, true); }

  V* Get(const K& key) { return Get(key, false); }

  bool GetCopySilently(const K& key, const V* val) {
    return GetCopy(key, val, true);
  }

  bool GetCopy(const K& key, const V* val) { return GetCopy(key, val, false); }

  size_t size() { return size_; }

  bool Full() { return size() > 0 && size() >= capacity_; }

  bool Empty() { return size() == 0; }

  size_t capacity() { return capacity_; }

  KvNode<K, V>* First() {
    if (size()) {
      return head_.next;
    }
    return nullptr;
  }

  bool Contains(const K& key) { return map_.find(key) != map_.end(); }

  bool Prioritize(const K& key) {
    if (Contains(key)) {
      auto* node = &map_[key];
      Detach(node);
      Attach(node);
      return true;
    }
    return false;
  }

  void Clear() {
    map_.clear();
    size_ = 0;
  }

 private:
  static constexpr size_t kDefaultCapacity = 10;

  const size_t capacity_;
  size_t size_;
  std::unordered_map<K, KvNode<K, V>> map_;
  KvNode<K, V> head_;
  KvNode<K, V> tail_;

  void Init() {
    head_.prev = nullptr;
    head_.next = &tail_;
    tail_.prev = &head_;
    tail_.next = nullptr;
    size_ = 0;
  }

  void Detach(KvNode<K, V>* node) {
    if (node->prev != nullptr) {
      node->prev->next = node->next;
    }
    if (node->next != nullptr) {
      node->next->prev = node->prev;
    }
    node->prev = nullptr;
    node->next = nullptr;
    --size_;
  }

  void Attach(KvNode<K, V>* node) {
    node->prev = &head_;
    node->next = head_.next;
    head_.next = node;
    if (node->next != nullptr) {
      node->next->prev = node;
    }
    ++size_;
  }

  template <typename VV>
  bool Update(const K& key, VV&& val, K* obs, bool add_only,
              bool silent_update) {
    if (obs == nullptr) {
      return false;
    }
    if (Contains(key)) {
      if (!add_only) {
        map_[key].val = std::forward<VV>(val);
        if (!silent_update) {
          auto* node = &map_[key];
          Detach(node);
          Attach(node);
        } else {
          return false;
        }
      }
    } else {
      if (Full() && !GetObsolete(obs)) {
        return false;
      }

      map_.emplace(key, KvNode<K, V>(key, std::forward<VV>(val)));
      Attach(&map_[key]);
    }
    return true;
  }

  V* Get(const K& key, bool silent) {
    if (Contains(key)) {
      auto* node = &map_[key];
      if (!silent) {
        Detach(node);
        Attach(node);
      }
      return &node->val;
    }
    return nullptr;
  }

  bool GetCopy(const K& key, const V* val, bool silent) {
    if (Contains(key)) {
      auto* node = &map_[key];
      if (!silent) {
        Detach(node);
        Attach(node);
      }
      *val = node->val;
      return true;
    }
    return false;
  }

  bool GetObsolete(K* key) {
    if (Full()) {
      auto* node = tail_.prev;
      Detach(node);
      *key = node->key;
      map_.erase(node->key);
      return true;
    }
    return false;
  }
};
  
} // namespace core
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_WEAK_CACHE_H_