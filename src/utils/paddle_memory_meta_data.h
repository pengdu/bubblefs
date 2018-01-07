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

// Paddle/paddle/memory/detail/meta_data.h
// Paddle/paddle/memory/detail/meta_data.cc

#pragma once

#include "utils/paddle_memory_block.h"

#include <stddef.h>
#include <functional>

namespace bubblefs {
namespace mypaddle {
namespace memory {
namespace detail {

class Metadata {
 public:
  Metadata(MemoryBlock::Type t, size_t i, size_t s, size_t ts, MemoryBlock* l,
           MemoryBlock* r);
  Metadata();

 public:
  /*! \brief Update the guards when metadata is changed */
  void update_guards();

  /*! \brief Check consistency to previous modification */
  bool check_guards() const;

 public:
  // TODO(gangliao): compress this
  // clang-format off
  size_t            guard_begin = 0;
  MemoryBlock::Type type        = MemoryBlock::INVALID_CHUNK;
  size_t            index       = 0;
  size_t            size        = 0;
  size_t            total_size  = 0;
  MemoryBlock*      left_buddy  = nullptr;
  MemoryBlock*      right_buddy = nullptr;
  size_t            guard_end   = 0;
  // clang-format on
};

Metadata::Metadata(MemoryBlock::Type t, size_t i, size_t s, size_t ts,
                   MemoryBlock* l, MemoryBlock* r)
    : type(t),
      index(i),
      size(s),
      total_size(ts),
      left_buddy(l),
      right_buddy(r) {}

Metadata::Metadata()
    : type(MemoryBlock::INVALID_CHUNK),
      index(0),
      size(0),
      total_size(0),
      left_buddy(nullptr),
      right_buddy(nullptr) {}

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

inline size_t hash(const Metadata* metadata, size_t initial_seed) {
  size_t seed = initial_seed;

  hash_combine(seed, (size_t)metadata->type);
  hash_combine(seed, metadata->index);
  hash_combine(seed, metadata->size);
  hash_combine(seed, metadata->total_size);
  hash_combine(seed, metadata->left_buddy);
  hash_combine(seed, metadata->right_buddy);

  return seed;
}

void Metadata::update_guards() {
  guard_begin = hash(this, 1);
  guard_end = hash(this, 2);
}

bool Metadata::check_guards() const {
  return guard_begin == hash(this, 1) && guard_end == hash(this, 2);
}

}  // namespace detail
}  // namespace memory
}  // namespace mypaddle
}  // namespace bubblefs