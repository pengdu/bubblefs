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

// Paddle/paddle/math/PoolAllocator.cc

#include "utils/paddle_pool_allocator.h"

namespace bubblefs {
namespace mypaddle {

PoolAllocator::PoolAllocator(Allocator* allocator,
                             size_t sizeLimit,
                             const std::string& name)
    : allocator_(allocator),
      sizeLimit_(sizeLimit),
      poolMemorySize_(0),
      name_(name) {}

PoolAllocator::~PoolAllocator() { freeAll(); }

void* PoolAllocator::alloc(size_t size) {
  if (sizeLimit_ > 0) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = pool_.find(size);
    if (it == pool_.end() || it->second.size() == 0) {
      if (poolMemorySize_ >= sizeLimit_) {
        freeAll();
      }
      return allocator_->alloc(size);
    } else {
      auto buf = it->second.back();
      it->second.pop_back();
      poolMemorySize_ -= size;
      return buf;
    }
  } else {
    return allocator_->alloc(size);
  }
}

void PoolAllocator::free(void* ptr, size_t size) {
  if (sizeLimit_ > 0) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto& it = pool_[size];
    it.push_back(ptr);
    poolMemorySize_ += size;
  } else {
    allocator_->free(ptr);
  }
}

void PoolAllocator::freeAll() {
  for (auto it : pool_) {
    for (auto ptr : it.second) {
      allocator_->free(ptr);
    }
  }
  poolMemorySize_ = 0;
  pool_.clear();
}

void PoolAllocator::printAll() {
  size_t memory = 0;
  PRINTF_INFO("%s:", name_.c_str());
  for (auto it : pool_) {
    PRINTF_INFO("  size:%zu", it.first);
    for (auto ptr : it.second) {
      PRINTF_INFO("    ptr:%p", ptr);
      memory += it.first;
    }
  }
  PRINTF_INFO("memory size: %zu\n", memory);
}

}  // namespace mypaddle
}  // namespace bubblefs