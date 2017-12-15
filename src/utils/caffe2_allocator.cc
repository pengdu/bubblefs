/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/core/allocator.cc

#include "utils/caffe2_allocator.h"

namespace bubblefs {
namespace mycaffe2 {

bool caffe2_report_cpu_memory_usage = false; // "If set, print out detailed memory usage");

bool caffe2_cpu_allocator_do_zero_fill = true; // "If set, do memory zerofilling when allocating on CPU");
  
void NoDelete(void*) {}

static std::unique_ptr<CPUAllocator> g_cpu_allocator(new DefaultCPUAllocator());
CPUAllocator* GetCPUAllocator() {
  return g_cpu_allocator.get();
}

void SetCPUAllocator(CPUAllocator* alloc) {
  g_cpu_allocator.reset(alloc);
}

//MemoryAllocationReporter CPUContext::reporter_;

void MemoryAllocationReporter::New(void* ptr, size_t nbytes) {
  std::lock_guard<std::mutex> guard(mutex_);
  size_table_[ptr] = nbytes;
  allocated_ += nbytes;
  //LOG(INFO) << "Caffe2 alloc " << nbytes << " bytes, total alloc " << allocated_
  //          << " bytes.";
}

void MemoryAllocationReporter::Delete(void* ptr) {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = size_table_.find(ptr);
  PANIC_ENFORCE(it != size_table_.end(), "size_table_.find(ptr) fail");
  allocated_ -= it->second;
  //LOG(INFO) << "Caffe2 deleted " << it->second << " bytes, total alloc "
  //          << allocated_ << " bytes.";
  size_table_.erase(it);
}

} // namespace mycaffe2
} // namespace bubblefs