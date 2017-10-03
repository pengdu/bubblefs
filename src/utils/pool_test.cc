/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tensorflow/tensorflow/compiler/xla/service/pool_test.cc

#include "utils/object_pool.h"
#include "platform/test.h"

namespace bubblefs {
  
namespace core {
namespace {

using PoolTest = ::testing::Test;

TEST_F(PoolTest, Test) {
  ObjectPool<int> pool;

  {
    auto ptr = pool.Allocate();
    EXPECT_NE(nullptr, ptr.get());
    *ptr = 5;
  }

  auto ptr = pool.Allocate();
  EXPECT_NE(nullptr, ptr.get());
  EXPECT_EQ(5, *ptr);
}

}  // namespace
}  // namespace core

}  // namespace bubblefs