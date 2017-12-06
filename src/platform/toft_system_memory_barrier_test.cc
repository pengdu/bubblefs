// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/memory/barrier_test.cpp

#include "platform/toft_system_memory_barrier.h"

#include "gtest/gtest.h"

namespace bubblefs {
namespace mytoft {

TEST(MemoryBarrier, Test)
{
    MemoryBarrier();
    MemoryReadBarrier();
    MemoryWriteBarrier();
}

} // namespace mytoft
} // namespace bubblefs