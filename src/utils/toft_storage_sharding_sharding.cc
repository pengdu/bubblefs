// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/storage/sharding/sharding.cc

#include "utils/toft_storage_sharding_sharding.h"

namespace bubblefs {
namespace mytoft {

ShardingPolicy::ShardingPolicy() : shard_num_(1) {
}

ShardingPolicy::~ShardingPolicy() {
}

}  // namespace mytoft
}  // namespace bubblefs