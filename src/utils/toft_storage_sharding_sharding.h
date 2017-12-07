// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/storage/sharding/sharding.h

#ifndef BUBBLEFS_UTILS_STORAGE_SHARDING_SHARDING_H_
#define BUBBLEFS_UTILS_STORAGE_SHARDING_SHARDING_H_

#include "utils/toft_base_uncopyable.h"
#include "utils/toft_base_class_registry_class_registry.h"

namespace bubblefs {
namespace mytoft {
  
class ShardingPolicy {
 public:
  ShardingPolicy();
  virtual ~ShardingPolicy();

  virtual void SetShardingNumber(int shard_num) {
    shard_num_ = shard_num;
  }

  virtual int Shard(const std::string& key) = 0;

 protected:
  int shard_num_;

 private:
  DECLARE_UNCOPYABLE(ShardingPolicy);
};

MYTOFT_CLASS_REGISTRY_DEFINE(sharding_policy_registry, ShardingPolicy);

#define MYTOFT_REGISTER_SHARDING_POLICY(class_name) \
    MYTOFT_CLASS_REGISTRY_REGISTER_CLASS( \
        ::bubblefs::mytoft::sharding_policy_registry, \
        ::bubblefs::mytoft::ShardingPolicy, \
        #class_name, \
        class_name)
        
}  // namespace mytoft
}  // namespace bubblefs

#define MYTOFT_CREATE_SHARDING_POLICY(name_as_string) \
    MYTOFT_CLASS_REGISTRY_CREATE_OBJECT(::bubblefs::mytoft::sharding_policy_registry, name_as_string)

#endif  // BUBBLEFS_UTILS_STORAGE_SHARDING_SHARDING_H_