// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/storage/sharding/fingerprint_sharding.h

#ifndef BUBBLEFS_UTILS_TOFT_STORAGE_SHARDING_FINGER_SHARDING_H_
#define BUBBLEFS_UTILS_TOFT_STORAGE_SHARDING_FINGER_SHARDING_H_

#include <string>

#include "utils/toft_storage_sharding_sharding.h"

namespace bubblefs {
namespace mytoft {
class FingerprintSharding : public ShardingPolicy {
    DECLARE_UNCOPYABLE(FingerprintSharding);

public:
    FingerprintSharding();
    virtual ~FingerprintSharding();

    virtual int Shard(const std::string& key);
};
}  // namespace mytoft
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_TOFT_STORAGE_SHARDING_FINGER_SHARDING_H_