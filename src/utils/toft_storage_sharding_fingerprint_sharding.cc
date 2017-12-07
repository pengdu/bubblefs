// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/storage/sharding/fingerprint_sharding.cc

#include "utils/toft_storage_sharding_fingerprint_sharding.h"

#include "utils/toft_hash_fingerprint.h"

namespace bubblefs {
namespace mytoft {

FingerprintSharding::FingerprintSharding() {
}

FingerprintSharding::~FingerprintSharding() {
}

int FingerprintSharding::Shard(const std::string& key) {
    int shard_id = Fingerprint64(key) % (shard_num_);
    return shard_id;
}

MYTOFT_REGISTER_SHARDING_POLICY(FingerprintSharding);

}  // namespace mytoft
}  // namespace bubblefs
