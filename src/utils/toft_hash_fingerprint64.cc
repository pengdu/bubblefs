// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/hash/fingerprint64.cpp

#include "utils/toft_hash_fingerprint.h"
#include "utils/toft_hash_murmur.h"

namespace bubblefs {
namespace mytoft {

namespace {
static const uint64_t kFingerPrintSeed = 19820125;
}

uint64_t Fingerprint64(const std::string& str) {
    return MurmurHash64A(str, kFingerPrintSeed);
}

}  // namespace mytoft
}  // namespace bubblefs