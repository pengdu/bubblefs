// Copyright (c) 2013, The Toft Authors. All rights reserved.
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/crypto/uuid/uuid_benchmark.cpp

#include "utils/toft_base_benchmark.h"
#include "utils/toft_crypto_uuid_uuid.h"
#include "platform/base_error.h"


static void CreateCanonicalUUIDString(int n) {
    for (int i = 0; i < n; i++) {
        std::string uuid = bubblefs::mytoft::CreateCanonicalUUIDString();
        PRINTF_INFO("%s\n", uuid.c_str());
    }
}

MYTOFT_BENCHMARK(CreateCanonicalUUIDString)->ThreadRange(1, NumCPUs());
