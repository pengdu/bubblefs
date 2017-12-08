// Copyright (c) 2013, The Toft Authors. All rights reserved.
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/hash/hash_benchmark.cpp

#include "utils/toft_hash_hash.h"
#include "utils/toft_base_benchmark.h"

const std::string test_str = "qwertyuiopasdfghjklmnbvcz";

static void CityHash32(int n) {
    for (int i = 0; i < n; i++) {
        bubblefs::mytoft::CityHash32(test_str);
    }
}

static void CityHash64(int n) {
    for (int i = 0; i < n; i++) {
        bubblefs::mytoft::CityHash64(test_str);
    }
}

static void CityHash128(int n) {
    for (int i = 0; i < n; i++) {
        bubblefs::mytoft::CityHash128(test_str);
    }
}

static void JenkinsOneAtATimeHash(int n) {
    for (int i = 0; i < n; i++) {
        bubblefs::mytoft::JenkinsOneAtATimeHash(test_str);
    }
}

static void SuperFastHash(int n) {
    for (int i = 0; i < n; i++) {
        bubblefs::mytoft::SuperFastHash(test_str);
    }
}

static void MurmurHash64A(int n) {
    for (int i = 0; i < n; i++) {
        bubblefs::mytoft::MurmurHash64A(test_str, 0);
    }
}

static void MurmurHash64B(int n) {
    for (int i = 0; i < n; i++) {
        bubblefs::mytoft::MurmurHash64B(test_str.data(), test_str.size(), 0);
    }
}

static void CRC32(int n) {
    for (int i = 0; i < n; i++) {
        bubblefs::mytoft::CRC32(test_str);
    }
}

MYTOFT_BENCHMARK(CityHash32)->ThreadRange(1, NumCPUs());
MYTOFT_BENCHMARK(CityHash64)->ThreadRange(1, NumCPUs());
MYTOFT_BENCHMARK(CityHash128)->ThreadRange(1, NumCPUs());
MYTOFT_BENCHMARK(JenkinsOneAtATimeHash)->ThreadRange(1, NumCPUs());
MYTOFT_BENCHMARK(SuperFastHash)->ThreadRange(1, NumCPUs());
MYTOFT_BENCHMARK(MurmurHash64A)->ThreadRange(1, NumCPUs());
MYTOFT_BENCHMARK(MurmurHash64B)->ThreadRange(1, NumCPUs());
MYTOFT_BENCHMARK(CRC32)->ThreadRange(1, NumCPUs());