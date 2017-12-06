// Copyright (c) 2013, The Toft Authors. All rights reserved.
// Author: Ye Shunping <yeshunping@gmail.com>

#include <algorithm>
#include "utils/toft_base_benchmark.h"

#include "gflags/gflags.h"

extern int bubblefs::mytoft::nbenchmarks;
extern bubblefs::mytoft::Benchmark* bubblefs::mytoft::benchmarks[];

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, false);
    for (int i = 0; i < bubblefs::mytoft::nbenchmarks; i++) {
        bubblefs::mytoft::Benchmark* b = bubblefs::mytoft::benchmarks[i];
        for (int j = b->threadlo; j <= b->threadhi; j++)
            for (int k = std::max(b->lo, 1); k <= std::max(b->hi, 1); k <<= 1)
                RunBench(b, j, k); // the compiler will use the first namespace which declares the RunBench function
    }
}
