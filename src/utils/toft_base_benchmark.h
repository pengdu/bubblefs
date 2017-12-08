// Copyright (c) 2013, The Toft Authors. All rights reserved.
// Author: Ye Shunping <yeshunping@gmail.com>
//  Notes: Idea got from benchmark framework of re2

// toft/base/benchmark.h

#ifndef BUBBLEFS_UTILS_TOFT_BASE_BENCHMARK_H_
#define BUBBLEFS_UTILS_TOFT_BASE_BENCHMARK_H_

#include <stdint.h>

namespace bubblefs {
namespace mytoft {
  
struct Benchmark {
    const char* name;
    void (*fn)(int a);
    void (*fnr)(int a, int b);
    int lo;
    int hi;
    int threadlo;
    int threadhi;

    void Register();
    Benchmark(const char* name, void (*f)(int)) {  // NOLINT
        Clear(name);
        fn = f;
        Register();
    }
    Benchmark(const char* name, void (*f)(int, int), int l, int h) {  // NOLINT
        Clear(name);
        fnr = f;
        lo = l;
        hi = h;
        Register();
    }
    void Clear(const char* n) {
        name = n;
        fn = 0;
        fnr = 0;
        lo = 0;
        hi = 0;
        threadlo = 0;
        threadhi = 0;
    }
    Benchmark* ThreadRange(int lo, int hi) {
        threadlo = lo;
        threadhi = hi;
        return this;
    }
};

void SetBenchmarkBytesProcessed(int64_t bytes_processed);
void StopBenchmarkTiming();
void StartBenchmarkTiming();
void BenchmarkMemoryUsage();
void SetBenchmarkItemsProcessed(int n);

void RunBench(Benchmark* b, int nthread, int siz);

extern int nbenchmarks;
extern Benchmark* benchmarks[10000];

}  // namespace mytoft
}  // namespace bubblefs

//  It's implemented in file: thirdparty/gperftools-2.0/src/base/sysinfo.cc
extern int NumCPUs();

#define MYTOFT_BENCHMARK(f) \
    ::bubblefs::mytoft::Benchmark* _benchmark_##f = (new ::bubblefs::mytoft::Benchmark(#f, f))

#define MYTOFT_BENCHMARK_RANGE(f, lo, hi) \
    :::bubblefs::mytoft::Benchmark* _benchmark_##f = \
    (new :::bubblefs::mytoft::Benchmark(#f, f, lo, hi))

#endif  // BUBBLEFS_UTILS_TOFT_BASE_BENCHMARK_H_