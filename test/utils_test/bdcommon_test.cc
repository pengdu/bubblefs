#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>
#include "platform/atomicops.h"
#include "platform/mutexlock.h"
#include "platform/time.h"
#include "utils/counter.h"
#include "utils/raw_coding.h"
#include "utils/thread_simple.h"
#include "gtest/gtest.h"

// case 1
/*
namespace bubblefs {
namespace core {

TEST(UtilTest, TestEncodeDecode) {
    char buf1[8];
    char buf2[8];
    uint64_t x = 123456789;
    uint64_t y = 200000000;
    EncodeBigEndian(buf1, x);
    EncodeBigEndian(buf2, y);
    printf("%s %s\n", std::string(buf1, 8).c_str(), std::string(buf2, 8).c_str());
    ASSERT_TRUE(std::string(reinterpret_cast<char*>(&x), 8) > std::string(reinterpret_cast<char*>(&y), 8));
    ASSERT_TRUE(std::string(buf1, 8) < std::string(buf2, 8));
    ASSERT_EQ(DecodeBigEndian64(buf1), x);
    ASSERT_EQ(DecodeBigEndian64(buf2), y);

    uint32_t a = 123456789;
    uint32_t b = 200000000;
    char bufa[4];
    char bufb[4];
    EncodeBigEndian(bufa, a);
    EncodeBigEndian(bufb, b);
    printf("%s %s\n", std::string(bufa, 4).c_str(), std::string(bufb, 4).c_str());
    ASSERT_TRUE(std::string(reinterpret_cast<char*>(&a), 4) > std::string(reinterpret_cast<char*>(&b), 4));
    ASSERT_TRUE(std::string(bufa, 4) < std::string(bufb, 4));
    ASSERT_EQ(DecodeBigEndian32(bufa), a);
    ASSERT_EQ(DecodeBigEndian32(bufb), b);
}
  
}  // namespace core
}  // namespace bubblefs

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
*/

// case 2
/*
bubblefs::port::Mutex x;
std::mutex y;
void LockFunc() {
  //x.Lock();
  y.lock();
}

void UnlockFunc() {
  //x.Unlock(); // return EPREM
  y.unlock();
}

void LockUnlockFun() {
  x.Lock();
  sleep(3);
  x.Unlock();
}

int main(int argc, char* argv[]) {
    bubblefs::bdcommon::Thread t1, t2;
    t1.Start(LockFunc);
    t1.Join();
    t2.Start(UnlockFunc);
    t2.Join();
    printf("Done\n");
    return 0;
}
*/

// case 3
/*
const int kThreadNum = 8;

namespace bubblefs {
namespace bdcommon {

void* AddProc(void* arg) {
    Counter* counter = reinterpret_cast<Counter*>(arg);
    while (1) {
        counter->Inc();
    }
    return NULL;
}

void RunPerfTest() {
    Counter counter;
    Thread* threads = new Thread[kThreadNum];
    for (int i = 0; i < kThreadNum; i++) {
        threads[i].Start(AddProc, &counter);
    }

    long s_time = timeutil::get_micros();
    while (1) {
        usleep(1000000);
        long n_time = timeutil::get_micros();
        long count = counter.Clear();
        printf("%ld\n", count * 1000000 / (n_time - s_time)); // count == 40000000
        s_time = n_time;
    }
}

} // namespace bdcommon
} // namespace bubblefs

int main(int arbdcommongc, char* argv[]) {
    bubblefs::bdcommon::RunPerfTest();
    return 0;
}
*/

// case 4

int* ring __attribute__((aligned(64)));
int r_len __attribute__((aligned(64))) = 102400;
volatile long r_s __attribute__((aligned(64))) = 0;
volatile long r_e __attribute__((aligned(64))) = 0;

namespace bubblefs {
namespace bdcommon {

port::Mutex mu __attribute__((aligned(64)));
port::CondVar c(&mu);
Counter items __attribute__((aligned(64)));

//boost::lockfree::queue<int, boost::lockfree::fixed_sized<false> > queue(1024000);
Counter counter __attribute__((aligned(64)));

void Consumer() {
    //sleep(5);
    while(1) {
        counter.Inc();
        c.Signal();
    }
}

void Producer() {
    while (1) {
        while (items.Get() > 100000) {
            MutexLock lock(&mu);
            c.Wait();
        }
        items.Inc();
    }
}

int Run(int tnum) {
    Thread* t1 = new Thread[tnum];
    Thread* t2 = new Thread[tnum];
    printf("Thread num: %d\n", tnum);
    for (int i = 0; i < tnum; i++) t1[i].Start(Consumer);
    for (int i = 0; i < tnum; i++) t2[i].Start(Producer);
    while (1) {
        sleep(1);
        fprintf(stderr, "%ld %ld\n", counter.Clear(), items.Get()); // 2500000 100001
    }
    return 0;
}

} // bdcommon
} // bubblefs

int main(int argc, char* argv[]) {
    int tnum = 1;
    if (argc > 1) {
        tnum = atoi(argv[1]);
    }
    ring = new int[r_len];
    return bubblefs::bdcommon::Run(tnum);
}