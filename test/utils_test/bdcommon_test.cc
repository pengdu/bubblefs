
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <vector>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "libco/co_closure.h"
#include "platform/atomicops.h"
#include "platform/logging_simple.h"
#include "platform/mutexlock.h"
#include "platform/time.h"
#include "tbb/concurrent_hash_map.h"
#include "utils/counter.h"
#include "utils/raw_coding.h"
#include "utils/stringpiece.h"
#include "utils/thread_simple.h"
#include "utils/threadpool_simple.h"

// case 1
/*
namespace bubblefs {
namespace core {

TEST(UtilTest, TestEncodeDecode) { // [ RUN ] UtilTest.TestEncodeDecode
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
} // // [ OK ] UtilTest.TestEncodeDecode (ms)
  
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
/*
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
*/

// case 5
/*
namespace bubblefs {
namespace bdcommon {

class ThreadTest : public ::testing::Test {
public:
    ThreadTest() : task_done_(&task_mu_) {}
    void Task() {
        MutexLock l(&task_mu_);
        task_done_.Signal();
    }

protected:
    port::CondVar task_done_;
    mutable port::Mutex task_mu_;
};

TEST_F(ThreadTest, Start) { // ThreadTest.Start
    Thread t;
    MutexLock l(&task_mu_);
    t.Start(std::bind(&ThreadTest::Task, this));
    bool ret = task_done_.IntervalWait(1000);
    ASSERT_TRUE(ret);
}

} // namespace bdcommon
} // namespace bubblefs

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
*/

// case 6
/*
const int kThreadNum = 8;
const int kMaxPending = 10000;

namespace bubblefs {
namespace bdcommon {

void Task() {
}

void* AddTask(void* arg) {
    ThreadPool* tp = reinterpret_cast<ThreadPool*>(arg);
    while (1) {
        while (tp->PendingNum() > kMaxPending) {
        }
        tp->AddTask(Task);
    }
    return NULL;
}

void RunPerfTest() {
    ThreadPool tp;
    Thread* threads = new Thread[kThreadNum];
    for (int i = 0; i < kThreadNum; i++) {
        threads[i].Start(AddTask, &tp);
    }
    while (1) {
        usleep(1000000);
        std::string plog = tp.ProfilingLog();
        long pending = tp.PendingNum();
        printf("%ld %s\n", pending, plog.c_str());
    }
}

} // namespace bdcommon
} // namespace bubblefs

int main(int argc, char* argv[]) {
    bubblefs::bdcommon::RunPerfTest();
    return 0;
}
*/

// case 7
/*
namespace bubblefs {
namespace bdcommon {

port::Mutex mu;
port::CondVar p(&mu);
port::CondVar c(&mu);
Counter items;
Counter counter;

void Consumer() {
    MutexLock lock(&mu);
    while(1) {
        while(items.Get() == 0) {
            p.Wait();
        }
        //printf("Consume\n");
        items.Clear();
        counter.Inc();
        c.Signal();
    }
}

void Producer() {
    MutexLock lock(&mu);
    while (1) {
        while (items.Get() > 0) {
            c.Wait();
        }
        //printf("Produce\n");
        items.Inc();
        p.Signal();
    }
}

int Run() {
    bubblefs::bdcommon::Thread t1,t2;
    t1.Start(Consumer);
    t2.Start(Producer);
    while (1) {
        sleep(1);
        fprintf(stderr, "%ld\n", counter.Clear()); // 150000
    }
    return 0;
}
}
}

int main() {
    return bubblefs::bdcommon::Run();
}
*/

// case 8
/*
int main(int argc, char* argv[]) {
    const char* char_pointer = "char*";
    bubblefs::StringPiece string = "std;";
    BDLOGS(bubblefs::bdcommon::INFO) << 88 << " " << char_pointer << " " << string;
    BDLOGS(bubblefs::bdcommon::INFO) << 88 << " " << char_pointer << " " << string;

    return 0;
}
*/

// case 9
/*
using namespace std;
DEFINE_string(confPath, "../conf/setup.ini", "program configure file.");
DEFINE_int32(port, 9090, "program listen port");
DEFINE_bool(daemon, true, "run daemon mode");
 
int main(int argc, char** argv)
{
  google::ParseCommandLineFlags(&argc, &argv, true);
 
  cout << "confPath = " << FLAGS_confPath << endl;
  cout << "port = " << FLAGS_port << endl;
 
  if (FLAGS_daemon) {
    cout << "run background ..." << endl;
  }
  else {
    cout << "run foreground ..." << endl;
  }
 
  cout << "good luck and good bye!" << endl;
 
  google::ShutDownCommandLineFlags();
  return 0;
}
*/

// case 10
/*
using namespace tbb;
using namespace std;

typedef concurrent_hash_map<long int, string> data_hash;

int main(){
  data_hash dh;
  data_hash::accessor a;

  long int k = 10;
  dh.insert(a,k);
  a->second = "hello";
  for (data_hash::iterator j = dh.begin();j != dh.end(); ++j){
    printf("%lu %s\n", j->first, j->second.c_str());
  }
  if (dh.find(a,9)){
    printf("true\n");
  } else {
    printf("false\n");      
  }
  a.release();
  return 0;
}
*/

// case 11
/*
int main(int argc,char* argv[])
{
    google::InitGoogleLogging(argv[0]);  // 初始化 glog
    google::ParseCommandLineFlags(&argc, &argv, true);  // 初始化 gflags
    LOG(INFO) << "Hello, GOOGLE!";  // INFO 级别的日志
    LOG(ERROR) << "ERROR, GOOGLE!";  // ERROR 级别的日志
    return 0;
}
*/

// case 12

using namespace std;

static void *thread_func( void * arg )
{
        stCoClosure_t *p = (stCoClosure_t*) arg;
        p->exec();
        return 0;
}
static void batch_exec( vector<stCoClosure_t*> &v )
{
        vector<pthread_t> ths;
        for( size_t i=0;i<v.size();i++ )
        {
                pthread_t tid;
                pthread_create( &tid,0,thread_func,v[i] );
                ths.push_back( tid );
        }
        for( size_t i=0;i<v.size();i++ )
        {
                pthread_join( ths[i],0 );
        }
}
int main( int argc,char *argv[] )
{
        vector< stCoClosure_t* > v;

        pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

        int total = 100;
        vector<int> v2;
        co_ref( ref,total,v2,m);
        for(int i=0;i<10;i++)
        {
                co_func( f,ref,i )
                {
                        printf("ref.total %d i %d\n",ref.total,i );
                        //lock
                        pthread_mutex_lock(&ref.m);
                        ref.v2.push_back( i );
                        pthread_mutex_unlock(&ref.m);
                        //unlock
                }
                co_func_end;
                v.push_back( new f( ref,i ) );
        }
        for(int i=0;i<2;i++)
        {
                co_func( f2,i )
                {
                        printf("i: %d\n",i);
                        for(int j=0;j<2;j++)
                        {
                                usleep( 1000 );
                                printf("i %d j %d\n",i,j);
                        }
                }
                co_func_end;
                v.push_back( new f2( i ) );
        }

        batch_exec( v );
        printf("done\n");

        return 0;
}