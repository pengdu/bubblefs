//---------------------------------------------------------
// For conditions of distribution and use, see
// https://github.com/preshing/cpp11-on-multicore/blob/master/LICENSE
//---------------------------------------------------------

// cpp11-on-multicore/tests/basetests/main.cpp

#include <chrono>
#include <iostream>

namespace bubblefs {
namespace rdsn {

//---------------------------------------------------------
// List of tests
//---------------------------------------------------------
struct TestInfo
{
    const char* name;
    bool (*testFunc)();
};

bool testBenaphore();
bool testRecursiveBenaphore();
bool testAutoResetEvent();
bool testRWLock();
bool testRWLockSimple();
bool testDiningPhilosophers();

#define ADD_TEST(name) { #name, name },
TestInfo g_tests[] =
{
    ADD_TEST(testBenaphore)
    ADD_TEST(testRecursiveBenaphore)
    ADD_TEST(testAutoResetEvent)
    ADD_TEST(testRWLock)
    ADD_TEST(testRWLockSimple)
    ADD_TEST(testDiningPhilosophers)
};

} // namespace rdsn
} // namespace bubblefs

//---------------------------------------------------------
// main
//---------------------------------------------------------
int main()
{
    bool allTestsPassed = true;

    for (const bubblefs::rdsn::TestInfo& test : bubblefs::rdsn::g_tests)
    {
        std::cout << "Running " << test.name << "...";

        auto start = std::chrono::high_resolution_clock::now();
        bool result = test.testFunc();
        auto end = std::chrono::high_resolution_clock::now();

        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << " " << (result ? "passed" : "***FAILED***") << " in " << millis << " ms\n";
        allTestsPassed = allTestsPassed && result;
    }

    return allTestsPassed ? 0 : 1;
}