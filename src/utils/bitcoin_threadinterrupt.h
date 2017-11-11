// Copyright (c) 2016 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// bitcoin/src/threadinterrupt.h

#ifndef BUBBLEFS_UTILS_BITCOIN_THREADINTERRUPT_H_
#define BUBBLEFS_UTILS_BITCOIN_THREADINTERRUPT_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace bubblefs {
namespace mybitcoin {
  
/*
    A helper class for interruptible sleeps. Calling operator() will interrupt
    any current sleep, and after that point operator bool() will return true
    until reset.
*/
class CThreadInterrupt
{
public:
    explicit operator bool() const;
    void operator()();
    void reset();
    bool sleep_for(std::chrono::milliseconds rel_time);
    bool sleep_for(std::chrono::seconds rel_time);
    bool sleep_for(std::chrono::minutes rel_time);

private:
    std::condition_variable cond;
    std::mutex mut;
    std::atomic<bool> flag;
};

} // namespace mybitcoin
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BITCOIN_THREADINTERRUPT_H_