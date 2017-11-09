// Copyright (c) 2015-2016 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// bitcoin/src/reverselock.h

#ifndef BUBBLEFS_UTILS_BITCOIN_REVERSELOCK_H_
#define BUBBLEFS_UTILS_BITCOIN_REVERSELOCK_H_

namespace bubblefs {
namespace bitcoin {

/**
 * An RAII-style reverse lock. Unlocks on construction and locks on destruction.
 */
template<typename Lock>
class reverse_lock
{
public:

    explicit reverse_lock(Lock& _lock) : lock(_lock) {
        _lock.unlock();
        _lock.swap(templock);
    }

    ~reverse_lock() {
        templock.lock();
        templock.swap(lock);
    }

private:
    reverse_lock(reverse_lock const&);
    reverse_lock& operator=(reverse_lock const&);

    Lock& lock;
    Lock templock;
};

} // namespace bitcoin
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BITCOIN_REVERSELOCK_H_