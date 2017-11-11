// Copyright (c) 2014-2016 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// bitcoin/src/crypto/sha512.h

#ifndef BUBBLEFS_UTILS_BITCOIN_SHA512_H_
#define BUBBLEFS_UTILS_BITCOIN_SHA512_H_

#include <stdint.h>
#include <stdlib.h>

namespace bubblefs {
namespace mybitcoin {

/** A hasher class for SHA-512. */
class CSHA512
{
private:
    uint64_t s[8];
    unsigned char buf[128];
    uint64_t bytes;

public:
    static const size_t OUTPUT_SIZE = 64;

    CSHA512();
    CSHA512& Write(const unsigned char* data, size_t len);
    void Finalize(unsigned char hash[OUTPUT_SIZE]);
    CSHA512& Reset();
};

} // namespace mybitcoin
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BITCOIN_SHA512_H_