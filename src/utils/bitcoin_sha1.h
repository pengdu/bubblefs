// Copyright (c) 2014-2016 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// bitcoin/src/crypto/sha1.h

#ifndef BUBBLEFS_UTILS_BITCOIN_SHA1_H_
#define BUBBLEFS_UTILS_BITCOIN_SHA1_H_

#include <stdint.h>
#include <stdlib.h>

namespace bubblefs {
namespace bitcoin {
  
/** A hasher class for SHA1. */
class CSHA1
{
private:
    uint32_t s[5];
    unsigned char buf[64];
    uint64_t bytes;

public:
    static const size_t OUTPUT_SIZE = 20;

    CSHA1();
    CSHA1& Write(const unsigned char* data, size_t len);
    void Finalize(unsigned char hash[OUTPUT_SIZE]);
    CSHA1& Reset();
};

} // namespace bitcoin
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BITCOIN_SHA1_H_