// Copyright (c) 2014-2016 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// bitcoin/src/crypto/sha256.h

#ifndef BUBBLEFS_UTILS_BITCOIN_SHA256_H_
#define BUBBLEFS_UTILS_BITCOIN_SHA256_H_

#include <stdint.h>
#include <stdlib.h>
#include <string>

namespace bubblefs {
namespace bitcoin {

/** A hasher class for SHA-256. */
class CSHA256
{
private:
    uint32_t s[8];
    unsigned char buf[64];
    uint64_t bytes;

public:
    static const size_t OUTPUT_SIZE = 32;

    CSHA256();
    CSHA256& Write(const unsigned char* data, size_t len);
    void Finalize(unsigned char hash[OUTPUT_SIZE]);
    CSHA256& Reset();
};

/** Autodetect the best available SHA256 implementation.
 *  Returns the name of the implementation.
 */
std::string SHA256AutoDetect();

} // namespace bitcoin
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BITCOIN_SHA256_H_