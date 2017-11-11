// Copyright (c) 2014 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// bitcoin/src/crypto/hmac_sha512.h

#ifndef BUBBLEFS_UTILS_BITCOIN_HMAC_SHA512_H_
#define BUBBLEFS_UTILS_BITCOIN_HMAC_SHA512_H_

#include <stdint.h>
#include <stdlib.h>
#include "utils/bitcoin_sha512.h"

namespace bubblefs {
namespace mybitcoin {

/** A hasher class for HMAC-SHA-512. */
class CHMAC_SHA512
{
private:
    CSHA512 outer;
    CSHA512 inner;

public:
    static const size_t OUTPUT_SIZE = 64;

    CHMAC_SHA512(const unsigned char* key, size_t keylen);
    CHMAC_SHA512& Write(const unsigned char* data, size_t len)
    {
        inner.Write(data, len);
        return *this;
    }
    void Finalize(unsigned char hash[OUTPUT_SIZE]);
};

} // namespace mybitcoin
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BITCOIN_HMAC_SHA512_H_