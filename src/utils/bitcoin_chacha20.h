// Copyright (c) 2017 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// bitcoin/src/crypto/chacha20.h

#ifndef BUBBLEFS_UTILS_BITCOIN_CHACHA20_H_
#define BUBBLEFS_UTILS_BITCOIN_CHACHA20_H_

#include <stdint.h>
#include <stdlib.h>

namespace bubblefs {
namespace mybitcoin {

/** A PRNG class for ChaCha20. */
class ChaCha20
{
private:
    uint32_t input[16];

public:
    ChaCha20();
    ChaCha20(const unsigned char* key, size_t keylen);
    void SetKey(const unsigned char* key, size_t keylen);
    void SetIV(uint64_t iv);
    void Seek(uint64_t pos);
    void Output(unsigned char* output, size_t bytes);
};

} // namespace mybitcoin
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BITCOIN_CHACHA20_H_