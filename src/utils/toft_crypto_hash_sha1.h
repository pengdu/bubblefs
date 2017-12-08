// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/crypto/hash/sha1.h

#ifndef BUBBLEFS_UTILS_TOFT_CRYPTO_HASH_SHA1_H_
#define BUBBLEFS_UTILS_TOFT_CRYPTO_HASH_SHA1_H_

#include <stdint.h>
#include <string>

#include "utils/toft_base_string_string_piece.h"
#include "utils/toft_base_uncopyable.h"

namespace bubblefs {
namespace mytoft {

class SHA1 {
    DECLARE_UNCOPYABLE(SHA1);

public:
    SHA1();
    ~SHA1();

    //  Init is called in constructor, but if you want to use the same object
    //  for many times, you SHOULD call Init before computing sha1 of new data.
    void Init();
    void Update(StringPiece sp);
    // Finalizes the Sha1 operation and fills the buffer with the digest.
    //  Data is uint8_t digest_[20]
    void Final(void* digest);
    //  Hex encoding for result
    std::string HexFinal();

    static std::string HexDigest(StringPiece sp);

private:
    void SHA1Transform(uint32_t state[5], const uint8_t buffer[64]);
    void Update(const uint8_t* data, size_t input_len);
    void FinalInternal();

private:
    struct SHA1_CTX {
        uint32_t state[5];
        uint32_t count[2];  // Bit count of input.
        uint8_t buffer[64];
    };

    SHA1_CTX context_;
    //uint8_t digest_[20];
};

}  // namespace mytoft
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_TOFT_CRYPTO_HASH_SHA1_H_