// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/encoding/hex.cpp

#include "utils/toft_encoding_hex.h"

namespace bubblefs {
namespace mytoft {

char* Hex::EncodeToBuffer(
    const void* data, size_t size,
    char* output,
    bool uppercase)
{
    const unsigned char* p = static_cast<const unsigned char*>(data);
    Encode(p, p + size, output, uppercase);
    output[2 * size] = '\0';
    return output;
}

std::string Hex::EncodeAsString(
    const void* data, size_t size,
    bool uppercase)
{
    std::string str;
    EncodeTo(data, size, &str, uppercase);
    return str;
}

} // namespace mytoft
} // namespace bubblefs