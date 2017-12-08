// Copyright (c) 2013, The Toft Authors.
// All rights reserved.
//
// Author: Ye Shunping <yeshunping@gmail.com>

// toft/crypto/uuid/uuid.cpp

#include "utils/toft_crypto_uuid_uuid.h"

#include <stdio.h>

#include "platform/base_error.h"
#include "utils/toft_base_string_algorithm.h"

namespace bubblefs {
namespace mytoft {

static const char uuidVersionRequired = '4';
static const int uuidVersionIdentifierIndex = 14;
static const char kUUIDFileName[] = "/proc/sys/kernel/random/uuid";

std::string CreateCanonicalUUIDString() {
    // This does not work for the linux system that turns on sandbox.
    FILE* fptr = fopen(kUUIDFileName, "r");
    if (!fptr) {
        PRINTF_ERROR("fail to open file: %s\n", kUUIDFileName);
        return std::string();
    }
    char uuidStr[37];
    char* result = fgets(uuidStr, sizeof(uuidStr), fptr);
    fclose(fptr);
    if (!result) {
        PRINTF_ERROR("fail to read uuid string from: %s\n", kUUIDFileName);
        return std::string();
    }
    std::string canonicalUuidStr(uuidStr);
    StringToLower(&canonicalUuidStr);
    assert(canonicalUuidStr[uuidVersionIdentifierIndex] == uuidVersionRequired);
    return canonicalUuidStr;
}

}  // namespace mytoft
}  // namespace bubblefs