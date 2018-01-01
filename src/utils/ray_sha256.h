/*********************************************************************
* Filename:   sha256.h
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Defines the API for the corresponding SHA1 implementation.
*********************************************************************/

// ray/src/common/thirdparty/sha256.h

#ifndef BUBBLEFS_UTILS_RAY_SHA256_H_
#define BUBBLEFS_UTILS_RAY_SHA256_H_

/*************************** HEADER FILES ***************************/
#include <stddef.h>

namespace bubblefs {
namespace myray {
  
/****************************** MACROS ******************************/
static const int SHA256_BLOCK_SIZE = 32;            // SHA256 outputs a 32 byte digest

/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte
typedef unsigned int  WORD;             // 32-bit word, change to "long" for 16-bit machines

typedef struct {
        BYTE data[64];
        WORD datalen;
        unsigned long long bitlen;
        WORD state[8];
} SHA256_CTX;

/*********************** FUNCTION DECLARATIONS **********************/
void sha256_init(SHA256_CTX *ctx);
void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);
void sha256_final(SHA256_CTX *ctx, BYTE hash[]);

} // namespace myray
} // namespace bubblefs

#endif   // BUBBLEFS_UTILS_RAY_SHA256_H_