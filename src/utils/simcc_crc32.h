
// simcc/simcc/misc/crc32.h

#ifndef BUBBLEFS_UTILS_SIMCC_CRC32_H_
#define BUBBLEFS_UTILS_SIMCC_CRC32_H_

#include "platform/types.h"
#include <stdint.h>
#include <string>

namespace bubblefs {
namespace mysimcc {

class CRC32 {
public:
    // @brief
    // @param d   Source data buffer, If length is length than dwLength, result is unknown.
    // @param len The size of d.
    // @return crc value.
    static uint32_t Sum(const void* d, size_t len);
    static uint32_t Sum(const string& s) {
        return Sum(s.data(), s.size());
    }

private:
    // Initialize the CRC table with 256 elements.
    static void InitTable(uint32_t* table);
    static uint32_t Reflect(uint32_t ref, char ch);
};

} // namespace mysimcc
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_SIMCC_CRC32_H_