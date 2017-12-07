//  GLOBLA_NOLINT(legal/copyright)
//-----------------------------------------------------------------------------

// toft/hash/murmur/MurmurHash2.cpp
// toft/hash/murmur/MurmurHash2A.cpp
// toft/hash/murmur/MurmurHash2_64.cpp
// toft/hash/murmur/MurmurHash3.cpp
// toft/hash/murmur/MurmurHashAligned2.cpp
// toft/hash/murmur/MurmurHashNeutral2.cpp

#include "utils/toft_hash_murmur.h"

#include <stdlib.h> // for _rotl

namespace bubblefs {
namespace mytoft {

#ifdef __GNUC__
// implement _rotl rotl64 of MSC
inline uint32_t _rotl(uint32_t n, int c)
{
    return (uint32_t) ((n << c) | (n >> (32 - c)));
}

inline uint64_t _rotl64(uint64_t n, int c)
{
    return (uint64_t) ((n << c) | (n >> (64 - c)));
}
#endif  
  
//-----------------------------------------------------------------------------
// MurmurHash2, by Austin Appleby

// Note - This code makes a few assumptions about how your machine behaves -

// 1. We can read a 4-byte value from any address without crashing
// 2. sizeof(int) == 4

// And it has a few limitations -

// 1. It will not work incrementally.
// 2. It will not produce the same results on little-endian and big-endian
//    machines.  
  
uint32_t MurmurHash2(const void * key, size_t len, uint32_t seed)
{
    // 'm' and 'r' are mixing constants generated offline.
    // They're not really 'magic', they just happen to work well.

    const uint32_t m = 0x5bd1e995;
    const int r = 24;

    // Initialize the hash to a 'random' value

    uint32_t h = seed ^ len;

    // Mix 4 bytes at a time into the hash

    const unsigned char * data = (const unsigned char *)key;

    while (len >= 4)
    {
        uint32_t k = *(uint32_t*)data;  // NOLINT

        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;

        data += 4;
        len -= 4;
    }

    // Handle the last few bytes of the input array

    switch (len)
    {
    case 3:
        h ^= data[2] << 16;
    case 2:
        h ^= data[1] << 8;
    case 1:
        h ^= data[0];
        h *= m;
    };

    // Do a few final mixes of the hash to ensure the last few
    // bytes are well-incorporated.

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}

//-----------------------------------------------------------------------------
// MurmurHash2A, by Austin Appleby

// This is a variant of MurmurHash2 modified to use the Merkle-Damgard
// construction. Bulk speed should be identical to Murmur2, small-key speed
// will be 10%-20% slower due to the added overhead at the end of the hash.

// This variant fixes a minor issue where null keys were more likely to
// collide with each other than expected, and also makes the algorithm
// more amenable to incremental implementations. All other caveats from
// MurmurHash2 still apply

#define MIX(h, k, m) { k *= m; k ^= k >> r; k *= m; h *= m; h ^= k; }  // NOLINT(whitespace/newline)

uint32_t MurmurHash2A(const void * key, size_t len, uint32_t seed)
{
    const uint32_t m = 0x5bd1e995;
    const int r = 24;
    uint32_t l = len;

    const unsigned char * data = (const unsigned char *)key;

    uint32_t h = seed;

    while (len >= 4)
    {
        uint32_t k = *(uint32_t*)data;  // NOLINT

        MIX(h, k, m);

        data += 4;
        len -= 4;
    }

    uint32_t t = 0;

    switch (len)
    {
    case 3:
        t ^= data[2] << 16;
    case 2:
        t ^= data[1] << 8;
    case 1:
        t ^= data[0];
    };

    MIX(h, t, m);
    MIX(h, l, m);

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}

//-----------------------------------------------------------------------------
// MurmurHash2, 64-bit versions, by Austin Appleby

// The same caveats as 32-bit MurmurHash2 apply here - beware of alignment
// and endian-ness issues if used across multiple platforms.

// 64-bit hash for 64-bit platforms
uint64_t MurmurHash64A(const void * key, size_t len, uint64_t seed)
{
    const uint64_t m = 0xc6a4a7935bd1e995ULL;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t * data = (const uint64_t *)key;
    const uint64_t * end = data + (len/8);

    while (data != end)
    {
        uint64_t k = *data++;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const unsigned char * data2 = (const unsigned char*)data;

    switch (len & 7)
    {
    case 7:
        h ^= uint64_t(data2[6]) << 48;
    case 6:
        h ^= uint64_t(data2[5]) << 40;
    case 5:
        h ^= uint64_t(data2[4]) << 32;
    case 4:
        h ^= uint64_t(data2[3]) << 24;
    case 3:
        h ^= uint64_t(data2[2]) << 16;
    case 2:
        h ^= uint64_t(data2[1]) << 8;
    case 1:
        h ^= uint64_t(data2[0]);
        h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

uint64_t MurmurHash64A(const std::string& buffer, uint64_t seed)
{
    return MurmurHash64A(buffer.data(), buffer.size(), seed);
}

// 64-bit hash for 32-bit platforms
uint64_t MurmurHash64B(const void * key, size_t len, uint64_t seed)
{
    const uint32_t m = 0x5bd1e995;
    const int r = 24;

    uint32_t h1 = static_cast<uint32_t>(seed ^ len);
    uint32_t h2 = 0;

    const uint32_t * data = (const uint32_t *)key;

    while (len >= 8)
    {
        uint32_t k1 = *data++;
        k1 *= m;
        k1 ^= k1 >> r;
        k1 *= m;
        h1 *= m;
        h1 ^= k1;
        len -= 4;

        uint32_t k2 = *data++;
        k2 *= m;
        k2 ^= k2 >> r;
        k2 *= m;
        h2 *= m;
        h2 ^= k2;
        len -= 4;
    }

    if (len >= 4)
    {
        uint32_t k1 = *data++;
        k1 *= m;
        k1 ^= k1 >> r;
        k1 *= m;
        h1 *= m;
        h1 ^= k1;
        len -= 4;
    }

    switch (len)
    {
    case 3:
        h2 ^= ((unsigned char*)data)[2] << 16;
    case 2:
        h2 ^= ((unsigned char*)data)[1] << 8;
    case 1:
        h2 ^= ((unsigned char*)data)[0];
        h2 *= m;
    };

    h1 ^= h2 >> 18;
    h1 *= m;
    h2 ^= h1 >> 22;
    h2 *= m;
    h1 ^= h2 >> 17;
    h1 *= m;
    h2 ^= h1 >> 19;
    h2 *= m;

    uint64_t h = h1;

    h = (h << 32) | h2;

    return h;
}

//-----------------------------------------------------------------------------
// murmurhash3
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

inline uint32_t getblock(const uint32_t * p, int i)
{
    return p[i];
}

//----------
// Finalization mix - force all bits of a hash block to avalanche

// avalanches all bits to within 0.25% bias

inline uint32_t fmix32(uint32_t h)
{
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

inline void bmix32(uint32_t & h1, uint32_t & k1, uint32_t & c1, uint32_t & c2)
{
    k1 *= c1;
    k1  = _rotl(k1, 11);
    k1 *= c2;
    h1 ^= k1;

    h1 = h1*3+0x52dce729;

    c1 = c1*5+0x7b7d159c;
    c2 = c2*5+0x6bce6396;
}

//----------

void MurmurHash3_x86_32(const void * key, int len, uint32_t seed, void * out)
{
    const uint8_t * data = (const uint8_t*)key;
    const int nblocks = len / 4;

    uint32_t h1 = 0x971e137b ^ seed;

    uint32_t c1 = 0x95543787;
    uint32_t c2 = 0x2ad7eb25;

    //----------
    // body

    const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);

    for (int i = -nblocks; i; i++)
    {
        uint32_t k1 = getblock(blocks, i);
        bmix32(h1, k1, c1, c2);
    }

    //----------
    // tail

    const uint8_t * tail = (const uint8_t*)(data + nblocks*4);

    uint32_t k1 = 0;

    switch (len & 3)
    {
    case 3: k1 ^= tail[2] << 16;
    case 2: k1 ^= tail[1] << 8;
    case 1: k1 ^= tail[0];
            bmix32(h1, k1, c1, c2);
    };

    //----------
    // finalization

    h1 ^= len;

    h1 = fmix32(h1);

    *(uint32_t*)out = h1;
}

//-----------------------------------------------------------------------------

inline void bmix32(uint32_t & h1, uint32_t & h2, uint32_t & k1, uint32_t & k2,
                   uint32_t & c1, uint32_t & c2)
{
    k1 *= c1;
    k1  = _rotl(k1, 11);
    k1 *= c2;
    h1 ^= k1;
    h1 += h2;

    h2 = _rotl(h2, 17);

    k2 *= c2;
    k2  = _rotl(k2, 11);
    k2 *= c1;
    h2 ^= k2;
    h2 += h1;

    h1 = h1*3+0x52dce729;
    h2 = h2*3+0x38495ab5;

    c1 = c1*5+0x7b7d159c;
    c2 = c2*5+0x6bce6396;
}

//----------

void MurmurHash3_x86_64(const void * key, const int len, const uint32_t seed, void * out)
{
    const uint8_t * data = (const uint8_t*)key;
    const int nblocks = len / 8;

    uint32_t h1 = 0x8de1c3ac ^ seed;
    uint32_t h2 = 0xbab98226 ^ seed;

    uint32_t c1 = 0x95543787;
    uint32_t c2 = 0x2ad7eb25;

    //----------
    // body

    const uint32_t * blocks = (const uint32_t *)(data + nblocks*8);

    for (int i = -nblocks; i; i++)
    {
        uint32_t k1 = getblock(blocks, i * 2 + 0);
        uint32_t k2 = getblock(blocks, i * 2 + 1);
        bmix32(h1, h2, k1, k2, c1, c2);
    }

    //----------
    // tail

    const uint8_t * tail = (const uint8_t*)(data + nblocks*8);

    uint32_t k1 = 0;
    uint32_t k2 = 0;

    switch (len & 7)
    {
    case 7: k2 ^= tail[6] << 16;
    case 6: k2 ^= tail[5] << 8;
    case 5: k2 ^= tail[4] << 0;
    case 4: k1 ^= tail[3] << 24;
    case 3: k1 ^= tail[2] << 16;
    case 2: k1 ^= tail[1] << 8;
    case 1: k1 ^= tail[0] << 0;
            bmix32(h1, h2, k1, k2, c1, c2);
    };

    //----------
    // finalization

    h2 ^= len;

    h1 += h2;
    h2 += h1;

    h1 = fmix32(h1);
    h2 = fmix32(h2);

    h1 += h2;
    h2 += h1;

    ((uint32_t*)out)[0] = h1;
    ((uint32_t*)out)[1] = h2;
}

//-----------------------------------------------------------------------------
// This mix is large enough that VC++ refuses to inline it unless we use
// __forceinline. It's also not all that fast due to register spillage.

inline void bmix32(uint32_t & h1, uint32_t & h2, uint32_t & h3, uint32_t & h4,
                   uint32_t & k1, uint32_t & k2, uint32_t & k3, uint32_t & k4,
                   uint32_t & c1, uint32_t & c2)
{
    k1 *= c1;
    k1  = _rotl(k1, 11);
    k1 *= c2;
    h1 ^= k1;
    h1 += h2;
    h1 += h3;
    h1 += h4;

    h1 = _rotl(h1, 17);

    k2 *= c2;
    k2  = _rotl(k2, 11);
    k2 *= c1;
    h2 ^= k2;
    h2 += h1;

    h1 = h1 * 3 + 0x52dce729;
    h2 = h2 * 3 + 0x38495ab5;

    c1 = c1 * 5 + 0x7b7d159c;
    c2 = c2 * 5 + 0x6bce6396;

    k3 *= c1;
    k3  = _rotl(k3, 11);
    k3 *= c2;
    h3 ^= k3;
    h3 += h1;

    k4 *= c2;
    k4  = _rotl(k4, 11);
    k4 *= c1;
    h4 ^= k4;
    h4 += h1;

    h3 = h3*3+0x52dce729;
    h4 = h4*3+0x38495ab5;

    c1 = c1*5+0x7b7d159c;
    c2 = c2*5+0x6bce6396;
}

//----------

void MurmurHash3_x86_128(const void * key, const int len, const uint32_t seed, void * out)
{
    const uint8_t * data = (const uint8_t*)key;
    const int nblocks = len / 16;

    uint32_t h1 = 0x8de1c3ac ^ seed;
    uint32_t h2 = 0xbab98226 ^ seed;
    uint32_t h3 = 0xfcba5b2d ^ seed;
    uint32_t h4 = 0x32452e3e ^ seed;

    uint32_t c1 = 0x95543787;
    uint32_t c2 = 0x2ad7eb25;

    //----------
    // body

    const uint32_t * blocks = (const uint32_t *)(data);

    for (int i = 0; i < nblocks; i++)
    {
        uint32_t k1 = getblock(blocks, i * 4 + 0);
        uint32_t k2 = getblock(blocks, i * 4 + 1);
        uint32_t k3 = getblock(blocks, i * 4 + 2);
        uint32_t k4 = getblock(blocks, i * 4 + 3);

        bmix32(h1, h2, h3, h4, k1, k2, k3, k4, c1, c2);
    }

    //----------
    // tail

    const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

    uint32_t k1 = 0;
    uint32_t k2 = 0;
    uint32_t k3 = 0;
    uint32_t k4 = 0;

    switch (len & 15)
    {
    case 15: k4 ^= tail[14] << 16;
    case 14: k4 ^= tail[13] << 8;
    case 13: k4 ^= tail[12] << 0;
    case 12: k3 ^= tail[11] << 24;
    case 11: k3 ^= tail[10] << 16;
    case 10: k3 ^= tail[ 9] << 8;
    case  9: k3 ^= tail[ 8] << 0;
    case  8: k2 ^= tail[ 7] << 24;
    case  7: k2 ^= tail[ 6] << 16;
    case  6: k2 ^= tail[ 5] << 8;
    case  5: k2 ^= tail[ 4] << 0;
    case  4: k1 ^= tail[ 3] << 24;
    case  3: k1 ^= tail[ 2] << 16;
    case  2: k1 ^= tail[ 1] << 8;
    case  1: k1 ^= tail[ 0] << 0;
             bmix32(h1, h2, h3, h4, k1, k2, k3, k4, c1, c2);
    };

    //----------
    // finalization

    h4 ^= len;
    h1 += h2;
    h1 += h3;
    h1 += h4;
    h2 += h1;
    h3 += h1;
    h4 += h1;

    h1 = fmix32(h1);
    h2 = fmix32(h2);
    h3 = fmix32(h3);
    h4 = fmix32(h4);

    h1 += h2;
    h1 += h3;
    h1 += h4;
    h2 += h1;
    h3 += h1;
    h4 += h1;

    ((uint32_t*)out)[0] = h1;
    ((uint32_t*)out)[1] = h2;
    ((uint32_t*)out)[2] = h3;
    ((uint32_t*)out)[3] = h4;
}

//-----------------------------------------------------------------------------
// Block read - if your platform needs to do endian-swapping or can only
// handle aligned reads, do the conversion here

inline uint64_t getblock(const uint64_t * p, int i)
{
    return p[i];
}

//----------
// Block mix - combine the key bits with the hash bits and scramble everything

inline void bmix64(uint64_t & h1, uint64_t & h2, uint64_t & k1, uint64_t & k2,
                   uint64_t & c1, uint64_t & c2)
{
    k1 *= c1;
    k1  = _rotl64(k1, 23);
    k1 *= c2;
    h1 ^= k1;
    h1 += h2;

    h2 = _rotl64(h2, 41);

    k2 *= c2;
    k2  = _rotl64(k2, 23);
    k2 *= c1;
    h2 ^= k2;
    h2 += h1;

    h1 = h1*3+0x52dce729;
    h2 = h2*3+0x38495ab5;

    c1 = c1*5+0x7b7d159c;
    c2 = c2*5+0x6bce6396;
}

//----------
// Finalization mix - avalanches all bits to within 0.05% bias

inline uint64_t fmix64(uint64_t k)
{
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;

    return k;
}

//----------

void MurmurHash3_x64_128(const void * key, const int len, const uint32_t seed, void * out)
{
    const uint8_t * data = (const uint8_t*)key;
    const int nblocks = len / 16;

    uint64_t h1 = 0x9368e53c2f6af274ULL ^ seed;
    uint64_t h2 = 0x586dcd208f7cd3fdULL ^ seed;

    uint64_t c1 = 0x87c37b91114253d5ULL;
    uint64_t c2 = 0x4cf5ad432745937fULL;

    //----------
    // body

    const uint64_t * blocks = (const uint64_t *)(data);

    for (int i = 0; i < nblocks; i++)
    {
        uint64_t k1 = getblock(blocks, i * 2 + 0);
        uint64_t k2 = getblock(blocks, i * 2 + 1);

        bmix64(h1, h2, k1, k2, c1, c2);
    }

    //----------
    // tail

    const uint8_t * tail = (const uint8_t*)(data + nblocks*16);

    uint64_t k1 = 0;
    uint64_t k2 = 0;

    switch (len & 15)
    {
    case 15: k2 ^= uint64_t(tail[14]) << 48;
    case 14: k2 ^= uint64_t(tail[13]) << 40;
    case 13: k2 ^= uint64_t(tail[12]) << 32;
    case 12: k2 ^= uint64_t(tail[11]) << 24;
    case 11: k2 ^= uint64_t(tail[10]) << 16;
    case 10: k2 ^= uint64_t(tail[ 9]) << 8;
    case  9: k2 ^= uint64_t(tail[ 8]) << 0;

    case  8: k1 ^= uint64_t(tail[ 7]) << 56;
    case  7: k1 ^= uint64_t(tail[ 6]) << 48;
    case  6: k1 ^= uint64_t(tail[ 5]) << 40;
    case  5: k1 ^= uint64_t(tail[ 4]) << 32;
    case  4: k1 ^= uint64_t(tail[ 3]) << 24;
    case  3: k1 ^= uint64_t(tail[ 2]) << 16;
    case  2: k1 ^= uint64_t(tail[ 1]) << 8;
    case  1: k1 ^= uint64_t(tail[ 0]) << 0;
             bmix64(h1, h2, k1, k2, c1, c2);
    };

    //----------
    // finalization

    h2 ^= len;

    h1 += h2;
    h2 += h1;

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 += h2;
    h2 += h1;

    ((uint64_t*)out)[0] = h1;
    ((uint64_t*)out)[1] = h2;
}

//-----------------------------------------------------------------------------
// If we need a smaller hash value, it's faster to just use a portion of the
// 128-bit hash

void MurmurHash3_x64_32(const void * key, int len, uint32_t seed, void * out)
{
    uint32_t temp[4];

    MurmurHash3_x64_128(key, len, seed, temp);

    *(uint32_t*)out = temp[0];
}

//----------

void MurmurHash3_x64_64(const void * key, int len, uint32_t seed, void * out)
{
    uint64_t temp[2];

    MurmurHash3_x64_128(key, len, seed, temp);

    *(uint64_t*)out = temp[0];
}

//-----------------------------------------------------------------------------
// MurmurHashAligned2, by Austin Appleby

// Same algorithm as MurmurHash2, but only does aligned reads - should be safer
// on certain platforms.

// Performance will be lower than MurmurHash2

uint32_t MurmurHashAligned2(const void * key, size_t len, uint32_t seed)
{
    const uint32_t m = 0x5bd1e995;
    const int r = 24;

    const unsigned char * data = (const unsigned char *)key;

    uint32_t h = seed ^ len;

    unsigned int align = (uintptr_t)data & 3;

    if (align && (len >= 4))
    {
        // Pre-load the temp registers

        uint32_t t = 0, d = 0;

        switch (align)
        {
        case 1:
            t |= data[2] << 16;
        case 2:
            t |= data[1] << 8;
        case 3:
            t |= data[0];
        }

        t <<= (8 * align);

        data += 4-align;
        len -= 4-align;

        int sl = 8 * (4-align);
        int sr = 8 * align;

        // Mix

        while (len >= 4)
        {
            d = *(uint32_t *)data;
            t = (t >> sr) | (d << sl);

            uint32_t k = t;

            MIX(h, k, m);

            t = d;

            data += 4;
            len -= 4;
        }

        // Handle leftover data in temp registers

        d = 0;

        if (len >= align)
        {
            switch (align)
            {
            case 3:
                d |= data[2] << 16;
            case 2:
                d |= data[1] << 8;
            case 1:
                d |= data[0];
            }

            uint32_t k = (t >> sr) | (d << sl);
            MIX(h, k, m);

            data += align;
            len -= align;

            //----------
            // Handle tail bytes

            switch (len)
            {
            case 3:
                h ^= data[2] << 16;
            case 2:
                h ^= data[1] << 8;
            case 1:
                h ^= data[0];
                h *= m;
            };
        }
        else
        {
            switch (len)
            {
            case 3:
                d |= data[2] << 16;
            case 2:
                d |= data[1] << 8;
            case 1:
                d |= data[0];
            case 0:
                h ^= (t >> sr) | (d << sl);
                h *= m;
            }
        }

        h ^= h >> 13;
        h *= m;
        h ^= h >> 15;

        return h;
    }
    else
    {
        while (len >= 4)
        {
            uint32_t k = *(uint32_t *)data;

            MIX(h, k, m);

            data += 4;
            len -= 4;
        }

        //----------
        // Handle tail bytes

        switch (len)
        {
        case 3:
            h ^= data[2] << 16;
        case 2:
            h ^= data[1] << 8;
        case 1:
            h ^= data[0];
            h *= m;
        };

        h ^= h >> 13;
        h *= m;
        h ^= h >> 15;

        return h;
    }
}

//-----------------------------------------------------------------------------
// MurmurHashNeutral2, by Austin Appleby

// Same as MurmurHash2, but endian- and alignment-neutral.
// Half the speed though, alas.

uint32_t MurmurHashNeutral2(const void * key, size_t len, uint32_t seed)
{
    const uint32_t m = 0x5bd1e995;
    const int r = 24;

    uint32_t h = seed ^ len;

    const unsigned char * data = (const unsigned char *)key;

    while (len >= 4)
    {
        uint32_t k;

        k  = data[0];
        k |= data[1] << 8;
        k |= data[2] << 16;
        k |= data[3] << 24;

        k *= m;
        k ^= k >> r;
        k *= m;

        h *= m;
        h ^= k;

        data += 4;
        len -= 4;
    }

    switch (len)
    {
    case 3:
        h ^= data[2] << 16;
    case 2:
        h ^= data[1] << 8;
    case 1:
        h ^= data[0];
        h *= m;
    };

    h ^= h >> 13;
    h *= m;
    h ^= h >> 15;

    return h;
}

} // namespace mytoft
} // namespace bubblefs