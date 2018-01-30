// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// tera/src/common/base/ascii.h
// sofa-pbrpc/src/sofa/pbrpc/bin2ascii.h

#ifndef BUBBLEFS_UTILS_BDCOM_ASCII_H_
#define BUBBLEFS_UTILS_BDCOM_ASCII_H_

#include <limits.h>
#include <stdint.h>
#include <string>
#include <stdexcept>

namespace bubblefs {
namespace mybdcom {

/// 专为 ASCII 特定的函数，如果涉及到 locale，请使用标准 C 的 ctype.h
/// 里定义的函数。

// 用 struct 来用做强的 namespace，禁止使用者 using。
struct Ascii
{
private:
    Ascii();
    ~Ascii();

private:
    /// 字符类型的掩码
    enum CharTypeMask
    {
        kUpper = 1 << 0,
        kLower = 1 << 1,
        kDigit = 1 << 2,
        kHexDigit = 1 << 3,
        kBlank = 1 << 4,
        kSpace = 1 << 5,
        kControl = 1 << 6,
        kPunct = 1 << 7,
        kPrint = 1 << 8,
        kGraph = 1 << 9,
    };

public:
    /** 判断是不是有效的 ASCII 码 */
    static bool IsValid(char c)
    {
        return (c & 0x80) == 0;
    }

    static inline bool IsLower(char c)
    {
        return CharIncludeAnyTypeMask(c, kLower);
    }

    static inline bool IsUpper(char c)
    {
        return CharIncludeAnyTypeMask(c, kUpper);
    }

    /** 判断是否为字母 */
    static bool IsAlpha(char c)
    {
        return CharIncludeAnyTypeMask(c, kUpper | kLower);
    }

    /** 判断是否为数字 */
    static bool IsDigit(char c)
    {
        return CharIncludeAnyTypeMask(c, kDigit);
    }

    /** 判断是否为英文或数字  */
    static bool IsAlphaNumber(char c)
    {
        return CharIncludeAnyTypeMask(c, kUpper | kLower | kDigit);
    }

    /** 判断是否为空白字符。空格,'\t', ' ' 算作空白字符*/
    static bool IsBlank(char c)
    {
        return CharIncludeAnyTypeMask(c, kBlank);
    }

    /** 判断是否为间隔字符。*/
    static inline bool IsSpace(char c)
    {
        return CharIncludeAnyTypeMask(c, kSpace);
    }

    /** 判断是否为控制字符。*/
    static bool IsControl(char c)
    {
        return CharIncludeAnyTypeMask(c, kControl);
    }

    /** 判断是否为标点符号字符。*/
    static inline bool IsPunct(char c)
    {
        return CharIncludeAnyTypeMask(c, kPunct);
    }

    /** 判断是否为十六进制数字字符。*/
    static inline bool IsHexDigit(char c)
    {
        return CharIncludeAnyTypeMask(c, kHexDigit);
    }

    /** 判断是否为可见字符。*/
    static inline bool IsGraph(char c)
    {
        return CharIncludeAnyTypeMask(c, kGraph);
    }

    /** 判断是否为可打印字符。*/
    static inline bool IsPrint(char c)
    {
        return CharIncludeAnyTypeMask(c, kPrint);
    }

    static inline char ToAscii(char c)
    {
        return c & 0x7F;
    }

    static inline char ToLower(char c)
    {
        return IsUpper(c) ? c + ('a' - 'A') : c;
    }

    static inline char ToUpper(char c)
    {
        return IsLower(c) ? c - ('a' - 'A') : c;
    }

private:
    static inline int GetCharTypeMask(char c)
    {
#if 0
        // // 此表由以下代码生成:
        // #include <ctype.h>
        // #include <stdio.h>

        // int main()
        // {
        //     for (int i = 0; i < 128; ++i)
        //     {
        //         printf("            /* 0x%02x(%c) */ ", i, isgraph(i) ? i : ' ');
        //         if (isblank(i)) printf("kBlank | ");
        //         if (isspace(i)) printf("kSpace | ");
        //         if (isupper(i)) printf("kUpper | ");
        //         if (islower(i)) printf("kLower | ");
        //         if (isdigit(i)) printf("kDigit | ");
        //         if (isxdigit(i)) printf("kHexDigit | ");
        //         if (ispunct(i)) printf("kPunct | ");
        //         if (iscntrl(i)) printf("kControl | ");
        //         if (isgraph(i)) printf("kGraph | ");
        //         if (isprint(i)) printf("kPrint | ");
        //         printf("0,\n");
        //     }
        // }

        // 编译后以下面的命令运行：
        // $ LC_ALL=C ./a.out
#endif
        static const uint16_t table[UCHAR_MAX + 1] =
        {
            /* 0x00( ) */ kControl | 0,
            /* 0x01( ) */ kControl | 0,
            /* 0x02( ) */ kControl | 0,
            /* 0x03( ) */ kControl | 0,
            /* 0x04( ) */ kControl | 0,
            /* 0x05( ) */ kControl | 0,
            /* 0x06( ) */ kControl | 0,
            /* 0x07( ) */ kControl | 0,
            /* 0x08( ) */ kControl | 0,
            /* 0x09( ) */ kBlank | kSpace | kControl | 0,
            /* 0x0a( ) */ kSpace | kControl | 0,
            /* 0x0b( ) */ kSpace | kControl | 0,
            /* 0x0c( ) */ kSpace | kControl | 0,
            /* 0x0d( ) */ kSpace | kControl | 0,
            /* 0x0e( ) */ kControl | 0,
            /* 0x0f( ) */ kControl | 0,
            /* 0x10( ) */ kControl | 0,
            /* 0x11( ) */ kControl | 0,
            /* 0x12( ) */ kControl | 0,
            /* 0x13( ) */ kControl | 0,
            /* 0x14( ) */ kControl | 0,
            /* 0x15( ) */ kControl | 0,
            /* 0x16( ) */ kControl | 0,
            /* 0x17( ) */ kControl | 0,
            /* 0x18( ) */ kControl | 0,
            /* 0x19( ) */ kControl | 0,
            /* 0x1a( ) */ kControl | 0,
            /* 0x1b( ) */ kControl | 0,
            /* 0x1c( ) */ kControl | 0,
            /* 0x1d( ) */ kControl | 0,
            /* 0x1e( ) */ kControl | 0,
            /* 0x1f( ) */ kControl | 0,
            /* 0x20( ) */ kBlank | kSpace | kPrint | 0,
            /* 0x21(!) */ kPunct | kGraph | kPrint | 0,
            /* 0x22(") */ kPunct | kGraph | kPrint | 0,
            /* 0x23(#) */ kPunct | kGraph | kPrint | 0,
            /* 0x24($) */ kPunct | kGraph | kPrint | 0,
            /* 0x25(%) */ kPunct | kGraph | kPrint | 0,
            /* 0x26(&) */ kPunct | kGraph | kPrint | 0,
            /* 0x27(') */ kPunct | kGraph | kPrint | 0,
            /* 0x28(() */ kPunct | kGraph | kPrint | 0,
            /* 0x29()) */ kPunct | kGraph | kPrint | 0,
            /* 0x2a(*) */ kPunct | kGraph | kPrint | 0,
            /* 0x2b(+) */ kPunct | kGraph | kPrint | 0,
            /* 0x2c(,) */ kPunct | kGraph | kPrint | 0,
            /* 0x2d(-) */ kPunct | kGraph | kPrint | 0,
            /* 0x2e(.) */ kPunct | kGraph | kPrint | 0,
            /* 0x2f(/) */ kPunct | kGraph | kPrint | 0,
            /* 0x30(0) */ kDigit | kHexDigit | kGraph | kPrint | 0,
            /* 0x31(1) */ kDigit | kHexDigit | kGraph | kPrint | 0,
            /* 0x32(2) */ kDigit | kHexDigit | kGraph | kPrint | 0,
            /* 0x33(3) */ kDigit | kHexDigit | kGraph | kPrint | 0,
            /* 0x34(4) */ kDigit | kHexDigit | kGraph | kPrint | 0,
            /* 0x35(5) */ kDigit | kHexDigit | kGraph | kPrint | 0,
            /* 0x36(6) */ kDigit | kHexDigit | kGraph | kPrint | 0,
            /* 0x37(7) */ kDigit | kHexDigit | kGraph | kPrint | 0,
            /* 0x38(8) */ kDigit | kHexDigit | kGraph | kPrint | 0,
            /* 0x39(9) */ kDigit | kHexDigit | kGraph | kPrint | 0,
            /* 0x3a(:) */ kPunct | kGraph | kPrint | 0,
            /* 0x3b(;) */ kPunct | kGraph | kPrint | 0,
            /* 0x3c(<) */ kPunct | kGraph | kPrint | 0,
            /* 0x3d(=) */ kPunct | kGraph | kPrint | 0,
            /* 0x3e(>) */ kPunct | kGraph | kPrint | 0,
            /* 0x3f(?) */ kPunct | kGraph | kPrint | 0,
            /* 0x40(@) */ kPunct | kGraph | kPrint | 0,
            /* 0x41(A) */ kUpper | kHexDigit | kGraph | kPrint | 0,
            /* 0x42(B) */ kUpper | kHexDigit | kGraph | kPrint | 0,
            /* 0x43(C) */ kUpper | kHexDigit | kGraph | kPrint | 0,
            /* 0x44(D) */ kUpper | kHexDigit | kGraph | kPrint | 0,
            /* 0x45(E) */ kUpper | kHexDigit | kGraph | kPrint | 0,
            /* 0x46(F) */ kUpper | kHexDigit | kGraph | kPrint | 0,
            /* 0x47(G) */ kUpper | kGraph | kPrint | 0,
            /* 0x48(H) */ kUpper | kGraph | kPrint | 0,
            /* 0x49(I) */ kUpper | kGraph | kPrint | 0,
            /* 0x4a(J) */ kUpper | kGraph | kPrint | 0,
            /* 0x4b(K) */ kUpper | kGraph | kPrint | 0,
            /* 0x4c(L) */ kUpper | kGraph | kPrint | 0,
            /* 0x4d(M) */ kUpper | kGraph | kPrint | 0,
            /* 0x4e(N) */ kUpper | kGraph | kPrint | 0,
            /* 0x4f(O) */ kUpper | kGraph | kPrint | 0,
            /* 0x50(P) */ kUpper | kGraph | kPrint | 0,
            /* 0x51(Q) */ kUpper | kGraph | kPrint | 0,
            /* 0x52(R) */ kUpper | kGraph | kPrint | 0,
            /* 0x53(S) */ kUpper | kGraph | kPrint | 0,
            /* 0x54(T) */ kUpper | kGraph | kPrint | 0,
            /* 0x55(U) */ kUpper | kGraph | kPrint | 0,
            /* 0x56(V) */ kUpper | kGraph | kPrint | 0,
            /* 0x57(W) */ kUpper | kGraph | kPrint | 0,
            /* 0x58(X) */ kUpper | kGraph | kPrint | 0,
            /* 0x59(Y) */ kUpper | kGraph | kPrint | 0,
            /* 0x5a(Z) */ kUpper | kGraph | kPrint | 0,
            /* 0x5b([) */ kPunct | kGraph | kPrint | 0,
            /* 0x5c(\) */ kPunct | kGraph | kPrint | 0,
            /* 0x5d(]) */ kPunct | kGraph | kPrint | 0,
            /* 0x5e(^) */ kPunct | kGraph | kPrint | 0,
            /* 0x5f(_) */ kPunct | kGraph | kPrint | 0,
            /* 0x60(`) */ kPunct | kGraph | kPrint | 0,
            /* 0x61(a) */ kLower | kHexDigit | kGraph | kPrint | 0,
            /* 0x62(b) */ kLower | kHexDigit | kGraph | kPrint | 0,
            /* 0x63(c) */ kLower | kHexDigit | kGraph | kPrint | 0,
            /* 0x64(d) */ kLower | kHexDigit | kGraph | kPrint | 0,
            /* 0x65(e) */ kLower | kHexDigit | kGraph | kPrint | 0,
            /* 0x66(f) */ kLower | kHexDigit | kGraph | kPrint | 0,
            /* 0x67(g) */ kLower | kGraph | kPrint | 0,
            /* 0x68(h) */ kLower | kGraph | kPrint | 0,
            /* 0x69(i) */ kLower | kGraph | kPrint | 0,
            /* 0x6a(j) */ kLower | kGraph | kPrint | 0,
            /* 0x6b(k) */ kLower | kGraph | kPrint | 0,
            /* 0x6c(l) */ kLower | kGraph | kPrint | 0,
            /* 0x6d(m) */ kLower | kGraph | kPrint | 0,
            /* 0x6e(n) */ kLower | kGraph | kPrint | 0,
            /* 0x6f(o) */ kLower | kGraph | kPrint | 0,
            /* 0x70(p) */ kLower | kGraph | kPrint | 0,
            /* 0x71(q) */ kLower | kGraph | kPrint | 0,
            /* 0x72(r) */ kLower | kGraph | kPrint | 0,
            /* 0x73(s) */ kLower | kGraph | kPrint | 0,
            /* 0x74(t) */ kLower | kGraph | kPrint | 0,
            /* 0x75(u) */ kLower | kGraph | kPrint | 0,
            /* 0x76(v) */ kLower | kGraph | kPrint | 0,
            /* 0x77(w) */ kLower | kGraph | kPrint | 0,
            /* 0x78(x) */ kLower | kGraph | kPrint | 0,
            /* 0x79(y) */ kLower | kGraph | kPrint | 0,
            /* 0x7a(z) */ kLower | kGraph | kPrint | 0,
            /* 0x7b({) */ kPunct | kGraph | kPrint | 0,
            /* 0x7c(|) */ kPunct | kGraph | kPrint | 0,
            /* 0x7d(}) */ kPunct | kGraph | kPrint | 0,
            /* 0x7e(~) */ kPunct | kGraph | kPrint | 0,
            /* 0x7f( ) */ kControl | 0,
            // 以下全为 0
        };
        return table[static_cast<unsigned char>(c)];
    }

    static bool CharIncludeAnyTypeMask(char c, int mask)
    {
        return (GetCharTypeMask(c) & mask) != 0;
    }

    static bool CharIncludeAallTypeMask(char c, int mask)
    {
        return (GetCharTypeMask(c) & mask) == mask;
    }
};

inline std::string hex2bin(const std::string &s)
{
        if (s.size() % 2)
                throw std::runtime_error("Odd hex data size");
        static const char lookup[] = ""
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x00
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x10
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x20
                "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x80\x80\x80\x80\x80\x80" // 0x30
                "\x80\x0a\x0b\x0c\x0d\x0e\x0f\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x40
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x50
                "\x80\x0a\x0b\x0c\x0d\x0e\x0f\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x60
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x70
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x80
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x90
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xa0
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xb0
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xc0
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xd0
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xe0
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xf0
                "";
        std::string r;
        r.reserve(s.size() / 2);
        for (size_t i = 0; i < s.size(); i += 2) {
                char hi = lookup[(size_t)s[i]];
                char lo = lookup[(size_t)s[i+1]];
                if (0x80 & (hi | lo))
                        throw std::runtime_error("Invalid hex data: " + s.substr(i, 6));
                r.push_back((hi << 4) | lo);
        }
        return r;
}

inline std::string bin2hex(const std::string &s)
{
        static const char lookup[] = "0123456789abcdef";
        std::string r;
        r.reserve(s.size() * 2);
        for (size_t i = 0; i < s.size(); i++) {
                char hi = s[i] >> 4;
                char lo = s[i] & 0xf;
                r.push_back(lookup[(size_t)hi]);
                r.push_back(lookup[(size_t)lo]);
        }
        return r;
}

inline std::string b64_encode(const std::string &s)
{
        typedef unsigned char u1;
        static const char lookup[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const u1 * data = (const u1 *) s.c_str();
        std::string r;
        r.reserve(s.size() * 4 / 3 + 3);
        for (size_t i = 0; i < s.size(); i += 3) {
                unsigned n = data[i] << 16;
                if (i + 1 < s.size()) n |= data[i + 1] << 8;
                if (i + 2 < s.size()) n |= data[i + 2];

                u1 n0 = (u1)(n >> 18) & 0x3f;
                u1 n1 = (u1)(n >> 12) & 0x3f;
                u1 n2 = (u1)(n >>  6) & 0x3f;
                u1 n3 = (u1)(n      ) & 0x3f;

                r.push_back(lookup[n0]);
                r.push_back(lookup[n1]);
                if (i + 1 < s.size()) r.push_back(lookup[n2]);
                if (i + 2 < s.size()) r.push_back(lookup[n3]);
        }
        for (size_t i = 0; i < (3 - s.size() % 3) % 3; i++)
                r.push_back('=');
        return r;
}

inline std::string b64_decode(const std::string &s)
{
        typedef unsigned char u1;
        static const char lookup[] = ""
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x00
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x10
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x3e\x80\x80\x80\x3f" // 0x20
                "\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x80\x80\x80\x00\x80\x80" // 0x30
                "\x80\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e" // 0x40
                "\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x80\x80\x80\x80\x80" // 0x50
                "\x80\x1a\x1b\x1c\x1d\x1e\x1f\x20\x21\x22\x23\x24\x25\x26\x27\x28" // 0x60
                "\x29\x2a\x2b\x2c\x2d\x2e\x2f\x30\x31\x32\x33\x80\x80\x80\x80\x80" // 0x70
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x80
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0x90
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xa0
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xb0
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xc0
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xd0
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xe0
                "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80" // 0xf0
                "";
        std::string r;
        if (!s.size()) return r;
        if (s.size() % 4)
                throw std::runtime_error("Invalid base64 data size");
        size_t pad = 0;
        if (s[s.size() - 1] == '=') pad++;
        if (s[s.size() - 2] == '=') pad++;

        r.reserve(s.size() * 3 / 4 + 3);
        for (size_t i = 0; i < s.size(); i += 4) {
                u1 n0 = lookup[(u1) s[i+0]];
                u1 n1 = lookup[(u1) s[i+1]];
                u1 n2 = lookup[(u1) s[i+2]];
                u1 n3 = lookup[(u1) s[i+3]];
                if (0x80 & (n0 | n1 | n2 | n3))
                        throw std::runtime_error("Invalid hex data: " + s.substr(i, 4));
                unsigned n = (n0 << 18) | (n1 << 12) | (n2 << 6) | n3;
                r.push_back((n >> 16) & 0xff);
                if (s[i+2] != '=') r.push_back((n >> 8) & 0xff);
                if (s[i+3] != '=') r.push_back((n     ) & 0xff);
        }
        return r;
}

} // namespace mybdcom
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BDCOM_ASCII_H_