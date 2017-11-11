/*
 * Ceph - scalable distributed file system
 *
 * Copyright (C) 2011 New Dream Network
 *
 * This is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1, as published by the Free Software
 * Foundation.  See file COPYING.
 *
 */
/****************************************************************************
 Copyright (c) 2014 cocos2d-x.org
 Copyright (c) 2014-2017 Chukong Technologies Inc.
 http://www.cocos2d-x.org
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 ****************************************************************************/

// ceph/src/common/utf8.c
// cocos2d-x/cocos/base/ccUTF8.cpp

#include "utils/ceph_utf8.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace bubblefs {
namespace myceph {
  
#define MAX_UTF8_SZ 6 
#define INVALID_UTF8_CHAR 0xfffffffful
  
template<typename T>
std::string toString(T arg)
{
    std::stringstream ss;
    ss << arg;
    return ss.str();
}

static int high_bits_set(int c)
{
        int ret = 0;
        while (1) {
                if ((c & 0x80) != 0x080)
                        break;
                c <<= 1;
                ++ret;
        }
        return ret;
}

/* Encode a 31-bit UTF8 code point to 'buf'.
 * Assumes buf is of size MAX_UTF8_SZ
 * Returns -1 on failure; number of bytes in the encoded value otherwise.
 */
int encode_utf8(unsigned long u, unsigned char *buf)
{
        int i;
        unsigned long max_val[MAX_UTF8_SZ] = {
                0x0000007ful, 0x000007fful, 0x0000fffful,
                0x001ffffful, 0x03fffffful, 0x7ffffffful
        };
        static const int MAX_VAL_SZ = sizeof(max_val) / sizeof(max_val[0]);

        for (i = 0; i < MAX_VAL_SZ; ++i) {
                if (u <= max_val[i])
                        break;
        }
        if (i == MAX_VAL_SZ) {
                // This code point is too big to encode.
                return -1;
        }

        if (i == 0) {
                buf[0] = u;
        }
        else {
                signed int j;
                for (j = i; j > 0; --j) {
                        buf[j] = 0x80 | (u & 0x3f);
                        u >>= 6;
                }

                unsigned char mask = ~(0xFF >> (i + 1));
                buf[0] = mask | u;
        }

        return i + 1;
}

/*
 * Decode a UTF8 character from an array of bytes. Return character code.
 * Upon error, return INVALID_UTF8_CHAR.
 */
unsigned long decode_utf8(unsigned char *buf, int nbytes)
{
        unsigned long code;
        int i, j;

        if (nbytes <= 0)
                return INVALID_UTF8_CHAR;

        if (nbytes == 1) {
                if (buf[0] >= 0x80)
                        return INVALID_UTF8_CHAR;
                return buf[0];
        }

        i = high_bits_set(buf[0]);
        if (i != nbytes)
                return INVALID_UTF8_CHAR;
        code = buf[0] & (0xff >> i);
        for (j = 1; j < nbytes; ++j) {
                if ((buf[j] & 0xc0) != 0x80)
                            return INVALID_UTF8_CHAR;
                code = (code << 6) | (buf[j] & 0x3f);
        }

        // Check for invalid code points
        if (code == 0xFFFE)
            return INVALID_UTF8_CHAR;
        if (code == 0xFFFF)
            return INVALID_UTF8_CHAR;
        if (code >= 0xD800 && code <= 0xDFFF)
            return INVALID_UTF8_CHAR;

        return code;
}

int check_utf8(const char *buf, int len)
{
        unsigned char u[MAX_UTF8_SZ];
        int enc_len = 0;
        int i = 0;
        while (1) {
                unsigned int c = buf[i];
                if (i >= len || c < 0x80 || (c & 0xC0) != 0x80) {
                        // the start of a new character. Process what we have
                        // in the buffer.
                        if (enc_len > 0) {
                                int re_encoded_len;
                                unsigned char re_encoded[MAX_UTF8_SZ];
                                unsigned long code = decode_utf8(u, enc_len);
                                if (code == INVALID_UTF8_CHAR) {
                                        //printf("decoded to invalid utf8");
                                        return i + 1;
                                }
                                re_encoded_len = encode_utf8(code, re_encoded);
                                if (enc_len != re_encoded_len) {
                                        //printf("originally encoded as %d bytes, "
                                        //      "but was re-encoded to %d!\n",
                                        //      enc_len, re_encoded_len);
                                        return i + 1;
                                }
                                if (memcmp(u, re_encoded, enc_len) != 0) {
                                        //printf("re-encoded to a different "
                                        //      "byte stream!");
                                        return i + 1;
                                }
                                //printf("code_point %lu\n", code);
                        }
                        enc_len = 0;
                        if (i >= len)
                                break;
                        // start collecting again?
                        if (c >= 0x80)
                                u[enc_len++] = c;
                } else {
                        if (enc_len == MAX_UTF8_SZ) {
                                //printf("too many enc_len in utf character!\n");
                                return i + 1;
                        }
                        //printf("continuation byte...\n");
                        u[enc_len++] = c;
                }
                ++i;
        }
        return 0;
}

int check_utf8_cstr(const char *buf)
{
        return check_utf8(buf, strlen(buf));
}

int is_control_character(int c)
{
        return (((c != 0) && (c < 0x20)) || (c == 0x7f));
}

int check_for_control_characters(const char *buf, int len)
{
        int i;
        for (i = 0; i < len; ++i) {
                if (is_control_character((int)(unsigned char)buf[i])) {
                        return i + 1;
                }
        }
        return 0;
}

int check_for_control_characters_cstr(const char *buf)
{
        return check_for_control_characters(buf, strlen(buf));
}

/*
 * @str:    the string to search through.
 * @c:        the character to not look for.
 *
 * Return value: the index of the last character that is not c.
 * */
unsigned int getIndexOfLastNotChar16(const std::vector<char16_t>& str, char16_t c)
{
    int len = static_cast<int>(str.size());

    int i = len - 1;
    for (; i >= 0; --i)
        if (str[i] != c) return i;

    return i;
}

/*
 * @str:    the string to trim
 * @index:    the index to start trimming from.
 *
 * Trims str st str=[0, index) after the operation.
 *
 * Return value: the trimmed string.
 * */
static void trimUTF16VectorFromIndex(std::vector<char16_t>& str, int index)
{
    int size = static_cast<int>(str.size());
    if (index >= size || index < 0)
        return;

    str.erase(str.begin() + index, str.begin() + size);
}
    
/*
 * @str:    the string to trim
 * @index:    the index to start trimming from.
 *
 * Trims str st str=[0, index) after the operation.
 *
 * Return value: the trimmed string.
 * */
static void trimUTF32VectorFromIndex(std::vector<char32_t>& str, int index)
{
    int size = static_cast<int>(str.size());
    if (index >= size || index < 0)
        return;
    
    str.erase(str.begin() + index, str.begin() + size);
}

/*
 * @ch is the unicode character whitespace?
 *
 * Reference: http://en.wikipedia.org/wiki/Whitespace_character#Unicode
 *
 * Return value: weather the character is a whitespace character.
 * */
bool isUnicodeSpace(char32_t ch)
{
    return  (ch >= 0x0009 && ch <= 0x000D) || ch == 0x0020 || ch == 0x0085 || ch == 0x00A0 || ch == 0x1680
    || (ch >= 0x2000 && ch <= 0x200A) || ch == 0x2028 || ch == 0x2029 || ch == 0x202F
    ||  ch == 0x205F || ch == 0x3000;
}

bool isCJKUnicode(char32_t ch)
{
    return (ch >= 0x4E00 && ch <= 0x9FBF)   // CJK Unified Ideographs
        || (ch >= 0x2E80 && ch <= 0x2FDF)   // CJK Radicals Supplement & Kangxi Radicals
        || (ch >= 0x2FF0 && ch <= 0x30FF)   // Ideographic Description Characters, CJK Symbols and Punctuation & Japanese
        || (ch >= 0x3100 && ch <= 0x31BF)   // Korean
        || (ch >= 0xAC00 && ch <= 0xD7AF)   // Hangul Syllables
        || (ch >= 0xF900 && ch <= 0xFAFF)   // CJK Compatibility Ideographs
        || (ch >= 0xFE30 && ch <= 0xFE4F)   // CJK Compatibility Forms
        || (ch >= 0x31C0 && ch <= 0x4DFF)   // Other extensions
        || (ch >= 0x1f004 && ch <= 0x1f682);// Emoji
}

void trimUTF16Vector(std::vector<char16_t>& str)
{
    int len = static_cast<int>(str.size());

    if ( len <= 0 )
        return;

    int last_index = len - 1;

    // Only start trimming if the last character is whitespace..
    if (isUnicodeSpace(str[last_index]))
    {
        for (int i = last_index - 1; i >= 0; --i)
        {
            if (isUnicodeSpace(str[i]))
                last_index = i;
            else
                break;
        }

        trimUTF16VectorFromIndex(str, last_index);
    }
}

void trimUTF32Vector(std::vector<char32_t>& str)
{
    int len = static_cast<int>(str.size());
    
    if ( len <= 0 )
        return;
    
    int last_index = len - 1;
    
    // Only start trimming if the last character is whitespace..
    if (isUnicodeSpace(str[last_index]))
    {
        for (int i = last_index - 1; i >= 0; --i)
        {
            if (isUnicodeSpace(str[i]))
                last_index = i;
            else
                break;
        }
        
        trimUTF32VectorFromIndex(str, last_index);
    }
}

} // namespace myceph
} // namespace bubblefs