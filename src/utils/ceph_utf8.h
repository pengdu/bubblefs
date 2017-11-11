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

// ceph/src/common/utf8.h
// cocos2d-x/cocos/base/ccUTF8.h

#ifndef BUBBLEFS_UTILS_CEPH_UTF8_H_
#define BUBBLEFS_UTILS_CEPH_UTF8_H_

#include <locale>
#include <string>
#include <sstream>
#include <vector>

namespace bubblefs {
namespace myceph {
  
/* Checks if a buffer is valid UTF-8.
 * Returns 0 if it is, and one plus the offset of the first invalid byte
 * if it is not.
 */
int check_utf8(const char *buf, int len);

/* Checks if a null-terminated string is valid UTF-8.
 * Returns 0 if it is, and one plus the offset of the first invalid byte
 * if it is not.
 */
int check_utf8_cstr(const char *buf);

/* Returns true if 'ch' is a control character.
 * We do count newline as a control character, but not NULL.
 */
int is_control_character(int ch);

/* Checks if a buffer contains control characters.
 */
int check_for_control_characters(const char *buf, int len);

/* Checks if a null-terminated string contains control characters.
 */
int check_for_control_characters_cstr(const char *buf);

/* Encode a 31-bit UTF8 code point to 'buf'.
 * Assumes buf is of size MAX_UTF8_SZ
 * Returns -1 on failure; number of bytes in the encoded value otherwise.
 */
int encode_utf8(unsigned long u, unsigned char *buf);

/*
 * Decode a UTF8 character from an array of bytes. Return character code.
 * Upon error, return INVALID_UTF8_CHAR.
 */
unsigned long decode_utf8(unsigned char *buf, int nbytes);

/**
 *  @brief Trims the unicode spaces at the end of char16_t vector.
 */
void trimUTF16Vector(std::vector<char16_t>& str);
    
/**
 *  @brief Trims the unicode spaces at the end of char32_t vector.
 */
void trimUTF32Vector(std::vector<char32_t>& str);

/**
 *  @brief Whether the character is a whitespace character.
 *  @param ch    The unicode character.
 *  @returns     Whether the character is a white space character.
 *
 *  @see http://en.wikipedia.org/wiki/Whitespace_character#Unicode
 *
 */
bool isUnicodeSpace(char32_t ch);

/**
 *  @brief Whether the character is a Chinese/Japanese/Korean character.
 *  @param ch    The unicode character.
 *  @returns     Whether the character is a Chinese character.
 *
 *  @see http://www.searchtb.com/2012/04/chinese_encode.html
 *  @see http://tieba.baidu.com/p/748765987
 *
 */
bool isCJKUnicode(char32_t ch);

} // namespace myceph
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CEPH_UTF8_H_