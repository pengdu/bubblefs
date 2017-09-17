/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
//=================================================================================
//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
/*
 * Tencent is pleased to support the open source community by making Pebble available.
 * Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.
 * Licensed under the MIT License (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 * http://opensource.org/licenses/MIT
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 */
// Tencent is pleased to support the open source community by making Mars available.
// Copyright (C) 2016 THL A29 Limited, a Tencent company. All rights reserved.

// Licensed under the MIT License (the "License"); you may not use this file except in 
// compliance with the License. You may obtain a copy of the License at
// http://opensource.org/licenses/MIT

// Unless required by applicable law or agreed to in writing, software distributed under the License is
// distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.
// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////

// protobuf/src/google/protobuf/stubs/strutil.cc
// Paddle/paddle/utils/StringUtil.cc
// Pebble/src/common/string_utility.cpp
// baidu/common/include/string_util.cc
// mars/mars/comm/strutil.cc
// rocksdb/util/string_util.cc
// tensorflow/tensorflow/core/lib/strings/str_util.cc

#include "utils/str_util.h"
#include <inttypes.h>
#include "float.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <utility>
#include <vector>
#include "utils/numbers.h"
#include "utils/stringprintf.h"

namespace bubblefs {
namespace str_util {

const string kNullptrString = "nullptr";  
  
static char hex_char[] = "0123456789abcdef";

// for micros < 10ms, print "XX us".
// for micros < 10sec, print "XX ms".
// for micros >= 10 sec, print "XX sec".
// for micros <= 1 hour, print Y:X M:S".
// for micros > 1 hour, print Z:Y:X H:M:S".
int AppendHumanMicros(uint64_t micros, char* output, int len,
                      bool fixed_format) {
  if (micros < 10000 && !fixed_format) {
    return snprintf(output, len, "%" PRIu64 " us", micros);
  } else if (micros < 10000000 && !fixed_format) {
    return snprintf(output, len, "%.3lf ms",
                    static_cast<double>(micros) / 1000);
  } else if (micros < 1000000l * 60 && !fixed_format) {
    return snprintf(output, len, "%.3lf sec",
                    static_cast<double>(micros) / 1000000);
  } else if (micros < 1000000ll * 60 * 60 && !fixed_format) {
    return snprintf(output, len, "%02" PRIu64 ":%05.3f M:S",
                    micros / 1000000 / 60,
                    static_cast<double>(micros % 60000000) / 1000000);
  } else {
    return snprintf(output, len, "%02" PRIu64 ":%02" PRIu64 ":%05.3f H:M:S",
                    micros / 1000000 / 3600, (micros / 1000000 / 60) % 60,
                    static_cast<double>(micros % 60000000) / 1000000);
  }
}

// for sizes >=10TB, print "XXTB"
// for sizes >=10GB, print "XXGB"
// etc.
// append file size summary to output and return the len
int AppendHumanBytes(uint64_t bytes, char* output, int len) {
  const uint64_t ull10 = 10;
  if (bytes >= ull10 << 40) {
    return snprintf(output, len, "%" PRIu64 "TB", bytes >> 40);
  } else if (bytes >= ull10 << 30) {
    return snprintf(output, len, "%" PRIu64 "GB", bytes >> 30);
  } else if (bytes >= ull10 << 20) {
    return snprintf(output, len, "%" PRIu64 "MB", bytes >> 20);
  } else if (bytes >= ull10 << 10) {
    return snprintf(output, len, "%" PRIu64 "KB", bytes >> 10);
  } else {
    return snprintf(output, len, "%" PRIu64 "B", bytes);
  }
}

void AppendNumberTo(string* str, uint64_t num) {
  char buf[30];
  snprintf(buf, sizeof(buf), "%" PRIu64, num);
  str->append(buf);
}

string NumberToString(int64_t num) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%ld", num);
    return string(buf);
}

string NumberToString(int num) {
    return NumberToString(static_cast<int64_t>(num));
}

string NumberToString(uint32_t num) {
    return NumberToString(static_cast<int64_t>(num));
}

string NumberToString(double num) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.3f", num);
    return string(buf);
}


string NumberToHumanString(int64_t num) {
  char buf[19];
  int64_t absnum = num < 0 ? -num : num;
  if (absnum < 10000) {
    snprintf(buf, sizeof(buf), "%" PRIi64, num);
  } else if (absnum < 10000000) {
    snprintf(buf, sizeof(buf), "%" PRIi64 "K", num / 1000);
  } else if (absnum < 10000000000LL) {
    snprintf(buf, sizeof(buf), "%" PRIi64 "M", num / 1000000);
  } else {
    snprintf(buf, sizeof(buf), "%" PRIi64 "G", num / 1000000000);
  }
  return std::string(buf);
}

string BytesToHumanString(uint64_t bytes) {
  const char* size_name[] = {"KB", "MB", "GB", "TB"};
  double final_size = static_cast<double>(bytes);
  size_t size_idx;

  // always start with KB
  final_size /= 1024;
  size_idx = 0;

  while (size_idx < 3 && final_size >= 1024) {
    final_size /= 1024;
    size_idx++;
  }

  char buf[20];
  snprintf(buf, sizeof(buf), "%.2f %s", final_size, size_name[size_idx]);
  return std::string(buf);
}

bool ParseBoolean(const string& type, const string& value) {
  if (value == "true" || value == "1") {
    return true;
  } else if (value == "false" || value == "0") {
    return false;
  }
  throw std::invalid_argument(type);
}

uint32_t ParseUint32(const string& value) {
  uint64_t num = ParseUint64(value);
  if ((num >> 32LL) == 0) {
    return static_cast<uint32_t>(num);
  } else {
    throw std::out_of_range(value);
  }
}

uint64_t ParseUint64(const string& value) {
  size_t endchar;
  uint64_t num = std::stoull(value.c_str(), &endchar);
  if (endchar < value.length()) {
    char c = value[endchar];
    if (c == 'k' || c == 'K')
      num <<= 10LL;
    else if (c == 'm' || c == 'M')
      num <<= 20LL;
    else if (c == 'g' || c == 'G')
      num <<= 30LL;
    else if (c == 't' || c == 'T')
      num <<= 40LL;
  }

  return num;
}

int ParseInt(const string& value) {
  size_t endchar;
  int num = std::stoi(value.c_str(), &endchar);
  if (endchar < value.length()) {
    char c = value[endchar];
    if (c == 'k' || c == 'K')
      num <<= 10;
    else if (c == 'm' || c == 'M')
      num <<= 20;
    else if (c == 'g' || c == 'G')
      num <<= 30;
  }

  return num;
}

double ParseDouble(const string& value) {
  return std::stod(value);
}

size_t ParseSizeT(const string& value) {
  return static_cast<size_t>(ParseUint64(value));
}

std::vector<int> ParseVectorInt(const string& value) {
  std::vector<int> result;
  size_t start = 0;
  while (start < value.size()) {
    size_t end = value.find(':', start);
    if (end == string::npos) {
      result.push_back(ParseInt(value.substr(start)));
      break;
    } else {
      result.push_back(ParseInt(value.substr(start, end - start)));
      start = end + 1;
    }
  }
  return result;
}

bool SerializeIntVector(const std::vector<int>& vec, string* value) {
  *value = "";
  for (size_t i = 0; i < vec.size(); ++i) {
    if (i > 0) {
      *value += ":";
    }
    *value += ToString(vec[i]);
  }
  return true;
}

void ToUpper(string* str)
{
    std::transform(str->begin(), str->end(), str->begin(), ::toupper);
}

void ToLower(string* str)
{
    std::transform(str->begin(), str->end(), str->begin(), ::tolower);
}

string Hex2Str(const char* _str, unsigned int _len) {
    string outstr="";
    for(unsigned int i = 0; i< _len;i++) {
        char tmp[8];
        memset(tmp,0,sizeof(tmp));
        snprintf(tmp,sizeof(tmp)-1,"%02x",(unsigned char)_str[i]);
        string tmpstr = tmp;
        outstr = outstr+tmpstr;

    }
    return outstr;
}

string Str2Hex(const char* _str, unsigned int _len) {
    char outbuffer[64];
    
    unsigned int outoffset = 0;
    const char * ptr = _str;
    unsigned int  length = _len/2;
    
    if (length > sizeof(outbuffer))
        length = sizeof(outbuffer);
    
    for(unsigned int i = 0; i< length;i++) {
        char tmp[4];
        
        memset(tmp,0,sizeof(tmp));
        tmp[0] = ptr[i*2];
        tmp[1] = ptr[i*2+1];
        char *p = nullptr;
        outbuffer[outoffset] = (char)strtol(tmp,&p,16);
        outoffset++;
    }
    std::string ret ;
    ret.assign(outbuffer,outoffset);
    return ret;
}

bool Hex2Bin(const char* hex_str, string* bin_str)
{
    if (nullptr == hex_str || nullptr == bin_str) {
        return false;
    }
    bin_str->clear();
    while (*hex_str != '\0') {
        if (hex_str[1] == '\0') {
            return false;
        }
        uint8_t high = static_cast<uint8_t>(hex_str[0]);
        uint8_t low = static_cast<uint8_t>(hex_str[1]);
#define ASCII2DEC(c) \
    if (c >= '0' && c <= '9') c -= '0'; \
        else if (c >= 'A' && c <= 'F') c -= ('A' - 10); \
        else if (c >= 'a' && c <= 'f') c -= ('a' - 10); \
        else return false
        ASCII2DEC(high);
        ASCII2DEC(low);
        bin_str->append(1, static_cast<char>((high << 4) + low));
        hex_str += 2;
    }
    return true;
}

bool Bin2Hex(const char* bin_str, string* hex_str)
{
    if (nullptr == bin_str || nullptr == hex_str) {
        return false;
    }
    hex_str->clear();
    while (*bin_str != '\0') {
        uint8_t high = (static_cast<uint8_t>(*bin_str) >> 4);
        uint8_t low = (static_cast<uint8_t>(*bin_str) & 0xF);
#define DEC2ASCII(c) \
    if (c <= 9) c += '0'; \
    else c += ('A' - 10)
        DEC2ASCII(high);
        DEC2ASCII(low);
        hex_str->append(1, static_cast<char>(high));
        hex_str->append(1, static_cast<char>(low));
        bin_str += 1;
    }
    return true;
}

static const char ENCODECHARS[1024] = {
    3, '%', '0', '0', 3, '%', '0', '1', 3, '%', '0', '2', 3, '%', '0', '3',
    3, '%', '0', '4', 3, '%', '0', '5', 3, '%', '0', '6', 3, '%', '0', '7',
    3, '%', '0', '8', 3, '%', '0', '9', 3, '%', '0', 'A', 3, '%', '0', 'B',
    3, '%', '0', 'C', 3, '%', '0', 'D', 3, '%', '0', 'E', 3, '%', '0', 'F',
    3, '%', '1', '0', 3, '%', '1', '1', 3, '%', '1', '2', 3, '%', '1', '3',
    3, '%', '1', '4', 3, '%', '1', '5', 3, '%', '1', '6', 3, '%', '1', '7',
    3, '%', '1', '8', 3, '%', '1', '9', 3, '%', '1', 'A', 3, '%', '1', 'B',
    3, '%', '1', 'C', 3, '%', '1', 'D', 3, '%', '1', 'E', 3, '%', '1', 'F',
    1, '+', '2', '0', 3, '%', '2', '1', 3, '%', '2', '2', 3, '%', '2', '3',
    3, '%', '2', '4', 3, '%', '2', '5', 3, '%', '2', '6', 3, '%', '2', '7',
    3, '%', '2', '8', 3, '%', '2', '9', 3, '%', '2', 'A', 3, '%', '2', 'B',
    3, '%', '2', 'C', 1, '-', '2', 'D', 1, '.', '2', 'E', 3, '%', '2', 'F',
    1, '0', '3', '0', 1, '1', '3', '1', 1, '2', '3', '2', 1, '3', '3', '3',
    1, '4', '3', '4', 1, '5', '3', '5', 1, '6', '3', '6', 1, '7', '3', '7',
    1, '8', '3', '8', 1, '9', '3', '9', 3, '%', '3', 'A', 3, '%', '3', 'B',
    3, '%', '3', 'C', 3, '%', '3', 'D', 3, '%', '3', 'E', 3, '%', '3', 'F',
    3, '%', '4', '0', 1, 'A', '4', '1', 1, 'B', '4', '2', 1, 'C', '4', '3',
    1, 'D', '4', '4', 1, 'E', '4', '5', 1, 'F', '4', '6', 1, 'G', '4', '7',
    1, 'H', '4', '8', 1, 'I', '4', '9', 1, 'J', '4', 'A', 1, 'K', '4', 'B',
    1, 'L', '4', 'C', 1, 'M', '4', 'D', 1, 'N', '4', 'E', 1, 'O', '4', 'F',
    1, 'P', '5', '0', 1, 'Q', '5', '1', 1, 'R', '5', '2', 1, 'S', '5', '3',
    1, 'T', '5', '4', 1, 'U', '5', '5', 1, 'V', '5', '6', 1, 'W', '5', '7',
    1, 'X', '5', '8', 1, 'Y', '5', '9', 1, 'Z', '5', 'A', 3, '%', '5', 'B',
    3, '%', '5', 'C', 3, '%', '5', 'D', 3, '%', '5', 'E', 1, '_', '5', 'F',
    3, '%', '6', '0', 1, 'a', '6', '1', 1, 'b', '6', '2', 1, 'c', '6', '3',
    1, 'd', '6', '4', 1, 'e', '6', '5', 1, 'f', '6', '6', 1, 'g', '6', '7',
    1, 'h', '6', '8', 1, 'i', '6', '9', 1, 'j', '6', 'A', 1, 'k', '6', 'B',
    1, 'l', '6', 'C', 1, 'm', '6', 'D', 1, 'n', '6', 'E', 1, 'o', '6', 'F',
    1, 'p', '7', '0', 1, 'q', '7', '1', 1, 'r', '7', '2', 1, 's', '7', '3',
    1, 't', '7', '4', 1, 'u', '7', '5', 1, 'v', '7', '6', 1, 'w', '7', '7',
    1, 'x', '7', '8', 1, 'y', '7', '9', 1, 'z', '7', 'A', 3, '%', '7', 'B',
    3, '%', '7', 'C', 3, '%', '7', 'D', 1, '~', '7', 'E', 3, '%', '7', 'F',
    3, '%', '8', '0', 3, '%', '8', '1', 3, '%', '8', '2', 3, '%', '8', '3',
    3, '%', '8', '4', 3, '%', '8', '5', 3, '%', '8', '6', 3, '%', '8', '7',
    3, '%', '8', '8', 3, '%', '8', '9', 3, '%', '8', 'A', 3, '%', '8', 'B',
    3, '%', '8', 'C', 3, '%', '8', 'D', 3, '%', '8', 'E', 3, '%', '8', 'F',
    3, '%', '9', '0', 3, '%', '9', '1', 3, '%', '9', '2', 3, '%', '9', '3',
    3, '%', '9', '4', 3, '%', '9', '5', 3, '%', '9', '6', 3, '%', '9', '7',
    3, '%', '9', '8', 3, '%', '9', '9', 3, '%', '9', 'A', 3, '%', '9', 'B',
    3, '%', '9', 'C', 3, '%', '9', 'D', 3, '%', '9', 'E', 3, '%', '9', 'F',
    3, '%', 'A', '0', 3, '%', 'A', '1', 3, '%', 'A', '2', 3, '%', 'A', '3',
    3, '%', 'A', '4', 3, '%', 'A', '5', 3, '%', 'A', '6', 3, '%', 'A', '7',
    3, '%', 'A', '8', 3, '%', 'A', '9', 3, '%', 'A', 'A', 3, '%', 'A', 'B',
    3, '%', 'A', 'C', 3, '%', 'A', 'D', 3, '%', 'A', 'E', 3, '%', 'A', 'F',
    3, '%', 'B', '0', 3, '%', 'B', '1', 3, '%', 'B', '2', 3, '%', 'B', '3',
    3, '%', 'B', '4', 3, '%', 'B', '5', 3, '%', 'B', '6', 3, '%', 'B', '7',
    3, '%', 'B', '8', 3, '%', 'B', '9', 3, '%', 'B', 'A', 3, '%', 'B', 'B',
    3, '%', 'B', 'C', 3, '%', 'B', 'D', 3, '%', 'B', 'E', 3, '%', 'B', 'F',
    3, '%', 'C', '0', 3, '%', 'C', '1', 3, '%', 'C', '2', 3, '%', 'C', '3',
    3, '%', 'C', '4', 3, '%', 'C', '5', 3, '%', 'C', '6', 3, '%', 'C', '7',
    3, '%', 'C', '8', 3, '%', 'C', '9', 3, '%', 'C', 'A', 3, '%', 'C', 'B',
    3, '%', 'C', 'C', 3, '%', 'C', 'D', 3, '%', 'C', 'E', 3, '%', 'C', 'F',
    3, '%', 'D', '0', 3, '%', 'D', '1', 3, '%', 'D', '2', 3, '%', 'D', '3',
    3, '%', 'D', '4', 3, '%', 'D', '5', 3, '%', 'D', '6', 3, '%', 'D', '7',
    3, '%', 'D', '8', 3, '%', 'D', '9', 3, '%', 'D', 'A', 3, '%', 'D', 'B',
    3, '%', 'D', 'C', 3, '%', 'D', 'D', 3, '%', 'D', 'E', 3, '%', 'D', 'F',
    3, '%', 'E', '0', 3, '%', 'E', '1', 3, '%', 'E', '2', 3, '%', 'E', '3',
    3, '%', 'E', '4', 3, '%', 'E', '5', 3, '%', 'E', '6', 3, '%', 'E', '7',
    3, '%', 'E', '8', 3, '%', 'E', '9', 3, '%', 'E', 'A', 3, '%', 'E', 'B',
    3, '%', 'E', 'C', 3, '%', 'E', 'D', 3, '%', 'E', 'E', 3, '%', 'E', 'F',
    3, '%', 'F', '0', 3, '%', 'F', '1', 3, '%', 'F', '2', 3, '%', 'F', '3',
    3, '%', 'F', '4', 3, '%', 'F', '5', 3, '%', 'F', '6', 3, '%', 'F', '7',
    3, '%', 'F', '8', 3, '%', 'F', '9', 3, '%', 'F', 'A', 3, '%', 'F', 'B',
    3, '%', 'F', 'C', 3, '%', 'F', 'D', 3, '%', 'F', 'E', 3, '%', 'F', 'F',
};

void UrlEncode(const string& src_str, string* dst_str)
{
    dst_str->clear();
    for (size_t i = 0; i < src_str.length() ; i++) {
        unsigned short offset = static_cast<unsigned short>(src_str[i]) * 4;
        dst_str->append((ENCODECHARS + offset + 1), ENCODECHARS[offset]);
    }
}

static const char HEX2DEC[256] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0 ,  1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1, -1, -1, -1, -1,
    -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
};

void UrlDecode(const string& src_str, string* dst_str)
{
    dst_str->clear();
    const unsigned char* src_begin = reinterpret_cast<const unsigned char*>(src_str.data());
    const unsigned char* src_end = src_begin + src_str.length();
    const unsigned char* src_last = src_end - 2;
    while (src_begin < src_last) {
        if ((*src_begin) == '%') {
            char dec1, dec2;
            if (-1 != (dec1 = HEX2DEC[*(src_begin + 1)])
                && -1 != (dec2 = HEX2DEC[*(src_begin + 2)])) {
                dst_str->append(1, (dec1 << 4) + dec2);
                src_begin += 3;
                continue;
            }
        } else if ((*src_begin) == '+') {
            dst_str->append(1, ' ');
            ++src_begin;
            continue;
        }
        dst_str->append(1, static_cast<char>(*src_begin));
        ++src_begin;
    }
    while (src_begin < src_end) {
        dst_str->append(1, static_cast<char>(*src_begin));
        ++src_begin;
    }
}

bool StartsWith(const string& str, const string& prefix) {
  if (prefix.length() > str.length()) {
      return false;
  }
  if (memcmp(str.c_str(), prefix.c_str(), prefix.length()) == 0) {
      return true;
  }
  return false;
}

bool EndsWith(const string& str, const string& suffix) {
  if (suffix.length() > str.length()) {
      return false;
  }
  return (str.substr(str.length() - suffix.length()) == suffix);
}

bool StripSuffix(string* str, const string& suffix) {
    if (str->length() >= suffix.length()) {
        size_t suffix_pos = str->length() - suffix.length();
        if (str->compare(suffix_pos, string::npos, suffix) == 0) {
            str->resize(str->size() - suffix.size());
            return true;
        }
    }

    return false;
}

bool StripPrefix(string* str, const string& prefix) {
    if (str->length() >= prefix.length()) {
        if (str->substr(0, prefix.size()) == prefix) {
            *str = str->substr(prefix.size());
            return true;
        }
    }
    return false;
}

string CEscape(StringPiece src) {
  string dest;

  for (unsigned char c : src) {
    switch (c) {
      case '\n':
        dest.append("\\n");
        break;
      case '\r':
        dest.append("\\r");
        break;
      case '\t':
        dest.append("\\t");
        break;
      case '\"':
        dest.append("\\\"");
        break;
      case '\'':
        dest.append("\\'");
        break;
      case '\\':
        dest.append("\\\\");
        break;
      default:
        // Note that if we emit \xNN and the src character after that is a hex
        // digit then that digit must be escaped too to prevent it being
        // interpreted as part of the character code by C.
        if ((c >= 0x80) || !isprint(c)) {
          dest.append("\\");
          dest.push_back(hex_char[c / 64]);
          dest.push_back(hex_char[(c % 64) / 8]);
          dest.push_back(hex_char[c % 8]);
        } else {
          dest.push_back(c);
          break;
        }
    }
  }

  return dest;
}

namespace {  // Private helpers for CUnescape().

inline bool is_octal_digit(unsigned char c) { return c >= '0' && c <= '7'; }

inline bool ascii_isxdigit(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
         (c >= 'A' && c <= 'F');
}

inline int hex_digit_to_int(char c) {
  int x = static_cast<unsigned char>(c);
  if (x > '9') {
    x += 9;
  }
  return x & 0xf;
}

bool CUnescapeInternal(StringPiece source, char* dest,
                       string::size_type* dest_len, string* error) {
  char* d = dest;
  const char* p = source.data();
  const char* end = source.end();
  const char* last_byte = end - 1;

  // Small optimization for case where source = dest and there's no escaping
  while (p == d && p < end && *p != '\\') p++, d++;

  while (p < end) {
    if (*p != '\\') {
      *d++ = *p++;
    } else {
      if (++p > last_byte) {  // skip past the '\\'
        if (error) *error = "String cannot end with \\";
        return false;
      }
      switch (*p) {
        case 'a':
          *d++ = '\a';
          break;
        case 'b':
          *d++ = '\b';
          break;
        case 'f':
          *d++ = '\f';
          break;
        case 'n':
          *d++ = '\n';
          break;
        case 'r':
          *d++ = '\r';
          break;
        case 't':
          *d++ = '\t';
          break;
        case 'v':
          *d++ = '\v';
          break;
        case '\\':
          *d++ = '\\';
          break;
        case '?':
          *d++ = '\?';
          break;  // \?  Who knew?
        case '\'':
          *d++ = '\'';
          break;
        case '"':
          *d++ = '\"';
          break;
        case '0':
        case '1':
        case '2':
        case '3':  // octal digit: 1 to 3 digits
        case '4':
        case '5':
        case '6':
        case '7': {
          const char* octal_start = p;
          unsigned int ch = *p - '0';
          if (p < last_byte && is_octal_digit(p[1])) ch = ch * 8 + *++p - '0';
          if (p < last_byte && is_octal_digit(p[1]))
            ch = ch * 8 + *++p - '0';  // now points at last digit
          if (ch > 0xff) {
            if (error) {
              *error = "Value of \\" +
                       string(octal_start, p + 1 - octal_start) +
                       " exceeds 0xff";
            }
            return false;
          }
          *d++ = ch;
          break;
        }
        case 'x':
        case 'X': {
          if (p >= last_byte) {
            if (error) *error = "String cannot end with \\x";
            return false;
          } else if (!ascii_isxdigit(p[1])) {
            if (error) *error = "\\x cannot be followed by a non-hex digit";
            return false;
          }
          unsigned int ch = 0;
          const char* hex_start = p;
          while (p < last_byte && ascii_isxdigit(p[1]))
            // Arbitrarily many hex digits
            ch = (ch << 4) + hex_digit_to_int(*++p);
          if (ch > 0xFF) {
            if (error) {
              *error = "Value of \\" + string(hex_start, p + 1 - hex_start) +
                       " exceeds 0xff";
            }
            return false;
          }
          *d++ = ch;
          break;
        }
        default: {
          if (error) *error = string("Unknown escape sequence: \\") + *p;
          return false;
        }
      }
      p++;  // read past letter we escaped
    }
  }
  *dest_len = d - dest;
  return true;
}

template <typename T>
bool SplitAndParseAsInts(StringPiece text, char delim,
                         std::function<bool(StringPiece, T*)> converter,
                         std::vector<T>* result) {
  result->clear();
  std::vector<string> num_strings = Split(text, delim);
  for (const auto& s : num_strings) {
    T num;
    if (!converter(s, &num)) return false;
    result->push_back(num);
  }
  return true;
}

}  // namespace

bool CUnescape(StringPiece source, string* dest, string* error) {
  dest->resize(source.size());
  string::size_type dest_size;
  if (!CUnescapeInternal(source, const_cast<char*>(dest->data()), &dest_size,
                         error)) {
    return false;
  }
  dest->erase(dest_size);
  return true;
}

void StripTrailingWhitespace(string* s) {
  string::size_type i;
  for (i = s->size(); i > 0 && isspace((*s)[i - 1]); --i) {
  }
  s->resize(i);
}

string& Ltrim(string& str) { // NOLINT
    string::iterator it = std::find_if(str.begin(), str.end(), std::not1(std::ptr_fun(::isspace)));
    str.erase(str.begin(), it);
    return str;
}

string& Rtrim(string& str) { // NOLINT
    string::reverse_iterator it = std::find_if(str.rbegin(),
        str.rend(), std::not1(std::ptr_fun(::isspace)));

    str.erase(it.base(), str.end());
    return str;
}

string& Trim(string& str) { // NOLINT
    return Rtrim(Ltrim(str));
}

void Trim(std::vector<string>* str_list) {
    if (nullptr == str_list) {
        return;
    }

    std::vector<string>::iterator it;
    for (it = str_list->begin(); it != str_list->end(); ++it) {
        *it = Trim(*it);
    }
}

string TrimString(const string& str, const string& trim) {
    string::size_type pos = str.find_first_not_of(trim);
    if (pos == string::npos) {
        return "";
    }
    string::size_type pos2 = str.find_last_not_of(trim);
    if (pos2 != string::npos) {
        return str.substr(pos, pos2 - pos + 1);
    }
    return str.substr(pos);
}

// Return lower-cased version of s.
string Lowercase(StringPiece s) {
  string result(s.data(), s.size());
  for (char& c : result) {
    c = tolower(c);
  }
  return result;
}

// Return upper-cased version of s.
string Uppercase(StringPiece s) {
  string result(s.data(), s.size());
  for (char& c : result) {
    c = toupper(c);
  }
  return result;
}

string ArgDefCase(StringPiece s) {
  const size_t n = s.size();

  // Compute the size of resulting string.
  // Number of extra underscores we will need to add.
  size_t extra_us = 0;
  // Number of non-alpha chars in the beginning to skip.
  size_t to_skip = 0;
  for (size_t i = 0; i < n; ++i) {
    // If we are skipping and current letter is non-alpha, skip it as well
    if (i == to_skip && !isalpha(s[i])) {
      ++to_skip;
      continue;
    }

    // If we are here, we are not skipping any more.
    // If this letter is upper case, not the very first char in the
    // resulting string, and previous letter isn't replaced with an underscore,
    // we will need to insert an underscore.
    if (isupper(s[i]) && i != to_skip && i > 0 && isalnum(s[i - 1])) {
      ++extra_us;
    }
  }

  // Initialize result with all '_'s. There is no string
  // constructor that does not initialize memory.
  string result(n + extra_us - to_skip, '_');
  // i - index into s
  // j - index into result
  for (size_t i = to_skip, j = 0; i < n; ++i, ++j) {
    DCHECK_LT(j, result.size());
    char c = s[i];
    // If c is not alphanumeric, we don't need to do anything
    // since there is already an underscore in its place.
    if (isalnum(c)) {
      if (isupper(c)) {
        // If current char is upper case, we might need to insert an
        // underscore.
        if (i != to_skip) {
          DCHECK_GT(j, 0);
          if (result[j - 1] != '_') ++j;
        }
        result[j] = tolower(c);
      } else {
        result[j] = c;
      }
    }
  }

  return result;
}

void TitlecaseString(string* s, StringPiece delimiters) {
  bool upper = true;
  for (string::iterator ss = s->begin(); ss != s->end(); ++ss) {
    if (upper) {
      *ss = toupper(*ss);
    }
    upper = (delimiters.find(*ss) != StringPiece::npos);
  }
}

string StringReplace(StringPiece s, StringPiece oldsub, StringPiece newsub,
                     bool replace_all) {
  // TODO(jlebar): We could avoid having to shift data around in the string if
  // we had a StringPiece::find() overload that searched for a StringPiece.
  string res = s.ToString();
  size_t pos = 0;
  while ((pos = res.find(oldsub.data(), pos, oldsub.size())) != string::npos) {
    res.replace(pos, oldsub.size(), newsub.data(), newsub.size());
    pos += newsub.size();
    if (oldsub.empty()) {
      pos++;  // Match at the beginning of the text and after every byte
    }
    if (!replace_all) {
      break;
    }
  }
  return res;
}

void string_replace(const string &sub_str1,
                    const string &sub_str2, string *str) {
    string::size_type pos = 0;
    string::size_type a = sub_str1.size();
    string::size_type b = sub_str2.size();
    while ((pos = str->find(sub_str1, pos)) != string::npos) {
        str->replace(pos, a, sub_str2);
        pos += b;
    }
}

size_t RemoveLeadingWhitespace(StringPiece* text) {
  size_t count = 0;
  const char* ptr = text->data();
  while (count < text->size() && isspace(*ptr)) {
    count++;
    ptr++;
  }
  text->remove_prefix(count);
  return count;
}

size_t RemoveTrailingWhitespace(StringPiece* text) {
  size_t count = 0;
  const char* ptr = text->data() + text->size() - 1;
  while (count < text->size() && isspace(*ptr)) {
    ++count;
    --ptr;
  }
  text->remove_suffix(count);
  return count;
}

size_t RemoveWhitespaceContext(StringPiece* text) {
  // use RemoveLeadingWhitespace() and RemoveTrailingWhitespace() to do the job
  return (RemoveLeadingWhitespace(text) + RemoveTrailingWhitespace(text));
}

bool ConsumePrefix(StringPiece* s, StringPiece expected) {
  if (s->starts_with(expected)) {
    s->remove_prefix(expected.size());
    return true;
  }
  return false;
}

bool ConsumeSuffix(StringPiece* s, StringPiece expected) {
  if (s->ends_with(expected)) {
    s->remove_suffix(expected.size());
    return true;
  }
  return false;
}

bool ConsumeLeadingDigits(StringPiece* s, uint64* val) {
  const char* p = s->data();
  const char* limit = p + s->size();
  uint64 v = 0;
  while (p < limit) {
    const char c = *p;
    if (c < '0' || c > '9') break;
    uint64 new_v = (v * 10) + (c - '0');
    if (new_v / 8 < v) {
      // Overflow occurred
      return false;
    }
    v = new_v;
    p++;
  }
  if (p > s->data()) {
    // Consume some digits
    s->remove_prefix(p - s->data());
    *val = v;
    return true;
  } else {
    return false;
  }
}

bool ConsumeNonWhitespace(StringPiece* s, StringPiece* val) {
  const char* p = s->data();
  const char* limit = p + s->size();
  while (p < limit) {
    const char c = *p;
    if (isspace(c)) break;
    p++;
  }
  const size_t n = p - s->data();
  if (n > 0) {
    val->set(s->data(), n);
    s->remove_prefix(n);
    return true;
  } else {
    val->clear();
    return false;
  }
}

string DebugString(const string& src) {
    size_t src_len = src.size();
    string dst;
    dst.resize(src_len << 2);

    size_t j = 0;
    for (size_t i = 0; i < src_len; i++) {
        uint8_t c = src[i];
        if (IsVisible(c)) {
            dst[j++] = c;
        } else {
            dst[j++] = '\\';
            dst[j++] = 'x';
            dst[j++] = ToHex(c >> 4);
            dst[j++] = ToHex(c & 0xF);
        }
    }
    return dst.substr(0, j);
}

void SplitString(const string& str,
                 const string& delim,
                 std::vector<string>* result) {
    result->clear();
    if (str.empty()) {
        return;
    }
    if (delim[0] == '\0') {
        result->push_back(str);
        return;
    }

    string::size_type delim_length = delim.length();

    for (string::size_type begin_index = 0; begin_index < str.size();) {
        string::size_type end_index = str.find(delim, begin_index);
        if (end_index == string::npos) {
            result->push_back(str.substr(begin_index));
            return;
        }
        if (end_index > begin_index) {
            result->push_back(str.substr(begin_index, (end_index - begin_index)));
        }

        begin_index = end_index + delim_length;
    }
}

std::vector<string> StringSplit(const string& arg, char delim) {
  std::vector<string> splits;
  stringstream ss(arg);
  string item;
  while (std::getline(ss, item, delim)) {
    splits.push_back(item);
  }
  return splits;
}

static const uint32_t MAX_PATH_LENGTH = 10240;
static bool SplitPath(const string& path,
                      std::vector<string>* element,
                      bool* isdir) {
    if (path.empty() || path[0] != '/' || path.size() > MAX_PATH_LENGTH) {
        return false;
    }
    element->clear();
    size_t last_pos = 0;
    for (size_t i = 1; i <= path.size(); i++) {
        if (i == path.size() || path[i] == '/') {
            if (last_pos + 1 < i) {
                element->push_back(path.substr(last_pos + 1, i - last_pos - 1));
            }
            last_pos = i;
        }
    }
    if (isdir) {
        *isdir = (path[path.size() - 1] == '/');
    }
    return true;
}

bool SplitAndParseAsInts(StringPiece text, char delim,
                         std::vector<int32>* result) {
  return SplitAndParseAsInts<int32>(text, delim, strings::safe_strto32, result);
}

bool SplitAndParseAsInts(StringPiece text, char delim,
                         std::vector<int64>* result) {
  return SplitAndParseAsInts<int64>(text, delim, strings::safe_strto64, result);
}

bool SplitAndParseAsFloats(StringPiece text, char delim,
                           std::vector<float>* result) {
  return SplitAndParseAsInts<float>(text, delim,
                                    [](StringPiece str, float* value) {
                                      return strings::safe_strtof(
                                          str.ToString().c_str(), value);
                                    },
                                    result);
}

}  // namespace str_util
}  // namespace bubblefs
