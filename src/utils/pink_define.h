// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/include/pink_define.h

#ifndef BUBBLEFS_UTILS_PINK_DEFINE_H_
#define BUBBLEFS_UTILS_PINK_DEFINE_H_

#include <functional>
#include <iostream>
#include <map>

namespace bubblefs {
namespace mypink {

constexpr int MYPINK_MAX_CLIENTS = 10240;
constexpr int MYPINK_MAX_MESSAGE = 1024;
constexpr int MYPINK_NAME_LEN = 1024;

constexpr int kProtoMaxMessage = 64 * 1024 * 1024;  // 64MB
constexpr int kCommandHeaderLength = 4;

/*
 * The pb head and code length
 */
constexpr int MYPINK_COMMAND_HEADER_LENGTH = 4;
constexpr int MYPINK_COMMAND_CODE_LENGTH = 4;

/*
 * The socket block type
 */
enum BlockType {
  kBlock = 0,
  kNonBlock = 1,
};

enum EventStatus {
  kNone = 0,
  kReadable = 1,
  kWriteable = 2,
};

enum ConnStatus {
  kHeader = 0,
  kPacket = 1,
  kComplete = 2,
  kBuildObuf = 3,
  kWriteObuf = 4,
};

enum ReadStatus {
  kReadHalf = 0,
  kReadAll = 1,
  kReadError = 2,
  kReadClose = 3,
  kFullError = 4,
  kParseError = 5,
  kDealError = 6,
  kOk = 7,
};

enum WriteStatus {
  kWriteHalf = 0,
  kWriteAll = 1,
  kWriteError = 2,
};

enum RetCode {
  kSuccess = 0,
  kBindError = 1,
  kCreateThreadError = 2,
  kListenError = 3,
  kSetSockOptError = 4,
};

/*
 * define the redis protocol
 */
constexpr int REDIS_MAX_MESSAGE = 67108864;  // 64MB
constexpr int DEFAULT_WBUF_SIZE = 262144; // 256KB
constexpr int REDIS_IOBUF_LEN = 16384;
constexpr int REDIS_REQ_INLINE = 1;
constexpr int REDIS_REQ_MULTIBULK = 2;

/*
 * define the pink cron interval (ms)
 */
constexpr int MYPINK_CRON_INTERVAL = 1000;

/*
 * define the macro in PINK_conf
 */

constexpr int MYPINK_WORD_SIZE = 1024;
constexpr int MYPINK_LINE_SIZE = 1024;
constexpr int MYPINK_CONF_MAX_NUM = 1024;

/*
 * define common character
 */
//#define SPACE ' '
//#define COLON ':'
//#define SHARP '#'

}  // namespace mypink
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PINK_DEFINE_H_