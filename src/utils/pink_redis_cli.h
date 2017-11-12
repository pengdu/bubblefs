// Copyright (c) 2015-present, Qihoo, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.

// pink/pink/include/redis_cli.h

#ifndef BUBBLEFS_UTILS_PINK_REDIS_CLI_H_
#define BUBBLEFS_UTILS_PINK_REDIS_CLI_H_

#include <string>
#include <vector>

namespace bubblefs {
namespace mypink {

// Usage like:
// std::string str;
// int ret = pink::SerializeRedisCommand(&str, "HSET %s %d", "key", i);
// RedisCmdArgsType vec;
// vec.push_back("hset"); vec.push_back("key"); vec.push_back(std::to_string(5));
// ret = pink::SerializeRedisCommand(vec, &str);
// PinkCli *rcli = NewRedisCli();
// rcli->set_connect_timeout(3000);
// Status s = rcli->Connect(ip, port, "101.199.114.205");
// ret = rcli->set_send_timeout(100);
// ret = rcli->set_recv_timeout(100);
// pink::RedisCmdArgsType redis_argv;
// std::string ping = "*1\r\n$4\r\nping\r\n";
// s = rcli->Send(&ping);
// s = rcli->Recv(&redis_argv);
// DeletePinkCli(&rcli);
  
typedef std::vector<std::string> RedisCmdArgsType;
// We can serialize redis command by 2 ways:
// 1. by variable argmuments;
//    eg.  RedisCli::Serialize(cmd, "set %s %d", "key", 5);
//        cmd will be set as the result string;
// 2. by a string vector;
//    eg.  RedisCli::Serialize(argv, cmd);
//        also cmd will be set as the result string.
extern int SerializeRedisCommand(std::string *cmd, const char *format, ...);
extern int SerializeRedisCommand(RedisCmdArgsType argv, std::string *cmd);

} // namespace mypink
} // namespace bubblefs

#endif  // BUBBLEFS_UTILS_PINK_REDIS_CLI_H_