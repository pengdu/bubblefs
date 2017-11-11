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

// ceph/src/common/pipe.h
// ceph/src/common/run_cmd.h
// tera/src/utils/utils_cmd.cc

#ifndef BUBBLEFS_PLATFORM_CEPH_RUN_CMD_H_
#define BUBBLEFS_PLATFORM_CEPH_RUN_CMD_H_

#include <string>

namespace bubblefs {
namespace myceph {
  
const uint32_t kMaxHostNameSize = 255;
const std::string kUnknownIpAddr = "255.255.255.255:0000";
  
/** Create a pipe and set both ends to have F_CLOEXEC
 *
 * @param pipefd        pipe array, just as in pipe(2)
 * @return              0 on success, errno otherwise 
 */
bool pipe_cloexec(int pipefd[2]);

bool pipe2_cloexec(int pipefd[2]);
  
//
// Fork a command and run it. The shell will not be invoked and shell
// expansions will not be done.
// This function takes a variable number of arguments. The last argument must
// be NULL.
//
// Example:
//   run_cmd("rm", "-rf", "foo", NULL)
//
// Returns an empty string on success, and an error string otherwise.
//
std::string run_cmd(const char *cmd, ...);

/// @brief 获取当前程序名
std::string GetSelfExeName();

bool ExecuteShellCmd(const std::string cmd, std::string* ret_str);

std::string GetCurrentLocationDir();

std::string GetLocalHostAddr();

std::string GetLocalHostName();

} // namespace myceph
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_CEPH_RUN_CMD_H_