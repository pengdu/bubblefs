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

// ceph/src/common/pipe.c
// ceph/src/common/run_cmd.cpp

#include "platform/ceph_run_cmd.h"
#include <linux/limits.h>
#include <sys/wait.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sstream>
#include <vector>
#include "platform/macros.h"

namespace bubblefs {
namespace myceph {

using std::ostringstream;

bool pipe_cloexec(int pipefd[2])
{
  int ret;
  ret = pipe(pipefd);
  if (ret != 0)
    return false;

  /*
   * The old-fashioned, race-condition prone way that we have to fall
   * back on if O_CLOEXEC does not exist.
   */
  ret = fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);
  if (ret == -1) {
    ret = -errno;
    goto out;
  }

  ret = fcntl(pipefd[1], F_SETFD, FD_CLOEXEC);
  if (ret == -1) {
    ret = -errno;
    goto out;
  }

  return true;

out:
  VOID_TEMP_FAILURE_RETRY(close(pipefd[0]));
  VOID_TEMP_FAILURE_RETRY(close(pipefd[1]));

  return (0 == ret);
}

bool pipe2_cloexec(int pipefd[2])
{
  int ret;
  ret = pipe2(pipefd, O_CLOEXEC);
  return (0 == ret);
}

std::string run_cmd(const char *cmd, ...)
{
  std::vector <const char *> arr;
  va_list ap;
  va_start(ap, cmd);
  const char *c = cmd;
  do {
    arr.push_back(c);
    c = va_arg(ap, const char*);
  } while (c != NULL);
  va_end(ap);
  arr.push_back(NULL);

  int fret = fork();
  if (fret == -1) {
    int err = errno;
    ostringstream oss;
    oss << "run_cmd(" << cmd << "): unable to fork(): " << strerror(err);
    return oss.str();
  }
  else if (fret == 0) {
    // execvp doesn't modify its arguments, so the const-cast here is safe.
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);
    execvp(cmd, (char * const*)&arr[0]);
    _exit(127);
  }
  int status;
  while (waitpid(fret, &status, 0) == -1) {
    int err = errno;
    if (err == EINTR)
      continue;
    ostringstream oss;
    oss << "run_cmd(" << cmd << "): waitpid error: "
         << strerror(err);
    return oss.str();
  }
  if (WIFEXITED(status)) {
    int wexitstatus = WEXITSTATUS(status);
    if (wexitstatus != 0) {
      ostringstream oss;
      oss << "run_cmd(" << cmd << "): exited with status " << wexitstatus;
      return oss.str();
    }
    return "";
  }
  else if (WIFSIGNALED(status)) {
    ostringstream oss;
    oss << "run_cmd(" << cmd << "): terminated by signal";
    return oss.str();
  }
  ostringstream oss;
  oss << "run_cmd(" << cmd << "): terminated by unknown mechanism";
  return oss.str();
}

std::string GetSelfExeName() {
    char path[64]       = {0};
    char link[PATH_MAX] = {0};

    snprintf(path, sizeof(path), "/proc/%d/exe", getpid());
    readlink(path, link, sizeof(link));

    std::string filename(strrchr(link, '/') + 1);

    return filename;
}

bool ExecuteShellCmd(const std::string cmd, std::string* ret_str) {
    char output_buffer[80];
    FILE *fp = popen(cmd.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "fail to execute cmd: %s\n", cmd.c_str());
        return false;
    }
    fgets(output_buffer, sizeof(output_buffer), fp);
    pclose(fp);
    if (ret_str) {
        *ret_str = std::string(output_buffer);
    }
    return true;
}

std::string GetCurrentLocationDir() {
    char current_path[1024] = {'\0'};
    std::string current_dir;

    if (getcwd(current_path, 1024)) {
        current_dir = current_path;
    }
    return current_dir;
}

std::string GetLocalHostAddr() {
    std::string cmd =
        "/sbin/ifconfig | grep 'inet addr:'| grep -v '127.0.0.1' | cut -d: -f2 | awk '{ print $1}'";
    std::string addr;
    if (!ExecuteShellCmd(cmd, &addr)) {
        fprintf(stderr, "fail to fetch local host addr\n");
    } else if (addr.length() > 1) {
        addr.erase(addr.length() - 1, 1);
    }
    return addr;
}

std::string GetLocalHostName() {
    char str[kMaxHostNameSize + 1];
    if (0 != gethostname(str, kMaxHostNameSize + 1)) {
        fprintf(stderr, "gethostname fail\n");
        return "";
    }
    std::string hostname(str);
    return hostname;
}

} // namespace myceph
} // namespace bubblefs