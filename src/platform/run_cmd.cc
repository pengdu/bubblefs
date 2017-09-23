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

#include <sys/wait.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sstream>
#include <vector>

namespace bubblefs {
namespace port {

using std::ostringstream;

# define VOID_TEMP_FAILURE_RETRY(expression) \
    static_cast<void>(TEMP_FAILURE_RETRY(expression))

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

constexpr int kExecuteCMDBufLength = 204800;  
  
int prun_cmd(const char* cmd, char* result) {
  char bufPs[kExecuteCMDBufLength];
  char ps[kExecuteCMDBufLength] = {0};
  FILE* ptr = nullptr;
  strncpy(ps, cmd, kExecuteCMDBufLength);
  if ((ptr = popen(ps, "r")) != nullptr) {
    size_t count = fread(bufPs, 1, kExecuteCMDBufLength, ptr);
    memcpy(result,
           bufPs,
           count - 1);  // why count-1: remove the '\n' at the end
    result[count] = 0;
    pclose(ptr);
    ptr = nullptr;
    return count - 1;
  } else {
    return -1;
  }
}

} // namespace port
} // namespace bubblefs