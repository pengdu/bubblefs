//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

// rocksdb/port/port_posix.cc
// tensorflow/tensorflow/core/platform/posix/net.cc
// tensorflow/tensorflow/core/platform/posix/port.cc

#include <netinet/in.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <assert.h>
#include <errno.h>
#include <malloc.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <unordered_set>
#include "platform/logging.h"
#include "platform/port_posix.h"
#include "platform/time.h"
#include "utils/bits.h"
#include "utils/strcat.h"

#if defined(__i386__) || defined(__x86_64__)
#include <cpuid.h>
#endif

namespace bubblefs {
  
namespace internal {

namespace { // namespace anonymous
  
bool IsPortAvailable(int* port, bool is_tcp) {
  const int protocol = is_tcp ? IPPROTO_TCP : 0;
  const int fd = socket(AF_INET, is_tcp ? SOCK_STREAM : SOCK_DGRAM, protocol);

  struct sockaddr_in addr;
  socklen_t addr_len = sizeof(addr);
  int actual_port;

  CHECK_GE(*port, 0);
  CHECK_LE(*port, 65535);
  if (fd < 0) {
    LOG(ERROR) << "socket() failed: " << strerror(errno);
    return false;
  }

  // SO_REUSEADDR lets us start up a server immediately after it exists.
  int one = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) < 0) {
    LOG(ERROR) << "setsockopt() failed: " << strerror(errno);
    close(fd);
    return false;
  }

  // Try binding to port.
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(static_cast<uint16_t>(*port));
  if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
    LOG(WARNING) << "bind(port=" << *port << ") failed: " << strerror(errno);
    close(fd);
    return false;
  }

  // Get the bound port number.
  if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&addr), &addr_len) <
      0) {
    LOG(WARNING) << "getsockname() failed: " << strerror(errno);
    close(fd);
    return false;
  }
  CHECK_LE(addr_len, sizeof(addr));
  actual_port = ntohs(addr.sin_port);
  CHECK_GT(actual_port, 0);
  if (*port == 0) {
    *port = actual_port;
  } else {
    CHECK_EQ(*port, actual_port);
  }
  close(fd);
  return true;
}

const int kNumRandomPortsToPick = 100;
const int kMaximumTrials = 1000;

}  // namespace anonymous

int PickUnusedPortOrDie() {
  static std::unordered_set<int> chosen_ports;

  // Type of port to first pick in the next iteration.
  bool is_tcp = true;
  int trial = 0;
  while (true) {
    int port;
    trial++;
    CHECK_LE(trial, kMaximumTrials)
        << "Failed to pick an unused port for testing.";
    if (trial == 1) {
      port = getpid() % (65536 - 30000) + 30000;
    } else if (trial <= kNumRandomPortsToPick) {
      port = rand() % (65536 - 30000) + 30000;
    } else {
      port = 0;
    }

    if (chosen_ports.find(port) != chosen_ports.end()) {
      continue;
    }
    if (!IsPortAvailable(&port, is_tcp)) {
      continue;
    }

    CHECK_GT(port, 0);
    if (!IsPortAvailable(&port, !is_tcp)) {
      is_tcp = !is_tcp;
      continue;
    }

    chosen_ports.insert(port);
    return port;
  }

  return 0;
}

}  // namespace internal

namespace port {

unsigned page_size = sysconf(_SC_PAGESIZE);
unsigned long page_mask = ~(unsigned long)(page_size - 1);
unsigned page_shift = GetBitsOf(page_size - 1); 

// TODO: Make sure SIGURG is not used by user.
// This empty handler is simply for triggering EINTR in blocking syscalls.
void do_nothing_handler(int) {}

static pthread_once_t register_sigurg_once = PTHREAD_ONCE_INIT;

static void register_sigurg() {
    signal(SIGURG, do_nothing_handler);
}

int interrupt_pthread(pthread_t th) {
    pthread_once(&register_sigurg_once, register_sigurg);
    return pthread_kill(th, SIGURG);
}
  
int PhysicalCoreID() {
#if defined(__x86_64__) && (__GNUC__ > 2 || (__GNUC__ == 2 && __GNUC_MINOR__ >= 22))
  // sched_getcpu uses VDSO getcpu() syscall since 2.22. I believe Linux offers VDSO
  // support only on x86_64. This is the fastest/preferred method if available.
  int cpuno = sched_getcpu();
  if (cpuno < 0) {
    return -1;
  }
  return cpuno;
#elif defined(__x86_64__) || defined(__i386__)
  // clang/gcc both provide cpuid.h, which defines __get_cpuid(), for x86_64 and i386.
  unsigned eax, ebx = 0, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    return -1;
  }
  return ebx >> 24;
#else
  // give up, the caller can generate a random number or something.
  return -1;
#endif
}

void *cacheline_aligned_alloc(size_t size) {
#if __GNUC__ < 5 && defined(__SANITIZE_ADDRESS__)
  return malloc(size);
#elif defined(_ISOC11_SOURCE)
  return aligned_alloc(CACHE_LINE_SIZE, size);
#elif ( _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600 || defined(__APPLE__))
  void *m;
  errno = posix_memalign(&m, CACHE_LINE_SIZE, size);
  return errno ? NULL : m;
#else
  return malloc(size);
#endif
}

void cacheline_aligned_free(void *memblock) {
  free(memblock);
}

void Crash(const std::string& srcfile, int srcline) {
  fprintf(stdout, "Crashing at %s:%d\n", srcfile.c_str(), srcline);
  fflush(stdout);
  kill(getpid(), SIGTERM);
}

pid_t Gettid(void)
{
  return syscall(SYS_gettid); // __linux__
}

bool get_env_bool(const char *key)
{
  const char *val = getenv(key);
  if (!val)
    return false;
  if (strcasecmp(val, "off") == 0)
    return false;
  if (strcasecmp(val, "no") == 0)
    return false;
  if (strcasecmp(val, "false") == 0)
    return false;
  if (strcasecmp(val, "0") == 0)
    return false;
  return true;
}

int get_env_int(const char *key)
{
  const char *val = getenv(key);
  if (!val)
    return 0;
  int v = atoi(val);
  return v;
}

int GetMaxOpenFiles() {
#if defined(RLIMIT_NOFILE)
  struct rlimit no_files_limit;
  if (getrlimit(RLIMIT_NOFILE, &no_files_limit) != 0) {
    return -1;
  }
  // protect against overflow
  if (no_files_limit.rlim_cur >= std::numeric_limits<int>::max()) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(no_files_limit.rlim_cur);
#endif
  return -1;
}

double GetMemoryUsage() {
  FILE* fp = fopen("/proc/meminfo", "r");
  CHECK(fp) << "failed to fopen /proc/meminfo";
  size_t bufsize = 256 * sizeof(char);
  char* buf = new (std::nothrow) char[bufsize];
  CHECK(buf);
  int totalMem = -1;
  int freeMem = -1;
  int bufMem = -1;
  int cacheMem = -1;
  while (getline(&buf, &bufsize, fp) >= 0) {
    if (0 == strncmp(buf, "MemTotal", 8)) {
      if (1 != sscanf(buf, "%*s%d", &totalMem)) {
        LOG(FATAL) << "failed to get MemTotal from string: [" << buf << "]";
      }
    } else if (0 == strncmp(buf, "MemFree", 7)) {
      if (1 != sscanf(buf, "%*s%d", &freeMem)) {
        LOG(FATAL) << "failed to get MemFree from string: [" << buf << "]";
      }
    } else if (0 == strncmp(buf, "Buffers", 7)) {
      if (1 != sscanf(buf, "%*s%d", &bufMem)) {
        LOG(FATAL) << "failed to get Buffers from string: [" << buf << "]";
      }
    } else if (0 == strncmp(buf, "Cached", 6)) {
      if (1 != sscanf(buf, "%*s%d", &cacheMem)) {
        LOG(FATAL) << "failed to get Cached from string: [" << buf << "]";
      }
    }
    if (totalMem != -1 && freeMem != -1 && bufMem != -1 && cacheMem != -1) {
      break;
    }
  }
  CHECK(totalMem != -1 && freeMem != -1 && bufMem != -1 && cacheMem != -1)
      << "failed to get all information";
  fclose(fp);
  delete[] buf;
  double usedMem = 1.0 - 1.0 * (freeMem + bufMem + cacheMem) / totalMem;
  return usedMem;
}

int64_t AmountOfMemory(int pages_name) {
  long pages = sysconf(pages_name);
  long page_size = sysconf(_SC_PAGESIZE);
  if (pages == -1 || page_size == -1) {
    DCHECK(false);
    return 0;
  }
  return static_cast<int64_t>(pages) * page_size;
}

int64_t AmountOfPhysicalMemory() {
  return AmountOfMemory(_SC_PHYS_PAGES);
}

int64_t AmountOfVirtualMemory() {
  struct rlimit limit;
  int result = getrlimit(RLIMIT_DATA, &limit);
  if (result != 0) {
    DCHECK(false);
    return 0;
  }
  return limit.rlim_cur == RLIM_INFINITY ? 0 : limit.rlim_cur;
}

int NumberOfProcessors() {
  // It seems that sysconf returns the number of "logical" processors on both
  // Mac and Linux.  So we get the number of "online logical" processors.
  long res = sysconf(_SC_NPROCESSORS_ONLN);
  if (res == -1) {
    DCHECK(false);
    return 1;
  }

  return static_cast<int>(res);
}

std::string OperatingSystemName() {
  struct utsname info;
  if (uname(&info) < 0) {
    DCHECK(false);
    return std::string();
  }
  return std::string(info.sysname);
}

std::string OperatingSystemVersion() {
  struct utsname info;
  if (uname(&info) < 0) {
    DCHECK(false);
    return std::string();
  }
  return std::string(info.release);
}

std::string OperatingSystemArchitecture() {
  struct utsname info;
  if (uname(&info) < 0) {
    DCHECK(false);
    return std::string();
  }
  std::string arch(info.machine);
  if (arch == "i386" || arch == "i486" || arch == "i586" || arch == "i686") {
    arch = "x86";
  } else if (arch == "amd64") {
    arch = "x86_64";
  }
  return arch;
}

std::string Hostname() {
  char hostname[1024];
  if (0 != gethostname(hostname, sizeof hostname)) {
    return "";
  }
  hostname[sizeof hostname - 1] = 0;
  return std::string(hostname);
}

}  // namespace port

}  // namespace bubblefs