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
/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// ceph/src/common/signal.cc
// caffe2/caffe2/utils/signal_handler.cc

#include "platform/ceph_signal.h"
#include <sys/syscall.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <assert.h>
#include <cxxabi.h>
#include <dirent.h>
#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <unwind.h>
#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

namespace bubblefs {
namespace myceph {
  
std::string signal_mask_to_str()
{
  sigset_t old_sigset;
  if (pthread_sigmask(SIG_SETMASK, NULL, &old_sigset)) {
    return "(pthread_signmask failed)";
  }

  std::ostringstream oss;
  oss << "show_signal_mask: { ";
  std::string sep("");
  for (int signum = 0; signum < NSIG; ++signum) {
    if (sigismember(&old_sigset, signum) == 1) {
      oss << sep << signum;
      sep = ", ";
    }
  }
  oss << " }";
  return oss.str();
}

/* Block the signals in 'siglist'. If siglist == NULL, block all signals. */
void block_signals(const int *siglist, sigset_t *old_sigset)
{
  sigset_t sigset;
  if (!siglist) {
    sigfillset(&sigset);
  }
  else {
    int i = 0;
    sigemptyset(&sigset);
    while (siglist[i]) {
      sigaddset(&sigset, siglist[i]);
      ++i;
    }
  }
  int ret = pthread_sigmask(SIG_BLOCK, &sigset, old_sigset);
  assert(ret == 0);
}

void restore_sigset(const sigset_t *old_sigset)
{
  int ret = pthread_sigmask(SIG_SETMASK, old_sigset, NULL);
  assert(ret == 0);
}

void unblock_all_signals(sigset_t *old_sigset)
{
  sigset_t sigset;
  sigfillset(&sigset);
  sigdelset(&sigset, SIGKILL);
  int ret = pthread_sigmask(SIG_UNBLOCK, &sigset, old_sigset);
  assert(ret == 0);
}

struct sigaction previousSighup;
struct sigaction previousSigint;
std::atomic<int> sigintCount(0);
std::atomic<int> sighupCount(0);
std::atomic<int> hookedUpCount(0);

void handleSignal(int signal) {
  switch (signal) {
    // TODO: what if the previous handler uses sa_sigaction?
    case SIGHUP:
      sighupCount += 1;
      if (previousSighup.sa_handler) {
        previousSighup.sa_handler(signal);
      }
      break;
    case SIGINT:
      sigintCount += 1;
      if (previousSigint.sa_handler) {
        previousSigint.sa_handler(signal);
      }
      break;
  }
}

void hookupHandler() {
  if (hookedUpCount++) {
    return;
  }
  struct sigaction sa;
  // Setup the handler
  sa.sa_handler = &handleSignal;
  // Restart the system call, if at all possible
  sa.sa_flags = SA_RESTART;
  // Block every signal during the handler
  sigfillset(&sa.sa_mask);
  // Intercept SIGHUP and SIGINT
  if (sigaction(SIGHUP, &sa, &previousSighup) == -1) {
    fprintf(stderr, "Cannot install SIGHUP handler.");
    abort();
  }
  if (sigaction(SIGINT, &sa, &previousSigint) == -1) {
    fprintf(stderr, "Cannot install SIGINT handler.");
    abort();
  }
}

// Set the signal handlers to the default.
void unhookHandler() {
  if (--hookedUpCount > 0) {
    return;
  }
  struct sigaction sa;
  // Setup the sighub handler
  sa.sa_handler = SIG_DFL;
  // Restart the system call, if at all possible
  sa.sa_flags = SA_RESTART;
  // Block every signal during the handler
  sigfillset(&sa.sa_mask);
  // Intercept SIGHUP and SIGINT
  if (sigaction(SIGHUP, &previousSighup, nullptr) == -1) {
    fprintf(stderr, "Cannot uninstall SIGHUP handler.");
    abort();
  }
  if (sigaction(SIGINT, &previousSigint, nullptr) == -1) {
    fprintf(stderr, "Cannot uninstall SIGINT handler.");
    abort();
  }
}

// The mutex protects the bool.
std::mutex fatalSignalHandlersInstallationMutex;
bool fatalSignalHandlersInstalled;
// We need to hold a reference to call the previous SIGUSR2 handler in case
// we didn't signal it
struct sigaction previousSigusr2;
// Flag dictating whether the SIGUSR2 handler falls back to previous handlers
// or is intercepted in order to print a stack trace.
std::atomic<bool> fatalSignalReceived(false);
// Global state set when a fatal signal is received so that backtracing threads
// know why they're printing a stacktrace.
const char* fatalSignalName("<UNKNOWN>");
int fatalSignum(-1);
// This wait condition is used to wait for other threads to finish writing
// their stack trace when in fatal sig handler (we can't use pthread_join
// because there's no way to convert from a tid to a pthread_t).
pthread_cond_t writingCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t writingMutex = PTHREAD_MUTEX_INITIALIZER;

struct {
  const char* name;
  int signum;
  struct sigaction previous;
} kSignalHandlers[] = {
  { "SIGABRT",  SIGABRT,  {} },
  { "SIGINT",   SIGINT,   {} },
  { "SIGILL",   SIGILL,   {} },
  { "SIGFPE",   SIGFPE,   {} },
  { "SIGBUS",   SIGBUS,   {} },
  { "SIGSEGV",  SIGSEGV,  {} },
  { nullptr,    0,        {} }
};  

struct sigaction* getPreviousSigaction(int signum) {
  for (auto handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (handler->signum == signum) {
      return &handler->previous;
    }
  }
  return nullptr;
}

const char* getSignalName(int signum) {
  for (auto handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (handler->signum == signum) {
      return handler->name;
    }
  }
  return nullptr;
}

_Unwind_Reason_Code unwinder(struct _Unwind_Context* context, void* userInfo) {
  auto& pcs = *reinterpret_cast<std::vector<uintptr_t>*>(userInfo);
  pcs.push_back(_Unwind_GetIP(context));
  return _URC_NO_REASON;
}

std::vector<uintptr_t> getBacktrace() {
  std::vector<uintptr_t> pcs;
  _Unwind_Backtrace(unwinder, &pcs);
  return pcs;
}

void printStacktrace() {
  std::vector<uintptr_t> pcs = getBacktrace();
  Dl_info info;
  size_t i = 0;
  for (uintptr_t pcAddr : pcs) {
    const void* pc = reinterpret_cast<const void*>(pcAddr);
    const char* path = nullptr;
    const char* name = "???";
    char* demangled = nullptr;
    int offset = -1;

    std::cerr << "[" << i << "] ";
    if (dladdr(pc, &info)) {
      path = info.dli_fname;
      name = info.dli_sname ?: "???";
      offset = reinterpret_cast<uintptr_t>(pc) -
          reinterpret_cast<uintptr_t>(info.dli_saddr);

      int status;
      demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
      if (status == 0) {
        name = demangled;
      }
    }
    std::cerr << name;
    if (offset >= 0) {
      std::cerr << "+" << reinterpret_cast<void*>(offset);
    }
    std::cerr << "(" << pc << ")";
    if (path) {
      std::cerr << " in " << path;
    }
    std::cerr << std::endl;
    if (demangled) {
      free(demangled);
    }
    i += 1;
  }
}

void callPreviousSignalHandler(
    struct sigaction* action,
    int signum,
    siginfo_t* info,
    void* ctx) {
  if (!action->sa_handler) {
    return;
  }
  if ((action->sa_flags & SA_SIGINFO) == SA_SIGINFO) {
    action->sa_sigaction(signum, info, ctx);
  } else {
    action->sa_handler(signum);
  }
}

// needsLock signals whether we need to lock our writing mutex.
void stacktraceSignalHandler(bool needsLock) {
  if (needsLock) {
    pthread_mutex_lock(&writingMutex);
  }
  pid_t tid = syscall(SYS_gettid);
  std::cerr << fatalSignalName << "(" << fatalSignum << "), Thread " << tid
            << ": " << std::endl;
  printStacktrace();
  std::cerr << std::endl;
  if (needsLock) {
    pthread_mutex_unlock(&writingMutex);
    pthread_cond_signal(&writingCond);
  }
}

// Our fatal signal entry point
void fatalSignalHandler(int signum) {
  // Check if this is a proper signal that we declared above.
  const char* name = getSignalName(signum);
  if (!name) {
    return;
  }
  if (fatalSignalReceived) {
    return;
  }
  // Set the flag so that our SIGUSR2 handler knows that we're aborting and
  // that it should intercept any SIGUSR2 signal.
  fatalSignalReceived = true;
  // Set state for other threads.
  fatalSignum = signum;
  fatalSignalName = name;
  // Linux doesn't have a nice userland API for enumerating threads so we
  // need to use the proc pseudo-filesystem.
  DIR* procDir = opendir("/proc/self/task");
  if (procDir) {
    pid_t pid = getpid();
    pid_t currentTid = syscall(SYS_gettid);
    struct dirent* entry;
    pthread_mutex_lock(&writingMutex);
    while ((entry = readdir(procDir)) != nullptr) {
      if (entry->d_name[0] == '.') {
        continue;
      }
      pid_t tid = atoi(entry->d_name);
      // If we've found the current thread then we'll jump into the SIGUSR2
      // handler before calling pthread_cond_wait thus deadlocking, so branch
      // our directly to the backtrace handler instead of signaling it.
      if (tid != currentTid) {
        syscall(SYS_tgkill, pid, tid, SIGUSR2);
        pthread_cond_wait(&writingCond, &writingMutex);
      } else {
        stacktraceSignalHandler(false);
      }
    }
    pthread_mutex_unlock(&writingMutex);
  } else {
    perror("Failed to open /proc/self/task");
  }
  sigaction(signum, getPreviousSigaction(signum), nullptr);
  raise(signum);
}

// Our SIGUSR2 entry point
void stacktraceSignalHandler(int signum, siginfo_t* info, void* ctx) {
  if (fatalSignalReceived) {
    stacktraceSignalHandler(true);
  } else {
    // We don't want to actually change the signal handler as we want to
    // remain the signal handler so that we may get the usr2 signal later.
    callPreviousSignalHandler(&previousSigusr2, signum, info, ctx);
  }
}

// Installs SIGABRT signal handler so that we get stack traces
// from every thread on SIGABRT caused exit. Also installs SIGUSR2 handler
// so that threads can communicate with each other (be sure if you use SIGUSR2)
// to install your handler before initing caffe2 (we properly fall back to
// the previous handler if we didn't initiate the SIGUSR2).
void installFatalSignalHandlers() {
  std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
  if (fatalSignalHandlersInstalled) {
    return;
  }
  fatalSignalHandlersInstalled = true;
  struct sigaction sa;
  sigemptyset(&sa.sa_mask);
  // Since we'll be in an exiting situation it's possible there's memory
  // corruption, so make our own stack just in case.
  sa.sa_flags = SA_ONSTACK | SA_SIGINFO;
  sa.sa_handler = fatalSignalHandler;
  for (auto* handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (sigaction(handler->signum, &sa, &handler->previous)) {
      std::string str("Failed to add ");
      str += handler->name;
      str += " handler!";
      perror(str.c_str());
    }
  }
  sa.sa_sigaction = stacktraceSignalHandler;
  if (sigaction(SIGUSR2, &sa, &previousSigusr2)) {
    perror("Failed to add SIGUSR2 handler!");
  }
}

void uninstallFatalSignalHandlers() {
  std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
  if (!fatalSignalHandlersInstalled) {
    return;
  }
  fatalSignalHandlersInstalled = false;
  for (auto* handler = kSignalHandlers; handler->name != nullptr; handler++) {
    if (sigaction(handler->signum, &handler->previous, nullptr)) {
      std::string str("Failed to remove ");
      str += handler->name;
      str += " handler!";
      perror(str.c_str());
    } else {
      handler->previous = {};
    }
  }
  if (sigaction(SIGUSR2, &previousSigusr2, nullptr)) {
    perror("Failed to add SIGUSR2 handler!");
  } else {
    previousSigusr2 = {};
  }
}

void setPrintStackTracesOnFatalSignal(bool print) {
  if (print) {
    installFatalSignalHandlers();
  } else {
    uninstallFatalSignalHandlers();
  }
}
bool printStackTracesOnFatalSignal() {
  std::lock_guard<std::mutex> locker(fatalSignalHandlersInstallationMutex);
  return fatalSignalHandlersInstalled;
}

} // namespace myceph
} // namespace bubblefs