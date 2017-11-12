//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//

// rocksdb/port/stack_trace.cc
// peloton/src/common/stack_trace.cpp

#include "platform/peloton_stack_trace.h"
#include <cxxabi.h>
#include <execinfo.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <memory>

namespace bubblefs {
namespace mypeloton {

namespace {

const char* GetExecutableName() {
  static char name[1024];

  char link[1024];
  snprintf(link, sizeof(link), "/proc/%d/exe", getpid());
  auto read = readlink(link, name, sizeof(name) - 1);
  if (-1 == read) {
    return nullptr;
  } else {
    name[read] = 0;
    return name;
  }
}

void PrintStackTraceLine(const char* symbol, void* frame) {
  static const char* executable = GetExecutableName();
  if (symbol) {
    fprintf(stderr, "%s ", symbol);
  }
  if (executable) {
    // out source to addr2line, for the address translation
    const int kLineMax = 256;
    char cmd[kLineMax];
    snprintf(cmd, kLineMax, "addr2line %p -e %s -f -C 2>&1", frame, executable);
    auto f = popen(cmd, "r");
    if (f) {
      char line[kLineMax];
      while (fgets(line, sizeof(line), f)) {
        line[strlen(line) - 1] = 0;  // remove newline
        fprintf(stderr, "%s\t", line);
      }
      pclose(f);
    }
  } else {
    fprintf(stderr, " %p", frame);
  }

  fprintf(stderr, "\n");
}

}  // namespace

void PrintStack(int first_frames_to_skip) {
  const int kMaxFrames = 100;
  void* frames[kMaxFrames];

  auto num_frames = backtrace(frames, kMaxFrames);
  auto symbols = backtrace_symbols(frames, num_frames);

  for (int i = first_frames_to_skip; i < num_frames; ++i) {
    fprintf(stderr, "#%-2d  ", i - first_frames_to_skip);
    PrintStackTraceLine((symbols != nullptr) ? symbols[i] : nullptr, frames[i]);
  }
  free(symbols);
}

static void StackTraceHandler(int sig) {
  // reset to default handler
  signal(sig, SIG_DFL);
  fprintf(stderr, "Received signal %d (%s)\n", sig, strsignal(sig));
  // skip the top three signal handler related frames
  PrintStack(3);
  // re-signal to default handler (so we still get core dump if needed...)
  raise(sig);
}

void InstallStackTraceHandler() {
  // just use the plain old signal as it's simple and sufficient
  // for this use case
  signal(SIGILL, StackTraceHandler);
  signal(SIGSEGV, StackTraceHandler);
  signal(SIGBUS, StackTraceHandler);
  signal(SIGABRT, StackTraceHandler);
}

void SignalHandler(int signum) {
  // associate each signal with a signal name string.
  const char* name = NULL;
  switch (signum) {
    case SIGABRT:
      name = "SIGABRT";
      break;
    case SIGSEGV:
      name = "SIGSEGV";
      break;
    case SIGBUS:
      name = "SIGBUS";
      break;
    case SIGILL:
      name = "SIGILL";
      break;
    case SIGFPE:
      name = "SIGFPE";
      break;
  }

  // Notify the user which signal was caught. We use printf, because this is the
  // most basic output function. Once you get a crash, it is possible that more
  // complex output systems like streams and the like may be corrupted. So we
  // make the most basic call possible to the lowest level, most
  // standard print function.
  if (name) {
    fprintf(stderr, "Caught signal %d (%s)", signum, name);
  } else {
    fprintf(stderr, "Caught signal %d", signum);
  }

  // Dump a stack trace.
  // This is the function we will be implementing next.
  PrintStackTrace();

  // If you caught one of the above signals, it is likely you just
  // want to quit your program right now.
  exit(signum);
}

// Based on :: http://panthema.net/2008/0901-stacktrace-demangled/
void PrintStackTrace(FILE *out, unsigned int max_frames) {
  fprintf(out, "Stack Trace:\n");

  /// storage array for stack trace address data
  void *addrlist[max_frames + 1];

  /// retrieve current stack addresses
  int addrlen = backtrace(addrlist, max_frames);

  if (addrlen == 0) {
    fprintf(out, "  <empty, possibly corrupt>\n");
    return;
  }

  /// resolve addresses into strings containing "filename(function+address)",
  char** symbol_list = backtrace_symbols(addrlist, addrlen);

  /// allocate string which will be filled with the demangled function name
  size_t func_name_size = 1024;
  std::unique_ptr<char> func_name(new char[func_name_size]);

  /// iterate over the returned symbol lines. skip the first, it is the
  /// address of this function.
  for (int i = 1; i < addrlen; i++) {
    char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

    /// find parentheses and +address offset surrounding the mangled name:
    /// ./module(function+0x15c) [0x8048a6d]
    for (char *p = symbol_list[i]; *p; ++p) {
      if (*p == '(')
        begin_name = p;
      else if (*p == '+')
        begin_offset = p;
      else if (*p == ')' && begin_offset) {
        end_offset = p;
        break;
      }
    }

    if (begin_name && begin_offset && end_offset &&
        begin_name < begin_offset) {
      *begin_name++ = '\0';
      *begin_offset++ = '\0';
      *end_offset = '\0';

      /// mangled name is now in [begin_name, begin_offset) and caller
      /// offset in [begin_offset, end_offset). now apply  __cxa_demangle():
      int status;
      char *ret = abi::__cxa_demangle(begin_name, func_name.get(), &func_name_size,
                                      &status);
      if (status == 0) {
        func_name.reset(ret);  // use possibly realloc()-ed string
        fprintf(out, "  %s : %s+%s\n", symbol_list[i], func_name.get(),
                  begin_offset);
      } else {
        /// demangling failed. Output function name as a C function with
        /// no arguments.
        fprintf(out, "  %s : %s()+%s\n", symbol_list[i], begin_name,
                  begin_offset);
      }
    } else {
      /// couldn't parse the line ? print the whole line.
      fprintf(out, "  %s\n", symbol_list[i]);
    }
  }

}

void RegisterSignalHandlers() {
  signal(SIGABRT, SignalHandler);
  signal(SIGSEGV, SignalHandler);
  signal(SIGILL, SignalHandler);
  signal(SIGFPE, SignalHandler);
}

}  // namespace mypeloton
}  // namespace bubblefs