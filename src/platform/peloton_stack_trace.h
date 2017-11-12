//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//===----------------------------------------------------------------------===//
//
//                         Peloton
//
// stack_trace.h
//
// Identification: src/include/common/stack_trace.h
//
// Copyright (c) 2015-16, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

// rocksdb/port/stack_trace.h
// peloton/src/include/common/stack_trace.h

#ifndef BUBBLEFS_PLATFORM_PELOTON_STACK_TRACE_H_
#define BUBBLEFS_PLATFORM_PELOTON_STACK_TRACE_H_

#include <stdio.h>

namespace bubblefs {
namespace mypeloton {

// Install a signal handler to print callstack on the following signals:
// SIGILL SIGSEGV SIGBUS SIGABRT
// Currently supports linux only. No-op otherwise.
void InstallStackTraceHandler();

// Prints stack, skips skip_first_frames frames
void PrintStack(int first_frames_to_skip = 0);

void RegisterSignalHandlers();

void PrintStackTrace(FILE *out = stderr,
                     unsigned int max_frames = 63);

void SignalHandler(int signum);

}  // namespace mypeloton
}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_PELOTON_STACK_TRACE_H_