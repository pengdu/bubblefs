//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

// rocksdb/port/stack_trace.h

#ifndef BUBBLEFS_PLATFORM_STACK_TRACE_H_
#define BUBBLEFS_PLATFORM_STACK_TRACE_H_

namespace bubblefs {
namespace port {

// Install a signal handler to print callstack on the following signals:
// SIGILL SIGSEGV SIGBUS SIGABRT
// Currently supports linux only. No-op otherwise.
void InstallStackTraceHandler();

// Prints stack, skips skip_first_frames frames
void PrintStack(int first_frames_to_skip = 0);

}  // namespace port
}  // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_STACK_TRACE_H_