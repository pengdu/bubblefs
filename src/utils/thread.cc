// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/threading/platform_thread_posix.cc

#include <errno.h>
#include <sched.h>
#include "platform/platform.h"
#include "utils/thread.h"

#if defined(OS_LINUX)
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace bubblefs {

namespace thread {


} // namespace thread

} // namespace bubblefs