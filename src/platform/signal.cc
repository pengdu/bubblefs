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

// ceph/src/common/signal.cc

#include "platform/signal.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <assert.h>
#include <signal.h>
#include <stdlib.h>
#include <sstream>

namespace bubblefs {
namespace port {

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

} // namespace port
} // namespace bubblefs