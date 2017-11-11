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

// ceph/src/common/signal.h

#ifndef BUBBLEFS_PLATFORM_CEPH_SIGNAL_H_
#define BUBBLEFS_PLATFORM_CEPH_SIGNAL_H_

#include <signal.h>
#include <string>

namespace bubblefs {
namespace myceph {
  
// Usage like: 
// sigset_t old_sigset;
// int to_block[] = { SIGPIPE , 0 };
// block_signals(to_block, &old_sigset);
// user_func()
// restore_sigset(&old_sigset);
  
// Returns a string showing the set of blocked signals for the calling thread.
// Other threads may have a different set (this is per-thread thing).
extern std::string signal_mask_to_str();

// Block a list of signals. If siglist == NULL, blocks all signals.
// If not, the list is terminated with a 0 element.
//
// On success, stores the old set of blocked signals in
// old_sigset. On failure, stores an invalid set of blocked signals in
// old_sigset.
extern void block_signals(const int *siglist, sigset_t *old_sigset);

// Restore the set of blocked signals. Will not restore an invalid set of
// blocked signals.
extern void restore_sigset(const sigset_t *old_sigset);

// Unblock all signals. On success, stores the old set of blocked signals in
// old_sigset. On failure, stores an invalid set of blocked signals in
// old_sigset.
extern void unblock_all_signals(sigset_t *old_sigset);

// This works by setting up certain fatal signal handlers. Previous fatal
// signal handlers will still be called when the signal is raised. Defaults
// to being off.
void setPrintStackTracesOnFatalSignal(bool print);
bool printStackTracesOnFatalSignal();

} // namespace myceph
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_SIGNAL_H_