/*
 * Ceph - scalable distributed file system
 *
 * Copyright (C) 2004-2011 New Dream Network
 *
 * This is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1, as published by the Free Software
 * Foundation.  See file COPYING.
 *
 */

#include "utils/ceph_thread_base.h"
#include <assert.h>
#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <syscall.h>
#include <unistd.h>
#include "platform/ceph_signal.h"

namespace bubblefs {
namespace myceph {

static bool _set_affinity(int id)
{
  if (id >= 0 && id < CPU_SETSIZE) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    CPU_SET(id, &cpuset);

    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) < 0)
      return false;
    /* guaranteed to take effect immediately */
    sched_yield();
  }
  return true;
}

static pid_t os_gettid(void)
{
  return syscall(SYS_gettid); // __linux__
}

ThreadBase::ThreadBase()
  : thread_id(0),
    pid(0),
    ioprio_class(-1),
    ioprio_priority(-1),
    cpuid(-1),
    thread_name(NULL)
{
}

ThreadBase::~ThreadBase()
{
}

void *ThreadBase::_entry_func(void *arg) {
  void *r = ((ThreadBase*)arg)->EntryWrapper();
  return r;
}

void *ThreadBase::EntryWrapper()
{
  int p = os_gettid(); // may return -ENOSYS on other platforms
  if (p > 0)
    pid = p;
  if (pid && cpuid >= 0)
    _set_affinity(cpuid);

  pthread_setname_np(pthread_self(), thread_name);
  return Entry();
}

const pthread_t &ThreadBase::GetThreadId() const
{
  return thread_id;
}

bool ThreadBase::IsStarted() const
{
  return thread_id != 0;
}

bool ThreadBase::AmSelf() const
{
  return (pthread_self() == thread_id);
}

bool ThreadBase::Kill(int signal)
{
  int ret = 0;
  if (thread_id)
    ret = pthread_kill(thread_id, signal);
  return (0 == ret);
}

bool ThreadBase::TryCreate(size_t stacksize)
{
  pthread_attr_t *thread_attr = NULL;
  pthread_attr_t thread_attr_loc;
  
  if (stacksize) {
    thread_attr = &thread_attr_loc;
    pthread_attr_init(thread_attr);
    pthread_attr_setstacksize(thread_attr, stacksize);
  }

  int r;
  // The child thread will inherit our signal mask.  Set our signal mask to
  // the set of signals we want to block.  (It's ok to block signals more
  // signals than usual for a little while-- they will just be delivered to
  // another thread or delieverd to this thread later.)
  sigset_t old_sigset;
  int to_block[] = { SIGPIPE , 0 };
  block_signals(to_block, &old_sigset);
  r = pthread_create(&thread_id, thread_attr, _entry_func, (void*)this);
  restore_sigset(&old_sigset);

  if (thread_attr) {
    pthread_attr_destroy(thread_attr);  
  }

  return (0 == r);
}

bool ThreadBase::Create(const char *name, size_t stacksize)
{
  thread_name = name;
  bool ret = TryCreate(stacksize);
  return ret;
}

bool ThreadBase::Join(void **prval)
{
  if (thread_id == 0) {
    assert("join on thread that was never started" == 0);
    return false;
  }

  int ret = pthread_join(thread_id, prval);
  if (ret != 0)
    return false;

  thread_id = 0;
  return (0 == ret);
}

bool ThreadBase::Detach()
{
  int ret = pthread_detach(thread_id);
  return (0 == ret);
}

bool ThreadBase::SetIoprio(int cls, int prio)
{
  // fixme, maybe: this can race with create()
  ioprio_class = cls;
  ioprio_priority = prio;
  return true;
}

bool ThreadBase::SetAffinity(int id)
{
  bool r = false;
  cpuid = id;
  if (pid && os_gettid() == pid)
    r = _set_affinity(id);
  return r;
}

} // namespace myceph
} // namespace bubblefs