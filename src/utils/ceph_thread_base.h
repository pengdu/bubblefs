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

// ceph/src/common/Thread.h

#ifndef BUBBLEFS_UTILS_CEPH_THREAD_BASE_H_
#define BUBBLEFS_UTILS_CEPH_THREAD_BASE_H_

#include <sys/types.h>
#include <pthread.h>

namespace bubblefs {
namespace myceph {

class ThreadBase {
 private:
  pthread_t thread_id;
  pid_t pid;
  int ioprio_class, ioprio_priority;
  int cpuid;
  const char *thread_name;

  void *EntryWrapper();

 public:
  ThreadBase(const ThreadBase&) = delete;
  ThreadBase& operator=(const ThreadBase&) = delete;

  ThreadBase();
  virtual ~ThreadBase();

 protected:
  virtual void *Entry() = 0;

 private:
  static void *_entry_func(void *arg);

 public:
  const pthread_t &GetThreadId() const;
  pid_t GetPid() const { return pid; }
  bool IsStarted() const;
  bool AmSelf() const;
  bool Kill(int signal);
  bool TryCreate(size_t stacksize);
  bool Create(const char *name, size_t stacksize = 0);
  bool Join(void **prval = 0);
  bool Detach();
  bool SetIoprio(int cls, int prio);
  bool SetAffinity(int cpuid);
};

} // namespace myceph
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CEPH_THREAD_BASE_H_