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

// ceph/src/common/Thread.cc

#include "platform/port.h"
#include "utils/thread_simple.h"

namespace bubblefs {
namespace bdcommon {
  
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

bool Thread::Start(Proc proc, void* arg, std::string name, size_t stack_size, bool joinable) {
  thread_name_ = name;
  pthread_attr_t attributes;
  pthread_attr_init(&attributes);
  // Pthreads are joinable by default, so only specify the detached
  // attribute if the thread should be non-joinable.
  if (!joinable) {
    pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
  }
  // Get a better default if available.
  if (stack_size <= 0)
    stack_size = 2 * (1 << 23);  // 2 times 8192K (the default stack size on Linux).
  pthread_attr_setstacksize(&attributes, stack_size);
        
  // The child thread will inherit our signal mask.
  // Set our signal mask to the set of signals we want to block ?
        
  int ret = pthread_create(&tid_, nullptr, proc, arg);
        
  pthread_attr_destroy(&attributes);
  return (ret == 0);
}
  
void Thread::SetIoprio(int cls, int prio) {
  ioprio_class_ = cls;
  ioprio_prio_ = prio;
}  

bool Thread::SetAffinity(int id)
{
  bool r = false;
  cpuid_ = id;
  if (pid_ && port::os_gettid() == pid_)
    r = _set_affinity(id);
  return r;
}

void Thread::set_thread_attrs() {
   pid_t p = port::os_gettid();
   if (p > 0)
     pid_ = p;
   if (pid_ &&
      ioprio_class_ >= 0 &&
      ioprio_prio_ >= 0) {
    port::os_ioprio_set(IOPRIO_WHO_PROCESS,
                        pid_,
                        IOPRIO_PRIO_VALUE(ioprio_class, ioprio_priority));
  }
  if (pid_ && cpuid_ >= 0)
    _set_affinity(cpuid_);

  pthread_setname_np(pthread_self(), thread_name_);
}
  
} // namespace bdcommon  
} // namespace bubblefs