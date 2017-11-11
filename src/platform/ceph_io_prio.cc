/*
 * Ceph - scalable distributed file system
 *
 * Copyright (C) 2012 Red Hat
 *
 * This is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1, as published by the Free Software
 * Foundation.  See file COPYING.
 *
 */

// ceph/src/common/io_priority.cc

#include "platform/ceph_io_prio.h"
#include <unistd.h>
#if defined(__FreeBSD__) || defined(__APPLE__)
#include <errno.h>
#endif
#ifdef __linux__
#include <sys/syscall.h>   /* For SYS_xxx definitions */
#endif
#include <algorithm>

namespace bubblefs {
namespace myceph {  
  
bool ioprio_set(int whence, int who, int ioprio)
{
  int ret = syscall(SYS_ioprio_set, whence, who, ioprio);
  return (0 == ret);
}

int ioprio_string_to_class(const std::string& s)
{
  std::string l = s;
  std::transform(l.begin(), l.end(), l.begin(), ::tolower);

  if (l == "idle")
    return IOPRIO_CLASS_IDLE;
  if (l == "be" || l == "besteffort" || l == "best effort")
    return IOPRIO_CLASS_BE;
  if (l == "rt" || l == "realtime" || l == "real time")
    return IOPRIO_CLASS_RT;
  return IOPRIO_CLASS_NONE;
}

} // namespace myceph
} // namespace bubblefs