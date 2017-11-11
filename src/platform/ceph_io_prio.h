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

// ceph/src/common/io_priority.h
//  <linux/iocontext.h>

#ifndef BUBBLEFS_PLATFORM_IO_PRIO_H_
#define BUBBLEFS_PLATFORM_IO_PRIO_H_

#include <string>

namespace bubblefs {
namespace myceph {
  
/*
 * These are the io priority groups as implemented by CFQ. RT is the realtime
 * class, it always gets premium service. BE is the best-effort scheduling
 * class, the default for any process. IDLE is the idle scheduling class, it
 * is only served when no one else is using the disk.
 */
enum {
    IOPRIO_CLASS_NONE = 0,
    IOPRIO_CLASS_RT,
    IOPRIO_CLASS_BE,
    IOPRIO_CLASS_IDLE,
};

/*
 * 8 best effort priority levels are supported
 */
constexpr int IOPRIO_BE_NR = 8;
 
enum {
    IOPRIO_WHO_PROCESS = 1,
    IOPRIO_WHO_PGRP,
    IOPRIO_WHO_USER,
};

/*
 * Gives us 8 prio classes with 13-bits of data for each class
 */
#define MYCEPH_IOPRIO_CLASS_SHIFT      (13)
#define MYCEPH_IOPRIO_PRIO_MASK        ((1UL << MYCEPH_IOPRIO_CLASS_SHIFT) - 1)
 
#define MYCEPH_IOPRIO_PRIO_CLASS(mask) ((mask) >> MYCEPH_IOPRIO_CLASS_SHIFT)
#define MYCEPH_IOPRIO_PRIO_DATA(mask)  ((mask) & MYCEPH_IOPRIO_PRIO_MASK)
#define MYCEPH_IOPRIO_PRIO_VALUE(class, data)  (((class) << MYCEPH_IOPRIO_CLASS_SHIFT) | data)
 
constexpr bool ioprio_is_valid(int mask) { return (MYCEPH_IOPRIO_PRIO_CLASS((mask)) != IOPRIO_CLASS_NONE); }
  
extern bool ioprio_set(int whence, int who, int ioprio);

extern int ioprio_string_to_class(const std::string& s);

} // namespace myceph
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_IO_PRIO_H_