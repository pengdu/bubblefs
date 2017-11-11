/*
 * Copyright (c) 2011 The LevelDB Authors.
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

#ifndef BUBBLEFS_PLATFORM_PDLFS_PORT_H_
#define BUBBLEFS_PLATFORM_PDLFS_PORT_H_

#include "platform/pdlfs_platform.h"

// Include the appropriate platform specific file below.
#if defined(PDLFS_PLATFORM_POSIX)
#include "platform/pdlfs_port_posix.h"
#endif

#endif // BUBBLEFS_PLATFORM_PDLFS_PORT_H_
