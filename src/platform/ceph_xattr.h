/*
 * Ceph - scalable distributed file system
 *
 * Copyright (C) 2011 Stanislav Sedov <stas@FreeBSD.org>
 *
 * This is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License version 2.1, as published by the Free Software
 * Foundation.  See file COPYING.
 */

// ceph/src/common/xattr.h

#include <sys/types.h>

#ifndef BUBBLEFS_PLATFORM_CEPH_XTATTR_H_
#define BUBBLEFS_PLATFORM_CEPH_XTATTR_H_

// Almost everyone defines ENOATTR, except for Linux,
// which does #define ENOATTR ENODATA.  It seems that occasionally that
// isn't defined, though, so let's make sure.
#ifndef ENOATTR
# define ENOATTR ENODATA
#endif

namespace bubblefs {
namespace myceph {
  
int fs_setxattr(const char *path, const char *name,
                const void *value, size_t size);
int fs_fsetxattr(int fd, const char *name, const void *value,
                 size_t size);
ssize_t fs_getxattr(const char *path, const char *name,
                    void *value, size_t size);
ssize_t fs_fgetxattr(int fd, const char *name, void *value,
                     size_t size);
ssize_t fs_listxattr(const char *path, char *list, size_t size);
ssize_t fs_flistxattr(int fd, char *list, size_t size);
int fs_removexattr(const char *path, const char *name);
int fs_fremovexattr(int fd, const char *name);

} // namespace myceph
} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_CEPH_XTATTR_H_