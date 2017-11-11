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

// ceph/src/common/xattr.c

#include "platform/ceph_xattr.h"
#if defined(__FreeBSD__)
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/types.h>
#include <sys/extattr.h>
#elif defined(__linux__)
#include <sys/types.h>
#include <sys/xattr.h>
#elif defined(__APPLE__)
#include <errno.h>
#include <sys/xattr.h>
#else
#error "Your system is not supported!"
#endif

namespace bubblefs {
namespace myceph {
  
/*
 * Sets extended attribute on a file.
 * Returns 0 on success, -1 on failure.
 */
int
fs_setxattr(const char *path, const char *name,
    const void *value, size_t size)
{
        int error = -1;

#if defined(__FreeBSD__)
        error = extattr_set_file(path, EXTATTR_NAMESPACE_USER, name, value,
            size);
        if (error > 0)
                error = 0;
#elif defined(__linux__) 
        error = setxattr(path, name, value, size, 0);
#elif defined(__APPLE__)
        error = setxattr(path, name, value, size, 0 /* position */, 0);
#endif

        return (error);
}

int
fs_fsetxattr(int fd, const char *name, const void *value,
    size_t size)
{
        int error = -1;

#if defined(__FreeBSD__)
        error = extattr_set_fd(fd, EXTATTR_NAMESPACE_USER, name, value, size);
        if (error > 0)
                error = 0;
#elif defined(__linux__)
        error = fsetxattr(fd, name, value, size, 0);
#elif defined(__APPLE__)
        error = fsetxattr(fd, name, value, size, 0, 0 /* no options should be indentical to Linux */ );
#endif

        return (error);
}

ssize_t
fs_getxattr(const char *path, const char *name,
void *value, size_t size)
{
        ssize_t error = -1;

#if defined(__FreeBSD__)
        if (value == NULL || size == 0) {
                error = extattr_get_file(path, EXTATTR_NAMESPACE_USER, name, value,
                    size);
        } else {
                error = extattr_get_file(path, EXTATTR_NAMESPACE_USER, name, NULL,
                    0);
                if (error > 0) {
                        if (error > size) {
                                errno = ERANGE;
                                error = -1;
                        } else  {
                                error = extattr_get_file(path, EXTATTR_NAMESPACE_USER,
                                    name, value, size);
                        }
                }
        }
#elif defined(__linux__)
        error = getxattr(path, name, value, size);
#elif defined(__APPLE__)
        error = getxattr(path, name, value, size, 0 /* position  */, 0);
        /* ENOATTR and ENODATA have different values */
        if (error < 0 && errno == ENOATTR)
                errno = ENODATA;
#endif

        return (error);
}

ssize_t
fs_fgetxattr(int fd, const char *name, void *value,
    size_t size)
{
        ssize_t error = -1;

#if defined(__FreeBSD__)
        if (value == NULL || size == 0) {
                error = extattr_get_fd(fd, EXTATTR_NAMESPACE_USER, name, value,
                    size);
        } else {
                error = extattr_get_fd(fd, EXTATTR_NAMESPACE_USER, name, NULL,
                    0);
                if (error > 0) {
                        if (error > size) {
                                errno = ERANGE;
                                error = -1;
                        } else  {
                                error = extattr_get_fd(fd, EXTATTR_NAMESPACE_USER,
                                    name, value, size);
                        }
                }
        }
#elif defined(__linux__)
        error = fgetxattr(fd, name, value, size);
#elif defined(__APPLE__)
        error = fgetxattr(fd, name, value, size, 0, 0 /* no options */);
        /* ENOATTR and ENODATA have different values */
        if (error < 0 && errno == ENOATTR)
                errno = ENODATA;
#endif

        return (error);
}

ssize_t
fs_listxattr(const char *path, char *list, size_t size)
{
        ssize_t error = -1;

#if defined(__FreeBSD__)
        /*
         * XXX. The format of the list FreeBSD returns differs
         * from the Linux ones.  We have to perform the conversion. :-(
         */
        char *newlist, *p, *p1;

        if (size != 0) {
                newlist = malloc(size);
                if (newlist != NULL) {
                        error = extattr_list_file(path, EXTATTR_NAMESPACE_USER,
                            newlist, size);
                        if (error > 0) {
                                p = newlist;
                                p1 = list;
                                while ((p - newlist) < error) {
                                        uint8_t len = *(uint8_t *)p;
                                        p++;
                                        if ((p + len - newlist) > error)
                                                break;
                                        if (len > 0) {
                                                bcopy(p, p1, len);
                                                p += len;
                                                p1 += len;
                                                *p1++ = '\0';
                                        }
                                }
                                error = p1 - list;
                        }
                        free(newlist);
                }
        } else {
                error = extattr_list_file(path, EXTATTR_NAMESPACE_USER,
                    list, size);
        }
#elif defined(__linux__)
        error = listxattr(path, list, size);
#elif defined(__APPLE__)
        error = listxattr(path, list, size, 0);
#endif

        return (error);
}

ssize_t
fs_flistxattr(int fd, char *list, size_t size)
{
        ssize_t error = -1;

#if defined(__FreeBSD__)
        /*
         * XXX. The format of the list FreeBSD returns differs
         * from the Linux ones.  We have to perform the conversion. :-(
         */
        char *newlist, *p, *p1;

        if (size != 0) {
                newlist = malloc(size);
                if (newlist != NULL) {
                        error = extattr_list_fd(fd, EXTATTR_NAMESPACE_USER,
                            newlist, size);
                        if (error > 0) {
                                p = newlist;
                                p1 = list;
                                while ((p - newlist) < error) {
                                        uint8_t len = *(uint8_t *)p;
                                        p++;
                                        if ((p + len - newlist) > error)
                                                break;
                                        if (len > 0) {
                                                bcopy(p, p1, len);
                                                p += len;
                                                p1 += len;
                                                *p1++ = '\0';
                                        }
                                }
                                error = p1 - list;
                        }
                        free(newlist);
                }
        } else {
                error = extattr_list_fd(fd, EXTATTR_NAMESPACE_USER,
                    list, size);
        }
#elif defined(__linux__)
        error = flistxattr(fd, list, size);
#elif defined(__APPLE__)
        error = flistxattr(fd, list, size, 0);
#endif

        return (error);
}

int
fs_removexattr(const char *path, const char *name)
{
        int error = -1;

#if defined(__FreeBSD__)
        error = extattr_delete_file(path, EXTATTR_NAMESPACE_USER, name);
#elif defined(__linux__)
        error = removexattr(path, name);
#elif defined(__APPLE__)
        error = removexattr(path, name, 0);
        /* ENOATTR and ENODATA have different values */
        if (error < 0 && errno == ENOATTR)
                errno = ENODATA;
#endif

        return (error);
}

int
fs_fremovexattr(int fd, const char *name)
{
        int error = -1;

#if defined(__FreeBSD__)
        error = extattr_delete_fd(fd, EXTATTR_NAMESPACE_USER, name);
#elif defined(__linux__)
        error = fremovexattr(fd, name);
#elif defined(__APPLE__)
        error = fremovexattr(fd, name, 0);
        /* ENOATTR and ENODATA have different values */
        if (error < 0 && errno == ENOATTR)
                errno = ENODATA;
#endif

        return (error);
}

} // namespace myceph
} // namespace bubblefs