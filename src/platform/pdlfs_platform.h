/*
 * pdlfs_platform.h.in  platform-wide constants
 * 29-Jul-2016  chuck@ece.cmu.edu
 */

// pdlfs-common/include/pdlfs-common/pdlfs_platform.h.in

#ifndef BUBBLEFS_PLATFORM_PDLFS_PLATFORM_H_
#define BUBBLEFS_PLATFORM_PDLFS_PLATFORM_H_

/*
 * cmake will convert these to real preprocessor defines as part
 * of the configuration process.
 */
#define PDLFS_PLATFORM_POSIX
//#define PDLFS_OS_CYGWIN
//#define PDLFS_OS_DARWIN
//#define PDLFS_OS_MACOSX
//#define PDLFS_OS_CRAYLINUX
#define PDLFS_OS_LINUX
//#define PDLFS_OS_SOLARIS
//#define PDLFS_OS_FREEBSD
//#define PDLFS_OS_NETBSD
//#define PDLFS_OS_OPENBSD
//#define PDLFS_OS_HPUX

#define PDLFS_TARGET_OS_VERSION "@PDLFS_TARGET_OS_VERSION@"
#define PDLFS_TARGET_OS "@PDLFS_TARGET_OS@"
#define PDLFS_HOST_OS_VERSION "@PDLFS_HOST_OS_VERSION@"
#define PDLFS_HOST_OS "@PDLFS_HOST_OS@"

#endif // BUBBLEFS_PLATFORM_PDLFS_PLATFORM_H_