/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors

// rocksdb/env/env_posix.cc
// tensorflow/tensorflow/core/platform/posix/posix_file_system.cc

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <thread>
#include <vector>
#include "platform/brpc_eintr_wrapper.h"
#include "platform/env.h"
#include "platform/error.h"
#include "platform/io_posix.h"
#include "platform/load_library.h"
#include "platform/logging.h"
#include "platform/macros.h"
#include "platform/port_posix.h"
#include "platform/types.h"
#include "utils/error_codes.h"
#include "utils/path.h"
#include "utils/threadpool_impl.h"
#include "utils/random.h"
#include "utils/status.h"
#include "utils/strcat.h"

#ifdef OS_LINUX
#include <linux/fs.h>
#include <linux/magic.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#endif // OS_LINUX

#if !defined(TMPFS_MAGIC)
#define TMPFS_MAGIC 0x01021994
#endif
#if !defined(XFS_SUPER_MAGIC)
#define XFS_SUPER_MAGIC 0x58465342
#endif
#if !defined(EXT4_SUPER_MAGIC)
#define EXT4_SUPER_MAGIC 0xEF53
#endif

// unimplented macros
#define IOSTATS_TIMER_GUARD(metric)
#define TEST_SYNC_POINT_CALLBACK(x, y)

namespace bubblefs {

using std::unique_ptr;
using std::shared_ptr;

#if defined(OS_BSD) || defined(OS_MACOSX) || defined(OS_NACL)
typedef struct stat stat_wrapper_t;
#else
typedef struct stat64 stat_wrapper_t;
#endif

// Bits and masks of the file permission.
enum FilePermissionBits {
  FILE_PERMISSION_MASK              = S_IRWXU | S_IRWXG | S_IRWXO,
  FILE_PERMISSION_USER_MASK         = S_IRWXU,
  FILE_PERMISSION_GROUP_MASK        = S_IRWXG,
  FILE_PERMISSION_OTHERS_MASK       = S_IRWXO,

  FILE_PERMISSION_READ_BY_USER      = S_IRUSR,
  FILE_PERMISSION_WRITE_BY_USER     = S_IWUSR,
  FILE_PERMISSION_EXECUTE_BY_USER   = S_IXUSR,
  FILE_PERMISSION_READ_BY_GROUP     = S_IRGRP,
  FILE_PERMISSION_WRITE_BY_GROUP    = S_IWGRP,
  FILE_PERMISSION_EXECUTE_BY_GROUP  = S_IXGRP,
  FILE_PERMISSION_READ_BY_OTHERS    = S_IROTH,
  FILE_PERMISSION_WRITE_BY_OTHERS   = S_IWOTH,
  FILE_PERMISSION_EXECUTE_BY_OTHERS = S_IXOTH,
};
  
namespace {  
  
// list of pathnames that are locked
static std::set<string> lockedFiles;
static port::Mutex mutex_lockedFiles;

static int LockOrUnlock(const string& fname, int fd, bool lock) {
  mutex_lockedFiles.Lock();
  if (lock) {
    // If it already exists in the lockedFiles set, then it is already locked,
    // and fail this lock attempt. Otherwise, insert it into lockedFiles.
    // This check is needed because fcntl() does not detect lock conflict
    // if the fcntl is issued by the same thread that earlier acquired
    // this lock.
    if (lockedFiles.insert(fname).second == false) {
      mutex_lockedFiles.Unlock();
      errno = ENOLCK;
      return -1;
    }
  } else {
    // If we are unlocking, then verify that we had locked it earlier,
    // it should already exist in lockedFiles. Remove it from lockedFiles.
    if (lockedFiles.erase(fname) != 1) {
      mutex_lockedFiles.Unlock();
      errno = ENOLCK;
      return -1;
    }
  }
  errno = 0;
  struct flock f;
  memset(&f, 0, sizeof(f));
  f.l_type = (lock ? F_WRLCK : F_UNLCK);
  f.l_whence = SEEK_SET;
  f.l_start = 0;
  f.l_len = 0;        // Lock/unlock entire file
  int value = fcntl(fd, F_SETLK, &f);
  if (value == -1 && lock) {
    // if there is an error in locking, then remove the pathname from lockedfiles
    lockedFiles.erase(fname);
  }
  mutex_lockedFiles.Unlock();
  return value;
}

class StdThread : public Thread {
 public:
  // name and thread_options are both ignored.
  StdThread(const ThreadOptions& thread_options, const string& name,
            std::function<void()> fn)
      : thread_(fn) {}
  ~StdThread() override { thread_.join(); }

 private:
  std::thread thread_;
};

class PosixEnv : public Env {
 public:
  PosixEnv();

  virtual ~PosixEnv() {
    for (const auto tid : threads_to_join_) {
      pthread_join(tid, nullptr);
    }
    for (int pool_id = 0; pool_id < Env::Priority::TOTAL; ++pool_id) {
      thread_pools_[pool_id].JoinAllThreads();
    }
    // Delete the thread_status_updater_ only when the current Env is not
    // Env::Default().  This is to avoid the free-after-use error when
    // Env::Default() is destructed while some other child threads are
    // still trying to update thread status.
    /*if (this != Env::Default()) {
      delete thread_status_updater_;
    }*/
  }

  void SetFD_CLOEXEC(int fd, const EnvOptions* options) {
    if ((options == nullptr || options->set_fd_cloexec) && fd > 0) {
      fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) | FD_CLOEXEC);
    }
  }

  virtual Status NewSequentialFile(const string& fname,
                                   unique_ptr<SequentialFile>* result,
                                   const EnvOptions& options) override {
    result->reset();
    int fd = -1;
    int flags = O_RDONLY;
    FILE* file = nullptr;

    if (options.use_direct_reads && !options.use_mmap_reads) {
#if !defined(OS_MACOSX) && !defined(OS_OPENBSD) && !defined(OS_SOLARIS)
      flags |= O_DIRECT;
#endif
    }

    do {
      IOSTATS_TIMER_GUARD(open_nanos);
      fd = open(fname.c_str(), flags, 0644);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
      return IOError("While opening a file for sequentially reading", fname,
                     errno);
    }

    SetFD_CLOEXEC(fd, &options);

    if (options.use_direct_reads && !options.use_mmap_reads) {
#ifdef OS_MACOSX
      if (fcntl(fd, F_NOCACHE, 1) == -1) {
        close(fd);
        return IOError("While fcntl NoCache", fname, errno);
      }
#endif
    } else {
      do {
        IOSTATS_TIMER_GUARD(open_nanos);
        file = fdopen(fd, "r");
      } while (file == nullptr && errno == EINTR);
      if (file == nullptr) {
        close(fd);
        return IOError("While opening file for sequentially read", fname,
                       errno);
      }
    }
    result->reset(new PosixSequentialFile(fname, file, fd, options));
    return Status::OK();
  }
  
  virtual Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override {
    Status s;
    int fd = open(fname.c_str(), O_RDONLY);
    if (fd < 0) {
      s = IOError(fname, errno);
    } else {
      result->reset(new PosixRandomAccessFile(fname, fd));
    }
    return s;
  }

  virtual Status NewRandomAccessFile(const string& fname,
                                     unique_ptr<RandomAccessFile>* result,
                                     const EnvOptions& options) override {
    result->reset();
    Status s;
    int fd;
    int flags = O_RDONLY;
    if (options.use_direct_reads && !options.use_mmap_reads) {
#if !defined(OS_MACOSX) && !defined(OS_OPENBSD) && !defined(OS_SOLARIS)
      flags |= O_DIRECT;
      TEST_SYNC_POINT_CALLBACK("NewRandomAccessFile:O_DIRECT", &flags);
#endif
    }

    do {
      IOSTATS_TIMER_GUARD(open_nanos);
      fd = open(fname.c_str(), flags, 0644);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
      return IOError("While open a file for random read", fname, errno);
    }
    SetFD_CLOEXEC(fd, &options);

    if (options.use_mmap_reads && sizeof(void*) >= 8) {
      // Use of mmap for random reads has been removed because it
      // kills performance when storage is fast.
      // Use mmap when virtual address-space is plentiful.
      uint64_t size;
      s = GetFileSize(fname, &size);
      if (s.ok()) {
        void* base = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (base != MAP_FAILED) {
          result->reset(new PosixMmapReadableFile(fd, fname, base,
                                                  size, options));
        } else {
          s = IOError("while mmap file for read", fname, errno);
        }
      }
      close(fd);
    } else {
      if (options.use_direct_reads && !options.use_mmap_reads) {
#ifdef OS_MACOSX
        if (fcntl(fd, F_NOCACHE, 1) == -1) {
          close(fd);
          return IOError("while fcntl NoCache", fname, errno);
        }
#endif
      }
      result->reset(new PosixRandomAccessFile(fname, fd, options));
    }
    return s;
  }
  
  virtual Status NewWritableFile(const string& fname,
                                 std::unique_ptr<WritableFile>* result) override {
    Status s;
    FILE* f = fopen(fname.c_str(), "w");
    if (f == nullptr) {
      s = IOError(fname, errno);
    } else {
      int fd = fileno(f);
      result->reset(new PosixWritableFile(fname, fd));
    }
    return s;
  }

  virtual Status OpenWritableFile(const string& fname,
                                  unique_ptr<WritableFile>* result,
                                  const EnvOptions& options,
                                  bool reopen = false) {
    result->reset();
    Status s;
    int fd = -1;
    int flags = (reopen) ? (O_CREAT | O_APPEND) : (O_CREAT | O_TRUNC);
    // Direct IO mode with O_DIRECT flag or F_NOCAHCE (MAC OSX)
    if (options.use_direct_writes && !options.use_mmap_writes) {
      // Note: we should avoid O_APPEND here due to ta the following bug:
      // POSIX requires that opening a file with the O_APPEND flag should
      // have no affect on the location at which pwrite() writes data.
      // However, on Linux, if a file is opened with O_APPEND, pwrite()
      // appends data to the end of the file, regardless of the value of
      // offset.
      // More info here: https://linux.die.net/man/2/pwrite
      flags |= O_WRONLY;
#if !defined(OS_MACOSX) && !defined(OS_OPENBSD) && !defined(OS_SOLARIS)
      flags |= O_DIRECT;
#endif
      TEST_SYNC_POINT_CALLBACK("NewWritableFile:O_DIRECT", &flags);
    } else if (options.use_mmap_writes) {
      // non-direct I/O
      flags |= O_RDWR;
    } else {
      flags |= O_WRONLY;
    }

    do {
      IOSTATS_TIMER_GUARD(open_nanos);
      fd = open(fname.c_str(), flags, 0644);
    } while (fd < 0 && errno == EINTR);

    if (fd < 0) {
      s = IOError("While open a file for appending", fname, errno);
      return s;
    }
    SetFD_CLOEXEC(fd, &options);

    if (options.use_mmap_writes) {
      if (!checkedDiskForMmap_) {
        // this will be executed once in the program's lifetime.
        // do not use mmapWrite on non ext-3/xfs/tmpfs systems.
        if (!SupportsFastAllocate(fname)) {
          forceMmapOff_ = true;
        }
        checkedDiskForMmap_ = true;
      }
    }
    if (options.use_mmap_writes && !forceMmapOff_) {
      result->reset(new PosixMmapFile(fname, fd, page_size_, options));
    } else if (options.use_direct_writes && !options.use_mmap_writes) {
#ifdef OS_MACOSX
      if (fcntl(fd, F_NOCACHE, 1) == -1) {
        close(fd);
        s = IOError("While fcntl NoCache an opened file for appending", fname,
                    errno);
        return s;
      }
#elif defined(OS_SOLARIS)
      if (directio(fd, DIRECTIO_ON) == -1) {
        if (errno != ENOTTY) { // ZFS filesystems don't support DIRECTIO_ON
          close(fd);
          s = IOError("While calling directio()", fname, errno);
          return s;
        }
      }
#endif
      result->reset(new PosixWritableFile(fname, fd, options));
    } else {
      // disable mmap writes
      EnvOptions no_mmap_writes_options = options;
      no_mmap_writes_options.use_mmap_writes = false;
      result->reset(new PosixWritableFile(fname, fd, no_mmap_writes_options));
    }
    return s;
  }

  virtual Status NewWritableFile(const string& fname,
                                 unique_ptr<WritableFile>* result,
                                 const EnvOptions& options) override {
    return OpenWritableFile(fname, result, options, false);
  }

  virtual Status ReopenWritableFile(const string& fname,
                                    unique_ptr<WritableFile>* result,
                                    const EnvOptions& options) override {
    return OpenWritableFile(fname, result, options, true);
  }

  virtual Status ReuseWritableFile(const string& fname,
                                   const string& old_fname,
                                   unique_ptr<WritableFile>* result,
                                   const EnvOptions& options) override {
    result->reset();
    Status s;
    int fd = -1;

    int flags = 0;
    // Direct IO mode with O_DIRECT flag or F_NOCAHCE (MAC OSX)
    if (options.use_direct_writes && !options.use_mmap_writes) {
      flags |= O_WRONLY;
#if !defined(OS_MACOSX) && !defined(OS_OPENBSD) && !defined(OS_SOLARIS)
      flags |= O_DIRECT;
#endif
      TEST_SYNC_POINT_CALLBACK("NewWritableFile:O_DIRECT", &flags);
    } else if (options.use_mmap_writes) {
      // mmap needs O_RDWR mode
      flags |= O_RDWR;
    } else {
      flags |= O_WRONLY;
    }

    do {
      IOSTATS_TIMER_GUARD(open_nanos);
      fd = open(old_fname.c_str(), flags, 0644);
    } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
      s = IOError("while reopen file for write", fname, errno);
      return s;
    }

    SetFD_CLOEXEC(fd, &options);
    // rename into place
    if (rename(old_fname.c_str(), fname.c_str()) != 0) {
      s = IOError("while rename file to " + fname, old_fname, errno);
      close(fd);
      return s;
    }

    if (options.use_mmap_writes) {
      if (!checkedDiskForMmap_) {
        // this will be executed once in the program's lifetime.
        // do not use mmapWrite on non ext-3/xfs/tmpfs systems.
        if (!SupportsFastAllocate(fname)) {
          forceMmapOff_ = true;
        }
        checkedDiskForMmap_ = true;
      }
    }
    if (options.use_mmap_writes && !forceMmapOff_) {
      result->reset(new PosixMmapFile(fname, fd, page_size_, options));
    } else if (options.use_direct_writes && !options.use_mmap_writes) {
#ifdef OS_MACOSX
      if (fcntl(fd, F_NOCACHE, 1) == -1) {
        close(fd);
        s = IOError("while fcntl NoCache for reopened file for append", fname,
                    errno);
        return s;
      }
#elif defined(OS_SOLARIS)
      if (directio(fd, DIRECTIO_ON) == -1) {
        if (errno != ENOTTY) { // ZFS filesystems don't support DIRECTIO_ON
          close(fd);
          s = IOError("while calling directio()", fname, errno);
          return s;
        }
      }
#endif
      result->reset(new PosixWritableFile(fname, fd, options));
    } else {
      // disable mmap writes
      EnvOptions no_mmap_writes_options = options;
      no_mmap_writes_options.use_mmap_writes = false;
      result->reset(new PosixWritableFile(fname, fd, no_mmap_writes_options));
    }
    return s;

    return s;
  }

  virtual Status NewRandomRWFile(const string& fname,
                                 unique_ptr<RandomRWFile>* result,
                                 const EnvOptions& options) override {
    int fd = -1;
    while (fd < 0) {
      IOSTATS_TIMER_GUARD(open_nanos);
      fd = open(fname.c_str(), O_CREAT | O_RDWR, 0644);
      if (fd < 0) {
        // Error while opening the file
        if (errno == EINTR) {
          continue;
        }
        return IOError("While open file for random read/write", fname, errno);
      }
    }

    SetFD_CLOEXEC(fd, &options);
    result->reset(new PosixRandomRWFile(fname, fd, options));
    return Status::OK();
  }
  
  virtual Status NewAppendableFile(
    const string& fname, std::unique_ptr<WritableFile>* result) override {
    Status s;
    FILE* f = fopen(fname.c_str(), "a");
    if (f == nullptr) {
      s = IOError(fname, errno);
    } else {
      int fd = fileno(f);
      result->reset(new PosixWritableFile(fname, fd));
    }
    return s;
  }
  
  virtual Status NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    Status s = Status::OK();
    int fd = open(fname.c_str(), O_RDONLY);
    if (fd < 0) {
      s = IOError(fname, errno);
    } else {
      struct stat st;
      ::fstat(fd, &st);
      const void* address =
          mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (address == MAP_FAILED) {
        s = IOError(fname, errno);
      } else {
        result->reset(new PosixReadOnlyMemoryRegion(address, st.st_size));
      }
      close(fd);
    }
    return s;
  }

  virtual Status NewDirectory(const string& name,
                              unique_ptr<Directory>* result) override {
    result->reset();
    int fd;
    {
      IOSTATS_TIMER_GUARD(open_nanos);
      fd = open(name.c_str(), 0);
    }
    if (fd < 0) {
      return IOError("While open directory", name, errno);
    } else {
      result->reset(new PosixDirectory(fd));
    }
    return Status::OK();
  }

  virtual Status FileExists(const string& fname) override {
    if (access(fname.c_str(), F_OK) == 0) {
    return Status::OK();
    }
    return errors::NotFound(fname, " not found");
  }

  virtual Status GetChildren(const string& dir,
                             std::vector<string>* result) override {
    result->clear();
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) {
      switch (errno) {
        case EACCES:
        case ENOENT:
        case ENOTDIR:
          return Status::NotFound();
        default:
          return IOError("While opendir", dir, errno);
      }
    }
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
      StringPiece basename = entry->d_name;
      if ((basename != ".") && (basename != "..")) {
        result->push_back(entry->d_name);
      }
    }
    closedir(d);
    return Status::OK();
  }
  
  virtual bool MatchPath(const string& path, const string& pattern) override {
    return fnmatch(pattern.c_str(), path.c_str(), FNM_PATHNAME) == 0;
  }

  virtual Status DeleteFile(const string& fname) override {
    Status result;
    if (unlink(fname.c_str()) != 0) {
      result = IOError("while unlink() file", fname, errno);
    }
    return result;
  };

  virtual Status CreateDir(const string& name) override {
    Status result;
    if (mkdir(name.c_str(), 0755) != 0) {
      result = IOError("While mkdir", name, errno);
    }
    return result;
  };

  virtual Status CreateDirIfMissing(const string& name) override {
    Status result;
    if (mkdir(name.c_str(), 0755) != 0) {
      if (errno != EEXIST) {
        result = IOError("While mkdir if missing", name, errno);
      } else if (!DirExists(name)) { // Check that name is actually a
                                     // directory.
        // Message is taken from mkdir
        result = Status::IOError("`"+name+"' exists but is not a directory");
      }
    }
    return result;
  };

  virtual Status DeleteDir(const string& name) override {
    Status result;
    if (rmdir(name.c_str()) != 0) {
      result = IOError("file rmdir", name, errno);
    }
    return result;
  };
  
  virtual Status Stat(const string& fname, FileStatistics* stats) override {
    Status s;
    struct stat sbuf;
    if (stat(fname.c_str(), &sbuf) != 0) {
      s = IOError(fname, errno);
    } else {
      stats->mode = sbuf.st_mode;
      stats->uid = sbuf.st_uid;
      stats->gid = sbuf.st_gid;
      stats->length = sbuf.st_size;
      stats->mtime_nsec = sbuf.st_mtime * 1e9;
      stats->is_directory = S_ISDIR(sbuf.st_mode);
    }
    return s;
  }

  virtual Status GetFileSize(const string& fname,
                             uint64_t* size) override {
    Status s;
    struct stat sbuf;
    if (stat(fname.c_str(), &sbuf) != 0) {
      *size = 0;
      s = IOError("while stat a file for size", fname, errno);
    } else {
      *size = sbuf.st_size;
    }
    return s;
  }
  
  virtual Status GetFileModificationTime(const string& fname,
                                         uint64_t* file_mtime) override {
    struct stat s;
    if (stat(fname.c_str(), &s) !=0) {
      return IOError("while stat a file for modification time", fname, errno);
    }
    *file_mtime = static_cast<uint64_t>(s.st_mtime);
    return Status::OK();
  }
  
  virtual Status RenameFile(const string& src,
                            const string& target) override {
    Status result;
    if (rename(src.c_str(), target.c_str()) != 0) {
      result = IOError("While renaming a file to " + target, src, errno);
    }
    return result;
  }

  virtual Status LinkFile(const string& src,
                          const string& target) override {
    Status result;
    if (link(src.c_str(), target.c_str()) != 0) {
      if (errno == EXDEV) {
        return Status::NotSupported("No cross FS links allowed");
      }
      result = IOError("while link file to " + target, src, errno);
    }
    return result;
  }

  virtual Status LockFile(const string& fname, FileLock** lock) override {
    *lock = nullptr;
    Status result;
    int fd;
    {
      IOSTATS_TIMER_GUARD(open_nanos);
      fd = open(fname.c_str(), O_RDWR | O_CREAT, 0644);
    }
    if (fd < 0) {
      result = IOError("while open a file for lock", fname, errno);
    } else if (LockOrUnlock(fname, fd, true) == -1) {
      result = IOError("While lock file", fname, errno);
      close(fd);
    } else {
      SetFD_CLOEXEC(fd, nullptr);
      PosixFileLock* my_lock = new PosixFileLock;
      my_lock->fd_ = fd;
      my_lock->filename = fname;
      *lock = my_lock;
    }
    return result;
  }

  virtual Status UnlockFile(FileLock* lock) override {
    PosixFileLock* my_lock = reinterpret_cast<PosixFileLock*>(lock);
    Status result;
    if (LockOrUnlock(my_lock->filename, my_lock->fd_, false) == -1) {
      result = IOError("unlock", my_lock->filename, errno);
    }
    close(my_lock->fd_);
    delete my_lock;
    return result;
  }
  
  virtual Status CallStat(const char* path, FileStatistics* stats) override {
    stat_wrapper_t sb;
    int ret = stat64(path, &sb);
    if (ret)
      return IOError("stat64", path, errno);
    stats->mode = sb.st_mode;
    stats->uid = sb.st_uid;
    stats->gid = sb.st_gid;
    stats->length = sb.st_size;
    stats->mtime_nsec = sb.st_mtime * 1e9;
    stats->is_directory = S_ISDIR(sb.st_mode);
    return Status::OK();
  }
  
  virtual Status CallLstat(const char* path, FileStatistics* stats) override {
    stat_wrapper_t sb;
    int ret = lstat64(path, &sb);
    if (ret)
      return IOError("lstat64", path, errno);
    stats->mode = sb.st_mode;
    stats->uid = sb.st_uid;
    stats->gid = sb.st_gid;
    stats->length = sb.st_size;
    stats->mtime_nsec = sb.st_mtime * 1e9;
    stats->is_directory = S_ISDIR(sb.st_mode);
    return Status::OK();
  }
  
  virtual Status RealPath(const string& path, string& real_path) override {
    char buf[PATH_MAX];
    if (!realpath(path.c_str(), buf))
      return IOError("realpath", path, errno);
    
    real_path = buf;
    return Status::OK();
  }
  
  virtual string MakeAbsoluteFilePath(const string& input) override {
    char full_path[PATH_MAX];
    if (realpath(input.c_str(), full_path) == NULL)
      return "";
    return full_path;
  }
  
  virtual Status SetPosixFilePermissions(const string& path, int mode) override {
    //ThreadRestrictions::AssertIOAllowed();
    DCHECK((mode & ~FILE_PERMISSION_MASK) == 0);
    FileStatistics stats;
    if (!CallStat(path.c_str(), &stats).ok())
      return IOError("set file permissions", path, errno);
    // Clears the existing permission bits, and adds the new ones.
    mode_t updated_mode_bits = stats.mode & ~FILE_PERMISSION_MASK;
    updated_mode_bits |= mode & FILE_PERMISSION_MASK;
    
    if (HANDLE_EINTR(chmod(path.c_str(), updated_mode_bits)) != 0)
      return IOError("chmod", path, errno);
    return Status::OK();
  }
  
  virtual bool IsLink(const string& file_path) override {
    FileStatistics stats;
    // If we can't lstat the file, it's safe to assume that the file won't at
    // least be a 'followable' link.
    if (!CallLstat(file_path.c_str(), &stats).ok())
      return false;
    if (S_ISLNK(stats.mode))
      return true;
    else
      return false;
  }
  
  virtual int WriteFileDescriptor(const int fd, const char* data, int size) override {
    // Allow for partial writes.
    ssize_t bytes_written_total = 0;
    for (ssize_t bytes_written_partial = 0; bytes_written_total < size;
         bytes_written_total += bytes_written_partial) {
      bytes_written_partial =
        HANDLE_EINTR(write(fd, data + bytes_written_total, size - bytes_written_total));
      if (bytes_written_partial < 0)
        return -1;
    }
    return bytes_written_total;
  }
  
  virtual FILE* OpenFile(const string& filename, const char* mode) override {
    //ThreadRestrictions::AssertIOAllowed();
    FILE* result = NULL;
    do {
      result = fopen(filename.c_str(), mode);
    } while (!result && errno == EINTR);
    return result;
  }
  
  virtual int ReadFile(const string& filename, char* data, int max_size) override {
    //ThreadRestrictions::AssertIOAllowed();
    int fd = HANDLE_EINTR(open(filename.c_str(), O_RDONLY));
    if (fd < 0)
      return -1;
    ssize_t bytes_read = HANDLE_EINTR(read(fd, data, max_size));
    if (IGNORE_EINTR(close(fd)) < 0)
      return -1;
    return bytes_read;
  }
  
  virtual int WriteFile(const string& filename, const char* data, int size) override {
    //ThreadRestrictions::AssertIOAllowed();
    int fd = HANDLE_EINTR(creat(filename.c_str(), 0640));
    if (fd < 0)
      return -1;
    int bytes_written = WriteFileDescriptor(fd, data, size);
    if (IGNORE_EINTR(close(fd)) < 0)
      return -1;
    return bytes_written;
  }
  
  virtual int  AppendToFile(const string& filename, const char* data, int size) override {
    //ThreadRestrictions::AssertIOAllowed();
    int fd = HANDLE_EINTR(open(filename.c_str(), O_WRONLY | O_APPEND));
    if (fd < 0)
      return -1;
    int bytes_written = WriteFileDescriptor(fd, data, size);
    if (IGNORE_EINTR(close(fd)) < 0)
      return -1;
    return bytes_written;
  }
  
  // Gets the current working directory for the process.
  virtual Status GetCurrentDirectory(string& dir) override {
    // getcwd can return ENOENT, which implies it checks against the disk.
    char system_buffer[PATH_MAX] = "";
    if (!getcwd(system_buffer, sizeof(system_buffer))) {
      NOTREACHED();
      return IOError("getcwd", errno);
    }
    dir = system_buffer;
    return Status::OK();
  }
  
  // Sets the current working directory for the process.
  Status SetCurrentDirectory(const string& path) override {
    int ret = chdir(path.c_str());
    if (ret)
      return IOError("chdir", path, errno);
    return Status::OK();
  }
  
  virtual Status CopyFileUnsafe(const string& from_path, const string& to_path) override {
    //ThreadRestrictions::AssertIOAllowed();
    int infile = HANDLE_EINTR(open(from_path.c_str(), O_RDONLY));
    if (infile < 0)
      return IOError("open file", from_path, errno);
    
    int outfile = HANDLE_EINTR(creat(to_path.c_str(), 0666));
    if (outfile < 0) {
      close(infile);
      return IOError("open file", to_path, errno);
    }
    
    const size_t kBufferSize = 32768;
    std::vector<char> buffer(kBufferSize);
    bool result = true;
    
    while (result) {
      ssize_t bytes_read = HANDLE_EINTR(read(infile, &buffer[0], buffer.size()));
      if (bytes_read < 0) {
        result = false;
        break;
      }
      if (bytes_read == 0)
        break;
      // Allow for partial writes
      ssize_t bytes_written_per_read = 0;
      do {
        ssize_t bytes_written_partial = HANDLE_EINTR(write(
          outfile,
          &buffer[bytes_written_per_read],
          bytes_read - bytes_written_per_read));
        if (bytes_written_partial < 0) {
          result = false;
          break;
        }
        bytes_written_per_read += bytes_written_partial;
      } while (bytes_written_per_read < bytes_read);
    }
    
    if (IGNORE_EINTR(close(infile)) < 0)
      result = false;
    if (IGNORE_EINTR(close(outfile)) < 0)
      result = false;
    
    if (false == result)
      return IOError("copy file", from_path + "->" + to_path, errno);
    return Status::OK();
  }
  
  virtual void GetLocalTempDirectories(std::vector<string>* list) override {
    list->clear();
    // Directories, in order of preference. If we find a dir that
    // exists, we stop adding other less-preferred dirs
    const char* candidates[] = {
        // Non-null only during unittest/regtest
        getenv("TEST_TMPDIR"),

        // Explicitly-supplied temp dirs
        getenv("TMPDIR"),
        getenv("TMP"),

        // If all else fails
        "/tmp",
    };

    for (const char* d : candidates) {
      if (!d || d[0] == '\0') continue;  // Empty env var

      // Make sure we don't surprise anyone who's expecting a '/'
      string dstr = d;
      if (dstr[dstr.size() - 1] != '/') {
        dstr += "/";
      }

      struct stat statbuf;
      if (!stat(d, &statbuf) && S_ISDIR(statbuf.st_mode) &&
          !access(dstr.c_str(), 0)) {
        // We found a dir that exists and is accessible - we're done.
        list->push_back(dstr);
        return;
      }
    }
  }

  virtual void Schedule(void (*function)(void* arg1), void* arg,
                        Priority pri = LOW, void* tag = nullptr,
                        void (*unschedFunction)(void* arg) = 0) override;

  virtual int UnSchedule(void* arg, Priority pri) override;

  virtual void StartThread(void (*function)(void* arg), void* arg) override;

  virtual void WaitForJoin() override;

  virtual unsigned int GetThreadPoolQueueLen(Priority pri = LOW) const override;

  virtual Status GetTestDirectory(string* result) override {
    const char* env = getenv("TEST_TMPDIR");
    if (env && env[0] != '\0') {
      *result = env;
    } else {
      char buf[100];
      snprintf(buf, sizeof(buf), "/tmp/bubblefs-%d", int(geteuid()));
      *result = buf;
    }
    // Directory may already exist
    CreateDir(*result);
    return Status::OK();
  }

  static uint64_t gettid(pthread_t tid) {
    uint64_t thread_id = 0;
    memcpy(&thread_id, &tid, std::min(sizeof(thread_id), sizeof(tid)));
    return thread_id;
  }

  static uint64_t gettid() {
    pthread_t tid = pthread_self();
    return gettid(tid);
  }

  virtual uint64_t GetThreadID() const override {
    return gettid(pthread_self());
  }

  virtual uint64_t NowMicros() override {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
  }

  virtual uint64_t NowNanos() override {
#if defined(OS_LINUX) || defined(OS_FREEBSD) || defined(OS_AIX)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
#elif defined(OS_SOLARIS)
    return gethrtime();
#elif defined(__MACH__)
    clock_serv_t cclock;
    mach_timespec_t ts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &ts);
    mach_port_deallocate(mach_task_self(), cclock);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
       std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
  }

  virtual void SleepForMicroseconds(int64 micros) override {
    while (micros > 0) {
      timespec sleep_time;
      sleep_time.tv_sec = 0;
      sleep_time.tv_nsec = 0;

      if (micros >= 1e6) {
        sleep_time.tv_sec =
            std::min<int64>(micros / 1e6, std::numeric_limits<time_t>::max());
        micros -= static_cast<int64>(sleep_time.tv_sec) * 1e6;
      }
      if (micros < 1e6) {
        sleep_time.tv_nsec = 1000 * micros;
        micros = 0;
      }
      while (nanosleep(&sleep_time, &sleep_time) != 0 && errno == EINTR) {
        // Ignore signals and wait for the full interval to elapse.
      }
    }
  }

  virtual Status GetHostName(char* name, uint64_t len) override {
    int ret = gethostname(name, static_cast<size_t>(len));
    if (ret < 0) {
      if (errno == EFAULT || errno == EINVAL)
        return Status::InvalidArgument(strerror(errno));
      else
        return IOError("GetHostName", name, errno);
    }
    return Status::OK();
  }

  virtual Status GetCurrentTime(int64_t* unix_time) override {
    time_t ret = time(nullptr);
    if (ret == (time_t) -1) {
      return IOError("GetCurrentTime", "", errno);
    }
    *unix_time = (int64_t) ret;
    return Status::OK();
  }

  virtual Status GetAbsolutePath(const string& db_path,
                                 string* output_path) override {
    if (db_path.find('/') == 0) {
      *output_path = db_path;
      return Status::OK();
    }

    char the_path[256];
    char* ret = getcwd(the_path, 256);
    if (ret == nullptr) {
      return Status::IOError(strerror(errno));
    }

    *output_path = ret;
    return Status::OK();
  }

  // Allow increasing the number of worker threads.
  virtual void SetBackgroundThreads(int num, Priority pri) override {
    assert(pri >= Priority::BOTTOM && pri <= Priority::HIGH);
    thread_pools_[pri].SetBackgroundThreads(num);
  }

  virtual int GetBackgroundThreads(Priority pri) override {
    assert(pri >= Priority::BOTTOM && pri <= Priority::HIGH);
    return thread_pools_[pri].GetBackgroundThreads();
  }

  // Allow increasing the number of worker threads.
  virtual void IncBackgroundThreadsIfNeeded(int num, Priority pri) override {
    assert(pri >= Priority::BOTTOM && pri <= Priority::HIGH);
    thread_pools_[pri].IncBackgroundThreadsIfNeeded(num);
  }

  virtual void LowerThreadPoolIOPriority(Priority pool = LOW) override {
    assert(pool >= Priority::BOTTOM && pool <= Priority::HIGH);
#ifdef OS_LINUX
    thread_pools_[pool].LowerIOPriority();
#endif
  }

  virtual string TimeToString(uint64_t secondsSince1970) override {
    const time_t seconds = (time_t)secondsSince1970;
    struct tm t;
    int maxsize = 64;
    string dummy;
    dummy.reserve(maxsize);
    dummy.resize(maxsize);
    char* p = &dummy[0];
    localtime_r(&seconds, &t);
    snprintf(p, maxsize,
             "%04d/%02d/%02d-%02d:%02d:%02d ",
             t.tm_year + 1900,
             t.tm_mon + 1,
             t.tm_mday,
             t.tm_hour,
             t.tm_min,
             t.tm_sec);
    return dummy;
  }
  
  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      std::function<void()> fn) override {
    return new StdThread(thread_options, name, fn);
  }

  void SchedClosure(std::function<void()> closure) override {
    // TODO(b/27290852): Spawning a new thread here is wasteful, but
    // needed to deal with the fact that many `closure` functions are
    // blocking in the current codebase.
    std::thread closure_thread(closure);
    closure_thread.detach();
  }

  void SchedClosureAfter(int64 micros, std::function<void()> closure) override {
    // TODO(b/27290852): Consuming a thread here is wasteful, but this
    // code is (currently) only used in the case where a step fails
    // (AbortStep). This could be replaced by a timer thread
    SchedClosure([this, micros, closure]() {
      SleepForMicroseconds(micros);
      closure();
    });
  }

  Status LoadLibrary(const char* library_filename, void** handle) override {
    return internal::LoadLibrary(library_filename, handle);
  }

  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              void** symbol) override {
    return internal::GetSymbolFromLibrary(handle, symbol_name, symbol);
  }

  string FormatLibraryFileName(const string& name,
                               const string& version) override {
    return internal::FormatLibraryFileName(name, version);
  }

 private:
  bool checkedDiskForMmap_;
  bool forceMmapOff_;  // do we override Env options?

  // Returns true iff the named directory exists and is a directory.
  virtual bool DirExists(const string& dname) {
    struct stat statbuf;
    if (stat(dname.c_str(), &statbuf) == 0) {
      return S_ISDIR(statbuf.st_mode);
    }
    return false; // stat() failed return false
  }

  bool SupportsFastAllocate(const string& path) {
#ifdef OS_LINUX
    struct statfs s;
    if (statfs(path.c_str(), &s)){
      return false;
    }
    switch (s.f_type) {
      case EXT4_SUPER_MAGIC:
        return true;
      case XFS_SUPER_MAGIC:
        return true;
      case TMPFS_MAGIC:
        return true;
      default:
        return false;
    }
#else
    return false;
#endif
  }

  size_t page_size_;

  std::vector<ThreadPoolImpl> thread_pools_;
  pthread_mutex_t mu_;
  std::vector<pthread_t> threads_to_join_;
}; // PosixEnv

// PosixEnv impl
PosixEnv::PosixEnv()
    : checkedDiskForMmap_(false),
      forceMmapOff_(false),
      page_size_(getpagesize()),
      thread_pools_(Priority::TOTAL) {
  ThreadPoolImpl::PthreadCall("mutex_init", pthread_mutex_init(&mu_, nullptr));
  for (int pool_id = 0; pool_id < Env::Priority::TOTAL; ++pool_id) {
    thread_pools_[pool_id].SetThreadPriority(
        static_cast<Env::Priority>(pool_id));
    // This allows later initializing the thread-local-env of each thread.
    thread_pools_[pool_id].SetHostEnv(this);
  }
  //thread_status_updater_ = CreateThreadStatusUpdater();
}

void PosixEnv::Schedule(void (*function)(void* arg1), void* arg, Priority pri,
                        void* tag, void (*unschedFunction)(void* arg)) {
  assert(pri >= Priority::BOTTOM && pri <= Priority::HIGH);
  thread_pools_[pri].Schedule(function, arg, tag, unschedFunction);
}

int PosixEnv::UnSchedule(void* arg, Priority pri) {
  return thread_pools_[pri].UnSchedule(arg);
}

unsigned int PosixEnv::GetThreadPoolQueueLen(Priority pri) const {
  assert(pri >= Priority::BOTTOM && pri <= Priority::HIGH);
  return thread_pools_[pri].GetQueueLen();
}

struct StartThreadState {
  void (*user_function)(void*);
  void* arg;
};

static void* StartThreadWrapper(void* arg) {
  StartThreadState* state = reinterpret_cast<StartThreadState*>(arg);
  state->user_function(state->arg);
  delete state;
  return nullptr;
}

void PosixEnv::StartThread(void (*function)(void* arg), void* arg) {
  pthread_t t;
  StartThreadState* state = new StartThreadState;
  state->user_function = function;
  state->arg = arg;
  ThreadPoolImpl::PthreadCall(
      "start thread", pthread_create(&t, nullptr, &StartThreadWrapper, state));
  ThreadPoolImpl::PthreadCall("lock", pthread_mutex_lock(&mu_));
  threads_to_join_.push_back(t);
  ThreadPoolImpl::PthreadCall("unlock", pthread_mutex_unlock(&mu_));
}

void PosixEnv::WaitForJoin() {
  for (const auto tid : threads_to_join_) {
    pthread_join(tid, nullptr);
  }
  threads_to_join_.clear();
}

}  // namespace

//
// Default Posix Env
//
Env* Env::Default() {
  // The following function call initializes the singletons of ThreadLocalPtr
  // right before the static default_env.  This guarantees default_env will
  // always being destructed before the ThreadLocalPtr singletons get
  // destructed as C++ guarantees that the destructions of static variables
  // is in the reverse order of their constructions.
  //
  // Since static members are destructed in the reverse order
  // of their construction, having this call here guarantees that
  // the destructor of static PosixEnv will go first, then the
  // the singletons of ThreadLocalPtr.
  
  //ThreadLocalPtr::InitSingletons();
  static PosixEnv default_env;
  return &default_env;
}
  
} // namespace bubblefs
