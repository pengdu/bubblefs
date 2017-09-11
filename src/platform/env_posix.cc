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

#include "platform/platform.h"
#ifdef OS_LINUX
#include <linux/fs.h>
#include <linux/magic.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#endif // OS_LINUX
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
#include "platform/env.h"
#include "platform/error.h"
#include "platform/io_posix.h"
#include "platform/logging.h"
#include "platform/port_posix.h"
#include "platform/types.h"
#include "utils/error_codes.h"
#include "utils/path.h"
#include "utils/random.h"
#include "utils/status.h"
#include "utils/strcat.h"

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

class PosixFileLock : public FileLock {
 public:
  int fd_;
  std::string filename;
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
    if (this != Env::Default()) {
      //delete thread_status_updater_;
    }
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
      result->reset(new PosixWritableFile(fname, f));
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
      result->reset(new PosixWritableFile(fname, f));
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
  
  bool MatchPath(const string& path, const string& pattern) override {
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

  virtual Status GetTestDirectory(std::string* result) override {
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

  virtual void SleepForMicroseconds(int micros) override { usleep(micros); }

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
};

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

std::string Env::GenerateUniqueId() {
  std::string uuid_file = "/proc/sys/kernel/random/uuid";

  Status s = FileExists(uuid_file);
  if (s.ok()) {
    std::string uuid;
    s = ReadFileToString(this, uuid_file, &uuid);
    if (s.ok()) {
      return uuid;
    }
  }
  // Could not read uuid_file - generate uuid using "nanos-random"
  random::Random64 r(time(nullptr));
  uint64_t random_uuid_portion =
    r.Uniform(std::numeric_limits<uint64_t>::max());
  uint64_t nanos_uuid_portion = NowNanos();
  char uuid2[200];
  snprintf(uuid2,
           200,
           "%lx-%lx",
           (unsigned long)nanos_uuid_portion,
           (unsigned long)random_uuid_portion);
  return uuid2;
}

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
  ThreadLocalPtr::InitSingletons();
  static PosixEnv default_env;
  return &default_env;
}
  
} // namespace bubblefs
