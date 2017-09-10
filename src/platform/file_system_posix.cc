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

#ifdef OS_LINUX
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#endif
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "platform/env.h"
#include "platform/error.h"
#include "platform/logging.h"
#include "platform/platform.h"
#include "platform/port_posix.h"
#include "utils/error_codes.h"
#include "utils/status.h"
#include "utils/strcat.h"

#define IOSTATS_TIMER_GUARD(metric)

namespace bubblefs {
  
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



void PosixFileSystem::SetFD_CLOEXEC(int fd, const EnvOptions* options) {
  if ((options == nullptr || options->set_fd_cloexec) && fd > 0) {
    fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) | FD_CLOEXEC);
  }
}

Status PosixFileSystem::NewSequentialFile(const string& fname,
                                          std::unique_ptr<SequentialFile>* result,
                                          const EnvOptions& options) {
  result->reset();
  int fd = -1;
  int flags = O_RDONLY;
  FILE* file = nullptr;

  if (options.use_direct_reads && !options.use_mmap_reads) {
    flags |= O_DIRECT;
  }

  string translated_fname = TranslateName(fname);
  do {
    IOSTATS_TIMER_GUARD(open_nanos);
    fd = open(translated_fname.c_str(), flags, 0644);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return IOError(fname, errno);
  }

  SetFD_CLOEXEC(fd, &options);

  if (options.use_direct_reads && !options.use_mmap_reads) {
#ifdef OS_MACOSX
    if (fcntl(fd, F_NOCACHE, 1) == -1) {
      close(fd);
      return IOError(fname, errno);
    }
#endif
  } else {
    do {
      IOSTATS_TIMER_GUARD(open_nanos);
      file = fdopen(fd, "r");
    } while (file == nullptr && errno == EINTR);
    if (file == nullptr) {
      close(fd);
      return IOError(fname, errno);
    }
  }
  result->reset(new PosixSequentialFile(fname, file, fd, options));
  return Status::OK();
}

Status PosixFileSystem::NewRandomAccessFile(
  const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixRandomAccessFile(translated_fname, fd));
  }
  return s;
}

Status PosixFileSystem::NewRandomAccessFile(const string& fname,
                                            std::unique_ptr<RandomAccessFile>* result,
                                            const EnvOptions& options) {
  result->reset();
  Status s;
  int fd;
  int flags = O_RDONLY;
  if (options.use_direct_reads && !options.use_mmap_reads) {
#if !defined(OS_MACOSX) && !defined(OS_OPENBSD) && !defined(OS_SOLARIS)
    flags |= O_DIRECT;
    //TEST_SYNC_POINT_CALLBACK("NewRandomAccessFile:O_DIRECT", &flags);
#endif
  }

  string translated_fname = TranslateName(fname);
  do {
    IOSTATS_TIMER_GUARD(open_nanos);
    fd = open(translated_fname.c_str(), flags, 0644);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return IOError(fname, errno);
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
        s = IOError(fname, errno);
      }
    }
    close(fd);
  } else {
    if (options.use_direct_reads && !options.use_mmap_reads) {
#ifdef OS_MACOSX
      if (fcntl(fd, F_NOCACHE, 1) == -1) {
        close(fd);
        return IOError(fname, errno);
      }
#endif
    }
    result->reset(new PosixRandomAccessFile(fname, fd, options));
  }
  return s;
}

Status PosixFileSystem::OpenWritableFile(const string& fname,
                                         std::unique_ptr<WritableFile>* result,
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
    //TEST_SYNC_POINT_CALLBACK("NewWritableFile:O_DIRECT", &flags);
  } else if (options.use_mmap_writes) {
    // non-direct I/O
    flags |= O_RDWR;
  } else {
    flags |= O_WRONLY;
  }

  string translated_fname = TranslateName(fname);
  do {
    IOSTATS_TIMER_GUARD(open_nanos);
    fd = open(translated_fname.c_str(), flags, 0644);
  } while (fd < 0 && errno == EINTR);

  if (fd < 0) {
    s = IOError(fname, errno);
    return s;
  }
  SetFD_CLOEXEC(fd, &options);

  if (options.use_mmap_writes) {
   // this will be executed once in the program's lifetime.
   // do not use mmapWrite on non ext-3/xfs/tmpfs systems.
  }
  if (options.use_mmap_writes) {
    result->reset(new PosixMmapFile(fname, fd, kDefaultPageSize, options));
  } else if (options.use_direct_writes && !options.use_mmap_writes) {
    result->reset(new PosixWritableFile(fname, fd, options));
  } else {
    // disable mmap writes
    EnvOptions no_mmap_writes_options = options;
    no_mmap_writes_options.use_mmap_writes = false;
    result->reset(new PosixWritableFile(fname, fd, no_mmap_writes_options));
  }
  return s;
}

Status PosixFileSystem::NewWritableFile(const string& fname,
                                        std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "w");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewWritableFile(const string& fname,
                                        std::unique_ptr<WritableFile>* result,
                                        const EnvOptions& options) {
  return OpenWritableFile(fname, result, options, false);
}

Status PosixFileSystem::ReopenWritableFile(const string& fname,
                                           std::unique_ptr<WritableFile>* result,
                                           const EnvOptions& options) {
  return OpenWritableFile(fname, result, options, true);
}

Status PosixFileSystem::ReuseWritableFile(const string& fname,
                                          const string& old_fname,
                                          std::unique_ptr<WritableFile>* result,
                                          const EnvOptions& options) {
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
    //TEST_SYNC_POINT_CALLBACK("NewWritableFile:O_DIRECT", &flags);
  } else if (options.use_mmap_writes) {
    // mmap needs O_RDWR mode
    flags |= O_RDWR;
  } else {
    flags |= O_WRONLY;
  }

  string old_translated_fname = TranslateName(fname);
  do {
    IOSTATS_TIMER_GUARD(open_nanos);
    fd = open(old_translated_fname.c_str(), flags, 0644);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    s = IOError(fname, errno);
    return s;
  }

  SetFD_CLOEXEC(fd, &options);
  // rename into place
  if (rename(old_fname.c_str(), fname.c_str()) != 0) {
    s = IOError(old_fname, errno);
    close(fd);
    return s;
  }

  if (options.use_mmap_writes) {
    // this will be executed once in the program's lifetime.
    // do not use mmapWrite on non ext-3/xfs/tmpfs systems.
  }
  if (options.use_mmap_writes) {
    result->reset(new PosixMmapFile(fname, fd, kDefaultPageSize, options));
  } else if (options.use_direct_writes && !options.use_mmap_writes) {
#ifdef OS_MACOSX
    if (fcntl(fd, F_NOCACHE, 1) == -1) {
      close(fd);
      s = IOError(fname, errno);
      return s;
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

Status PosixFileSystem::NewRandomRWFile(const string& fname,
                                        std::unique_ptr<RandomRWFile>* result,
                                        const EnvOptions& options) {
  int fd = -1;
  string translated_fname = TranslateName(fname);
  while (fd < 0) {
    IOSTATS_TIMER_GUARD(open_nanos);
    fd = open(translated_fname.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0) {
      // Error while opening the file
      if (errno == EINTR) {
        continue;
      }
      return IOError(fname, errno);
    }
  }

  SetFD_CLOEXEC(fd, &options);
  result->reset(new PosixRandomRWFile(fname, fd, options));
  return Status::OK();
}

Status PosixFileSystem::NewAppendableFile(
  const string& fname, std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "a");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewReadOnlyMemoryRegionFromFile(
  const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  string translated_fname = TranslateName(fname);
  Status s = Status::OK();
  int fd = open(translated_fname.c_str(), O_RDONLY);
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

Status PosixFileSystem::NewDirectory(const string& name,
                                     std::unique_ptr<Directory>* result) {
  result->reset();
  int fd;
  {
    string translated_name = TranslateName(name);
    IOSTATS_TIMER_GUARD(open_nanos);
    fd = open(translated_name.c_str(), 0);
  }
  if (fd < 0) {
    return IOError(name, errno);
  } else {
    result->reset(new PosixDirectory(fd));
  }
  return Status::OK();
}

Status PosixFileSystem::FileExists(const string& fname) {
  if (access(TranslateName(fname).c_str(), F_OK) == 0) {
    return Status::OK();
  }
  return errors::NotFound(fname, " not found");
}

Status PosixFileSystem::GetChildren(const string& dir,
                                    std::vector<string>* result) {
  string translated_dir = TranslateName(dir);
  result->clear();
  DIR* d = opendir(translated_dir.c_str());
  if (d == nullptr) {
    return IOError(dir, errno);
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

Status PosixFileSystem::DeleteFile(const string& fname) {
  Status result;
  if (unlink(TranslateName(fname).c_str()) != 0) {
    result = IOError(fname, errno);
  }
  return result;
}

Status PosixFileSystem::CreateDir(const string& name) {
  Status result;
  if (mkdir(TranslateName(name).c_str(), 0755) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status PosixFileSystem::DeleteDir(const string& name) {
  Status result;
  if (rmdir(TranslateName(name).c_str()) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status PosixFileSystem::GetFileSize(const string& fname, uint64* size) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *size = 0;
    s = IOError(fname, errno);
  } else {
    *size = sbuf.st_size;
  }
  return s;
}

Status PosixFileSystem::GetFileModificationTime(const string& fname,
                                                uint64_t* file_mtime) {
  struct stat s;
  if (stat(TranslateName(fname).c_str(), &s) !=0) {
    return IOError(fname, errno);
  }
  *file_mtime = static_cast<uint64_t>(s.st_mtime);
  return Status::OK();
}

Status PosixFileSystem::Stat(const string& fname, FileStatistics* stats) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    s = IOError(fname, errno);
  } else {
    stats->length = sbuf.st_size;
    stats->mtime_nsec = sbuf.st_mtime * 1e9;
    stats->is_directory = S_ISDIR(sbuf.st_mode);
  }
  return s;
}

Status PosixFileSystem::RenameFile(const string& src, const string& target) {
  Status result;
  if (rename(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    result = IOError(src, errno);
  }
  return result;
}

Status PosixFileSystem::LinkFile(const string& src,
                                 const string& target)  {
  Status result;
  if (link(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    if (errno == EXDEV) {
      return Status::NotSupported("No cross FS links allowed");
    }
    result = IOError(src, errno);
  }
  return result;
}

Status PosixFileSystem::LockFile(const string& fname, FileLock** lock) {
  *lock = nullptr;
  Status result;
  int fd;
  {
    IOSTATS_TIMER_GUARD(open_nanos);
    string translated_fname = TranslateName(fname);
    fd = open(translated_fname.c_str(), O_RDWR | O_CREAT, 0644);
  }
  if (fd < 0) {
     result = IOError(fname, errno);
  } else if (LockOrUnlock(fname, fd, true) == -1) {
    result = IOError(fname, errno);
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

Status PosixFileSystem::UnlockFile(FileLock* lock) {
  PosixFileLock* my_lock = reinterpret_cast<PosixFileLock*>(lock);
  Status result;
  if (LockOrUnlock(my_lock->filename, my_lock->fd_, false) == -1) {
    result = IOError("unlock", my_lock->filename, errno);
  }
  close(my_lock->fd_);
  delete my_lock;
  return result;
}

}  // namespace bubblefs