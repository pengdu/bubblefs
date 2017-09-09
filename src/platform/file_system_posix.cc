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

#if defined(OS_LINUX)
#include <linux/fs.h>
#endif
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
#include "platform/file_system_posix.h"
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

// A wrapper for fadvise, if the platform doesn't support fadvise,
// it will simply return 0.
int Fadvise(int fd, off_t offset, size_t len, int advice) {
#ifdef OS_LINUX
  return posix_fadvise(fd, offset, len, advice);
#else
  return 0;  // simply do nothing.
#endif
}

namespace {
size_t GetLogicalBufferSize(int __attribute__((__unused__)) fd) {
#ifdef OS_LINUX
  struct stat buf;
  int result = fstat(fd, &buf);
  if (result == -1) {
    return kDefaultPageSize;
  }
  if (major(buf.st_dev) == 0) {
    // Unnamed devices (e.g. non-device mounts), reserved as null device number.
    // These don't have an entry in /sys/dev/block/. Return a sensible default.
    return kDefaultPageSize;
  }

  // Reading queue/logical_block_size does not require special permissions.
  const int kBufferSize = 100;
  char path[kBufferSize];
  char real_path[PATH_MAX + 1];
  snprintf(path, kBufferSize, "/sys/dev/block/%u:%u", major(buf.st_dev),
           minor(buf.st_dev));
  if (realpath(path, real_path) == nullptr) {
    return kDefaultPageSize;
  }
  string device_dir(real_path);
  if (!device_dir.empty() && device_dir.back() == '/') {
    device_dir.pop_back();
  }
  // NOTE: sda3 does not have a `queue/` subdir, only the parent sda has it.
  // $ ls -al '/sys/dev/block/8:3'
  // lrwxrwxrwx. 1 root root 0 Jun 26 01:38 /sys/dev/block/8:3 ->
  // ../../block/sda/sda3
  size_t parent_end = device_dir.rfind('/', device_dir.length() - 1);
  if (parent_end == std::string::npos) {
    return kDefaultPageSize;
  }
  size_t parent_begin = device_dir.rfind('/', parent_end - 1);
  if (parent_begin == std::string::npos) {
    return kDefaultPageSize;
  }
  if (device_dir.substr(parent_begin + 1, parent_end - parent_begin - 1) !=
      "block") {
    device_dir = device_dir.substr(0, parent_end);
  }
  string fname = device_dir + "/queue/logical_block_size";
  FILE* fp;
  size_t size = 0;
  fp = fopen(fname.c_str(), "r");
  if (fp != nullptr) {
    char* line = nullptr;
    size_t len = 0;
    if (getline(&line, &len, fp) != -1) {
      sscanf(line, "%zu", &size);
    }
    free(line);
    fclose(fp);
  }
  if (size != 0 && (size & (size - 1)) == 0) {
    return size;
  }
#endif
  return kDefaultPageSize;
}
} //  namespace

/*
 * DirectIOHelper
 */
#ifndef NDEBUG
namespace {  
  
bool IsSectorAligned(const size_t off, size_t sector_size) {
  return off % sector_size == 0;
}

bool IsSectorAligned(const void* ptr, size_t sector_size) {
  return uintptr_t(ptr) % sector_size == 0;
}

} // namespace
#endif

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

/*
 * PosixSequentialFile
 */
PosixSequentialFile::PosixSequentialFile(const std::string& fname, FILE* file,
                                         int fd, const EnvOptions& options)
    : filename_(fname),
      file_(file),
      fd_(fd),
      use_direct_io_(options.use_direct_reads),
      logical_sector_size_(GetLogicalBufferSize(fd_)) {
  assert(!options.use_direct_reads || !options.use_mmap_reads);
}

PosixSequentialFile::~PosixSequentialFile() {
  if (!use_direct_io()) {
    assert(file_);
    fclose(file_);
  } else {
    assert(fd_);
    close(fd_);
  }
}

Status PosixSequentialFile::Read(size_t n, StringPiece* result, char* scratch) {
  assert(result != nullptr && !use_direct_io());
  Status s;
  size_t r = 0;
  do {
    r = fread_unlocked(scratch, 1, n, file_);
  } while (r == 0 && ferror(file_) && errno == EINTR);
  *result = StringPiece(scratch, r);
  if (r < n) {
    if (feof(file_)) {
      // We leave status as ok if we hit the end of the file
      // We also clear the error so that the reads can continue
      // if a new data is written to the file
      clearerr(file_);
    } else {
      // A partial read with an error: return a non-ok status
      s = IOError(filename_, errno);
    }
  }
  return s;
}

Status PosixSequentialFile::PositionedRead(uint64_t offset, size_t n,
                                           StringPiece* result, char* scratch) {
  if (use_direct_io()) {
    assert(IsSectorAligned(offset, GetRequiredBufferAlignment()));
    assert(IsSectorAligned(n, GetRequiredBufferAlignment()));
    assert(IsSectorAligned(scratch, GetRequiredBufferAlignment()));
  }
  Status s;
  ssize_t r = -1;
  size_t left = n;
  char* ptr = scratch;
  assert(use_direct_io());
  while (left > 0) {
    r = pread(fd_, ptr, left, static_cast<off_t>(offset));
    if (r <= 0) {
      if (r == -1 && errno == EINTR) {
        continue;
      }
      break;
    }
    ptr += r;
    offset += r;
    left -= r;
    if (r % static_cast<ssize_t>(GetRequiredBufferAlignment()) != 0) {
      // Bytes reads don't fill sectors. Should only happen at the end
      // of the file.
      break;
    }
  }
  if (r < 0) {
    // An error: return a non-ok status
    s = IOError(filename_, errno);
  }
  *result = StringPiece(scratch, (r < 0) ? 0 : n - left);
  return s;
}

Status PosixSequentialFile::Skip(uint64_t n) {
  if (fseek(file_, static_cast<long int>(n), SEEK_CUR)) {
    return IOError(filename_, errno);
  }
  return Status::OK();
}

Status PosixSequentialFile::InvalidateCache(size_t offset, size_t length) {
#ifndef OS_LINUX
  return Status::OK();
#else
  if (!use_direct_io()) {
    // free OS pages
    int ret = Fadvise(fd_, offset, length, POSIX_FADV_DONTNEED);
    if (ret != 0) {
      return IOError(filename_, errno);
    }
  }
  return Status::OK();
#endif
}

/*
 * PosixRandomAccessFile
 *
 * pread() based random-access
 */
PosixRandomAccessFile::PosixRandomAccessFile(const string& fname, int fd,
                                             const EnvOptions& options)
    : filename_(fname),
      fd_(fd),
      use_direct_io_(options.use_direct_reads),
      logical_sector_size_(GetLogicalBufferSize(fd_)) {
  assert(!options.use_direct_reads || !options.use_mmap_reads);
  assert(!options.use_mmap_reads || sizeof(void*) < 8);
}

PosixRandomAccessFile::~PosixRandomAccessFile() { close(fd_); }

Status PosixRandomAccessFile::Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
  if (use_direct_io()) {
    assert(IsSectorAligned(offset, GetRequiredBufferAlignment()));
    assert(IsSectorAligned(n, GetRequiredBufferAlignment()));
    assert(IsSectorAligned(scratch, GetRequiredBufferAlignment()));
  }
  Status s;
  char* dst = scratch;
  while (n > 0 && s.ok()) {
    ssize_t r = pread(fd_, dst, n, static_cast<off_t>(offset));
    if (r > 0) {
      dst += r;
      n -= r;
      offset += r;
      if (use_direct_io() &&
        r % static_cast<ssize_t>(GetRequiredBufferAlignment()) != 0) {
        // Bytes reads don't fill sectors. Should only happen at the end
        // of the file.
        break;
      }
    } else if (r == 0) {
      s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
    } else if (errno == EINTR || errno == EAGAIN) {
      // Retry
    } else {
      s = IOError(filename_, errno);
    }
  }
  *result = StringPiece(scratch, dst - scratch);
  return s;
}

PosixWritableFile::~PosixWritableFile() override {
  if (file_ != nullptr) {
    // Ignoring any potential errors
    fclose(file_);
  }
}

Status PosixRandomAccessFile::Prefetch(uint64_t offset, size_t n) {
  Status s;
  if (!use_direct_io()) {
    ssize_t r = 0;
#ifdef OS_LINUX
    r = readahead(fd_, offset, n);
#endif
    if (r == -1) {
      s = IOError(filename_, errno);
    }
  }
  return s;
}

void PosixRandomAccessFile::Hint(AccessPattern pattern) {
  if (use_direct_io()) {
    return;
  }
  switch (pattern) {
    case NORMAL:
      Fadvise(fd_, 0, 0, POSIX_FADV_NORMAL);
      break;
    case RANDOM:
      Fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
      break;
    case SEQUENTIAL:
      Fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
      break;
    case WILLNEED:
      Fadvise(fd_, 0, 0, POSIX_FADV_WILLNEED);
      break;
    case DONTNEED:
      Fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
      break;
    default:
      assert(false);
      break;
  }
}

Status PosixRandomAccessFile::InvalidateCache(size_t offset, size_t length) {
  if (use_direct_io()) {
    return Status::OK();
  }
#ifndef OS_LINUX
  return Status::OK();
#else
  // free OS pages
  int ret = Fadvise(fd_, offset, length, POSIX_FADV_DONTNEED);
  if (ret == 0) {
    return Status::OK();
  }
  return IOError(filename_, errno);
#endif
}

/*
 * PosixMmapReadableFile
 *
 * mmap() based random-access
 */
// base[0,length-1] contains the mmapped contents of the file.
PosixMmapReadableFile::PosixMmapReadableFile(const int fd,
                                             const string& fname,
                                             void* base, size_t length,
                                             const EnvOptions& options)
    : fd_(fd), filename_(fname), mmapped_region_(base), length_(length) {
  fd_ = fd_ + 0;  // suppress the warning for used variables
  assert(options.use_mmap_reads);
  assert(!options.use_direct_reads);
}

PosixMmapReadableFile::~PosixMmapReadableFile() {
  int ret = munmap(mmapped_region_, length_);
  if (ret != 0) {
    fprintf(stdout, "failed to munmap %p length %ld\n",
            mmapped_region_, (long)length_);
  }
}

Status PosixMmapReadableFile::Read(uint64_t offset, size_t n, StringPiece* result,
                                   char* scratch) const {
  Status s;
  if (offset > length_) {
    *result = StringPiece();
    return IOError(filename_, EINVAL);
  } else if (offset + n > length_) {
    n = static_cast<size_t>(length_ - offset);
  }
  *result = StringPiece(reinterpret_cast<char*>(mmapped_region_) + offset, n);
  return s;
}

Status PosixMmapReadableFile::InvalidateCache(size_t offset, size_t length) {
#ifndef OS_LINUX
  return Status::OK();
#else
  // free OS pages
  int ret = Fadvise(fd_, offset, length, POSIX_FADV_DONTNEED);
  if (ret == 0) {
    return Status::OK();
  }
  return IOError(filename_, errno);
#endif
}

/*
 * PosixMmapFile
 *
 * We preallocate up to an extra megabyte and use memcpy to append new
 * data to the file.  This is safe since we either properly close the
 * file before reading from it, or for log files, the reading code
 * knows enough to skip zero suffixes.
 */
Status PosixMmapFile::UnmapCurrentRegion() {
  //TEST_KILL_RANDOM("PosixMmapFile::UnmapCurrentRegion:0", rocksdb_kill_odds);
  if (base_ != nullptr) {
    int munmap_status = munmap(base_, limit_ - base_);
    if (munmap_status != 0) {
      return IOError(filename_, munmap_status);
    }
    file_offset_ += limit_ - base_;
    base_ = nullptr;
    limit_ = nullptr;
    last_sync_ = nullptr;
    dst_ = nullptr;

    // Increase the amount we map the next time, but capped at 1MB
    if (map_size_ < (1 << 20)) {
      map_size_ *= 2;
    }
  }
  return Status::OK();
}

Status PosixMmapFile::MapNewRegion() {
  return Status::NotSupported("This platform doesn't support fallocate()");
}

Status PosixMmapFile::Msync() {
  if (dst_ == last_sync_) {
    return Status::OK();
  }
  // Find the beginnings of the pages that contain the first and last
  // bytes to be synced.
  size_t p1 = TruncateToPageBoundary(last_sync_ - base_);
  size_t p2 = TruncateToPageBoundary(dst_ - base_ - 1);
  last_sync_ = dst_;
  //TEST_KILL_RANDOM("PosixMmapFile::Msync:0", rocksdb_kill_odds);
  if (msync(base_ + p1, p2 - p1 + page_size_, MS_SYNC) < 0) {
    return IOError(filename_, errno);
  }
  return Status::OK();
}

PosixMmapFile::PosixMmapFile(const string& fname, int fd, size_t page_size,
                             const EnvOptions& options)
    : filename_(fname),
      fd_(fd),
      page_size_(page_size),
      map_size_(Roundup(65536, page_size)),
      base_(nullptr),
      limit_(nullptr),
      dst_(nullptr),
      last_sync_(nullptr),
      file_offset_(0) {
  assert((page_size & (page_size - 1)) == 0);
  assert(options.use_mmap_writes);
  assert(!options.use_direct_writes);
}

PosixMmapFile::~PosixMmapFile() {
  if (fd_ >= 0) {
    PosixMmapFile::Close();
  }
}

Status PosixMmapFile::Append(const StringPiece& data) {
  const char* src = data.data();
  size_t left = data.size();
  while (left > 0) {
    assert(base_ <= dst_);
    assert(dst_ <= limit_);
    size_t avail = limit_ - dst_;
    if (avail == 0) {
      Status s = UnmapCurrentRegion();
      if (!s.ok()) {
        return s;
      }
      s = MapNewRegion();
      if (!s.ok()) {
        return s;
      }
      //TEST_KILL_RANDOM("PosixMmapFile::Append:0", rocksdb_kill_odds);
    }

    size_t n = (left <= avail) ? left : avail;
    assert(dst_);
    memcpy(dst_, src, n);
    dst_ += n;
    src += n;
    left -= n;
  }
  return Status::OK();
}

Status PosixMmapFile::Close() {
  Status s;
  size_t unused = limit_ - dst_;

  s = UnmapCurrentRegion();
  if (!s.ok()) {
    s = IOError(filename_, errno);
  } else if (unused > 0) {
    // Trim the extra space at the end of the file
    if (ftruncate(fd_, file_offset_ - unused) < 0) {
      s = IOError(filename_, errno);
    }
  }

  if (close(fd_) < 0) {
    if (s.ok()) {
      s = IOError(filename_, errno);
    }
  }

  fd_ = -1;
  base_ = nullptr;
  limit_ = nullptr;
  return s;
}

Status PosixMmapFile::Flush() { return Status::OK(); }

Status PosixMmapFile::Sync() {
  if (fdatasync(fd_) < 0) {
    return IOError(filename_, errno);
  }

  return Msync();
}

/**
 * Flush data as well as metadata to stable storage.
 */
Status PosixMmapFile::Fsync() {
  if (fsync(fd_) < 0) {
    return IOError(filename_, errno);
  }

  return Msync();
}

/**
 * Get the size of valid data in the file. This will not match the
 * size that is returned from the filesystem because we use mmap
 * to extend file by map_size every time.
 */
uint64_t PosixMmapFile::GetFileSize() {
  size_t used = dst_ - base_;
  return file_offset_ + used;
}

Status PosixMmapFile::InvalidateCache(size_t offset, size_t length) {
#ifndef OS_LINUX
  return Status::OK();
#else
  // free OS pages
  int ret = Fadvise(fd_, offset, length, POSIX_FADV_DONTNEED);
  if (ret == 0) {
    return Status::OK();
  }
  return IOError(filename_, errno);
#endif
}

/*
 * PosixWritableFile
 *
 * Use posix write to write data to a file.
 */
PosixWritableFile::PosixWritableFile(const string& fname, int fd,
                                     const EnvOptions& options)
    : filename_(fname),
      use_direct_io_(options.use_direct_writes),
      fd_(fd),
      filesize_(0),
      logical_sector_size_(GetLogicalBufferSize(fd_)) {
  assert(!options.use_mmap_writes);
}

PosixWritableFile::~PosixWritableFile() {
  if (fd_ >= 0) {
    PosixWritableFile::Close();
  }
}

Status PosixWritableFile::Append(const StringPiece& data) {
  if (use_direct_io()) {
    assert(IsSectorAligned(data.size(), GetRequiredBufferAlignment()));
    assert(IsSectorAligned(data.data(), GetRequiredBufferAlignment()));
  }
  const char* src = data.data();
  size_t left = data.size();
  while (left != 0) {
    ssize_t done = write(fd_, src, left);
    if (done < 0) {
      if (errno == EINTR) {
        continue;
      }
      return IOError(filename_, errno);
    }
    left -= done;
    src += done;
  }
  filesize_ += data.size();
  return Status::OK();
}

Status PosixWritableFile::PositionedAppend(const StringPiece& data, uint64_t offset) {
  if (use_direct_io()) {
    assert(IsSectorAligned(offset, GetRequiredBufferAlignment()));
    assert(IsSectorAligned(data.size(), GetRequiredBufferAlignment()));
    assert(IsSectorAligned(data.data(), GetRequiredBufferAlignment()));
  }
  assert(offset <= std::numeric_limits<off_t>::max());
  const char* src = data.data();
  size_t left = data.size();
  while (left != 0) {
    ssize_t done = pwrite(fd_, src, left, static_cast<off_t>(offset));
    if (done < 0) {
      if (errno == EINTR) {
        continue;
      }
      return IOError(filename_, errno);
    }
    left -= done;
    offset += done;
    src += done;
  }
  filesize_ = offset;
  return Status::OK();
}

Status PosixWritableFile::Truncate(uint64_t size) {
  Status s;
  int r = ftruncate(fd_, size);
  if (r < 0) {
    s = IOError(filename_, errno);
  } else {
    filesize_ = size;
  }
  return s;
}

Status PosixWritableFile::Close() {
  Status s;

  size_t block_size;
  size_t last_allocated_block;
  GetPreallocationStatus(&block_size, &last_allocated_block);
  if (last_allocated_block > 0) {
    // trim the extra space preallocated at the end of the file
    // NOTE(ljin): we probably don't want to surface failure as an IOError,
    // but it will be nice to log these errors.
    int dummy __attribute__((unused));
    dummy = ftruncate(fd_, filesize_);
  }

  if (close(fd_) < 0) {
    s = IOError(filename_, errno);
  }
  fd_ = -1;
  return s;
}

// write out the cached data to the OS cache
Status PosixWritableFile::Flush() { return Status::OK(); }

Status PosixWritableFile::Sync() {
  if (fdatasync(fd_) < 0) {
    return IOError(filename_, errno);
  }
  return Status::OK();
}

Status PosixWritableFile::Fsync() {
  if (fsync(fd_) < 0) {
    return IOError(filename_, errno);
  }
  return Status::OK();
}

bool PosixWritableFile::IsSyncThreadSafe() const { return true; }

uint64_t PosixWritableFile::GetFileSize() { return filesize_; }

Status PosixWritableFile::InvalidateCache(size_t offset, size_t length) {
  if (use_direct_io()) {
    return Status::OK();
  }
#ifndef OS_LINUX
  return Status::OK();
#else
  // free OS pages
  int ret = Fadvise(fd_, offset, length, POSIX_FADV_DONTNEED);
  if (ret == 0) {
    return Status::OK();
  }
  return IOError(filename_, errno);
#endif
}

/*
 * PosixRandomRWFile
 */

PosixRandomRWFile::PosixRandomRWFile(const string& fname, int fd,
                                     const EnvOptions& options)
    : filename_(fname), fd_(fd) {}

PosixRandomRWFile::~PosixRandomRWFile() {
  if (fd_ >= 0) {
    Close();
  }
}

Status PosixRandomRWFile::Write(uint64_t offset, const StringPiece& data) {
  const char* src = data.data();
  size_t left = data.size();
  while (left != 0) {
    ssize_t done = pwrite(fd_, src, left, offset);
    if (done < 0) {
      // error while writing to file
      if (errno == EINTR) {
        // write was interrupted, try again.
        continue;
      }
      return IOError(filename_, errno);
    }

    // Wrote `done` bytes
    left -= done;
    offset += done;
    src += done;
  }

  return Status::OK();
}

Status PosixRandomRWFile::Read(uint64_t offset, size_t n, StringPiece* result,
                               char* scratch) const {
  size_t left = n;
  char* ptr = scratch;
  while (left > 0) {
    ssize_t done = pread(fd_, ptr, left, offset);
    if (done < 0) {
      // error while reading from file
      if (errno == EINTR) {
        // read was interrupted, try again.
        continue;
      }
      return IOError(filename_, errno);
    } else if (done == 0) {
      // Nothing more to read
      break;
    }

    // Read `done` bytes
    ptr += done;
    offset += done;
    left -= done;
  }

  *result = StringPiece(scratch, n - left);
  return Status::OK();
}

Status PosixRandomRWFile::Flush() { return Status::OK(); }

Status PosixRandomRWFile::Sync() {
  if (fdatasync(fd_) < 0) {
    return IOError("While fdatasync random read/write file", filename_, errno);
  }
  return Status::OK();
}

Status PosixRandomRWFile::Fsync() {
  if (fsync(fd_) < 0) {
    return IOError("While fsync random read/write file", filename_, errno);
  }
  return Status::OK();
}

Status PosixRandomRWFile::Close() {
  if (close(fd_) < 0) {
    return IOError("While close random read/write file", filename_, errno);
  }
  fd_ = -1;
  return Status::OK();
}

PosixReadOnlyMemoryRegion::~PosixReadOnlyMemoryRegion() {
  munmap(const_cast<void*>(address_), length_);
}

/*
 * PosixDirectory
 */

PosixDirectory::~PosixDirectory() { close(fd_); }

Status PosixDirectory::Fsync() {
#ifndef OS_AIX
  if (fsync(fd_) == -1) {
    return IOError("While fsync", "a directory", errno);
  }
#endif
  return Status::OK();
}

void PosixFileSystem::SetFD_CLOEXEC(int fd, const EnvOptions* options) {
  if ((options == nullptr || options->set_fd_cloexec) && fd > 0) {
    fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) | FD_CLOEXEC);
  }
}

Status PosixFileSystem::NewSequentialFile(const string& fname,
                                          std::unique_ptr<SequentialFile>* result,
                                          const EnvOptions& options) override {
  result->reset();
  int fd = -1;
  int flags = O_RDONLY;
  FILE* file = nullptr;

  if (options.use_direct_reads && !options.use_mmap_reads) {
    flags |= O_DIRECT;
  }

  do {
    IOSTATS_TIMER_GUARD(open_nanos);
    fd = open(fname.c_str(), flags, 0644);
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

}  // namespace bubblefs