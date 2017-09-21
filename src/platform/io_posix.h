//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// rocksdb/env/io_posix.h

#ifndef BUBBLEFS_PLATFORM_IO_POSIX_H_
#define BUBBLEFS_PLATFORM_IO_POSIX_H_

#include <errno.h>
#include <unistd.h>
#include <atomic>
#include <string>
#include "platform/env.h"

// For non linux platform, the following macros are used only as place
// holder.
#if !(defined OS_LINUX) && !(defined CYGWIN) && !(defined OS_AIX)
#define POSIX_FADV_NORMAL 0     /* [MC1] no further special treatment */
#define POSIX_FADV_RANDOM 1     /* [MC1] expect random page refs */
#define POSIX_FADV_SEQUENTIAL 2 /* [MC1] expect sequential page refs */
#define POSIX_FADV_WILLNEED 3   /* [MC1] will need these pages */
#define POSIX_FADV_DONTNEED 4   /* [MC1] dont need these pages */
#endif

namespace bubblefs {  
  
class PosixFileLock : public FileLock {
 public:
  int fd_;
  string filename;
};

class PosixHelper {
 public:
  static size_t GetUniqueIdFromFile(int fd, char* id, size_t max_size);
};

class PosixSequentialFile : public SequentialFile {
 private:
  string filename_;
  FILE* file_;
  int fd_;
  bool use_direct_io_;
  size_t logical_sector_size_;

 public:
  PosixSequentialFile(const string& fname, FILE* file, int fd,
                      const EnvOptions& options);
  virtual ~PosixSequentialFile();

  virtual Status Read(size_t n, StringPiece* result, char* scratch) override;
  virtual Status PositionedRead(uint64_t offset, size_t n, StringPiece* result,
                                char* scratch) override;
  virtual Status Skip(uint64_t n) override;
  virtual Status InvalidateCache(size_t offset, size_t length) override;
  virtual bool use_direct_io() const override { return use_direct_io_; }
  virtual size_t GetRequiredBufferAlignment() const override {
    return logical_sector_size_;
  }
};


// pread() based random-access
class PosixRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  int fd_;
  bool use_direct_io_;
  size_t logical_sector_size_;

 public:
  PosixRandomAccessFile(const string& fname, int fd)
      : filename_(fname), fd_(fd) {}
  PosixRandomAccessFile(const string& fname, int fd,
                        const EnvOptions& options);
  ~PosixRandomAccessFile() override;

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override;
  
  virtual Status Prefetch(uint64_t offset, size_t n) override;
  
#if defined(OS_LINUX) || defined(OS_MACOSX) || defined(OS_AIX)
  virtual size_t GetUniqueId(char* id, size_t max_size) const override;
#endif
  //virtual size_t GetUniqueId(char* id, size_t max_size) const override;
  virtual void Hint(AccessPattern pattern) override;
  virtual Status InvalidateCache(size_t offset, size_t length) override;
  virtual bool use_direct_io() const override { return use_direct_io_; }
  virtual size_t GetRequiredBufferAlignment() const override {
    return logical_sector_size_;
  }
};

class PosixWritableFile : public WritableFile {
 private:
  string filename_;
  const bool use_direct_io_;
  int fd_;
  uint64_t filesize_;
  size_t logical_sector_size_;
  bool allow_fallocate_;
  bool fallocate_with_keep_size_;

 public:
  PosixWritableFile(const string& fname, int fd)
      : filename_(fname), use_direct_io_(false), fd_(fd) {}
  PosixWritableFile(const string& fname, int fd,
                    const EnvOptions& options);

  ~PosixWritableFile() override;
  
  // Need to implement this so the file is truncated correctly
  // with direct I/O
  virtual Status Truncate(uint64_t size) override;
  virtual Status Close() override;
  virtual Status Append(const StringPiece& data) override;
  virtual Status PositionedAppend(const StringPiece& data, uint64_t offset) override;
  virtual Status Flush() override;
  virtual Status Sync() override;
  virtual Status Fsync() override;
  virtual bool IsSyncThreadSafe() const override;
  virtual bool use_direct_io() const override { return use_direct_io_; }
  virtual uint64_t GetFileSize() override;
  virtual Status InvalidateCache(size_t offset, size_t length) override;
  virtual size_t GetRequiredBufferAlignment() const override {
    return logical_sector_size_;
  }
#ifdef OS_LINUX
  virtual Status Allocate(uint64_t offset, uint64_t len) override;
#endif
#ifdef OS_LINUX
  virtual Status RangeSync(uint64_t offset, uint64_t nbytes) override;
#endif
#ifdef OS_LINUX
  virtual size_t GetUniqueId(char* id, size_t max_size) const override;
#endif
};

// mmap() based random-access
class PosixMmapReadableFile : public RandomAccessFile {
 private:
  int fd_;
  string filename_;
  void* mmapped_region_;
  size_t length_;

 public:
  PosixMmapReadableFile(const int fd, const string& fname, void* base,
                        size_t length, const EnvOptions& options);
  virtual ~PosixMmapReadableFile();
  virtual Status Read(uint64_t offset, size_t n, StringPiece* result,
                      char* scratch) const override;
  virtual Status InvalidateCache(size_t offset, size_t length) override;
};

class PosixMmapFile : public WritableFile {
 private:
  string filename_;
  int fd_;
  size_t page_size_;
  size_t map_size_;       // How much extra memory to map at a time
  char* base_;            // The mapped region
  char* limit_;           // Limit of the mapped region
  char* dst_;             // Where to write next  (in range [base_,limit_])
  char* last_sync_;       // Where have we synced up to
  uint64_t file_offset_;  // Offset of base_ in file
  bool allow_fallocate_;  // If false, fallocate calls are bypassed
  bool fallocate_with_keep_size_;

  // Roundup x to a multiple of y
  static size_t Roundup(size_t x, size_t y) { return ((x + y - 1) / y) * y; }

  size_t TruncateToPageBoundary(size_t s) {
    s -= (s & (page_size_ - 1));
    assert((s % page_size_) == 0);
    return s;
  }

  Status MapNewRegion();
  Status UnmapCurrentRegion();
  Status Msync();

 public:
  PosixMmapFile(const string& fname, int fd, size_t page_size,
                const EnvOptions& options);
  ~PosixMmapFile();

  // Means Close() will properly take care of truncate
  // and it does not need any additional information
  virtual Status Truncate(uint64_t size) override { return Status::OK(); }
  virtual Status Close() override;
  virtual Status Append(const StringPiece& data) override;
  virtual Status Flush() override;
  virtual Status Sync() override;
  virtual Status Fsync() override;
  virtual uint64_t GetFileSize() override;
  virtual Status InvalidateCache(size_t offset, size_t length) override;
#ifdef OS_LINUX
  virtual Status Allocate(uint64_t offset, uint64_t len) override;
#endif
};

class PosixRandomRWFile : public RandomRWFile {
 public:
  explicit PosixRandomRWFile(const string& fname, int fd,
                             const EnvOptions& options);
  virtual ~PosixRandomRWFile();

  virtual Status Write(uint64_t offset, const StringPiece& data) override;

  virtual Status Read(uint64_t offset, size_t n, StringPiece* result,
                      char* scratch) const override;

  virtual Status Flush() override;
  virtual Status Sync() override;
  virtual Status Fsync() override;
  virtual Status Close() override;

 private:
  const string filename_;
  int fd_;
};

class PosixReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  PosixReadOnlyMemoryRegion(const void* address, uint64 length)
      : address_(address), length_(length) {}
  ~PosixReadOnlyMemoryRegion() override;
  const void* data() override { return address_; }
  uint64 length() override { return length_; }

 private:
  const void* const address_;
  const uint64 length_;
};
class PosixDirectory : public Directory {
 public:
  explicit PosixDirectory(int fd) : fd_(fd) {}
  ~PosixDirectory();
  virtual Status Fsync() override;

 private:
  int fd_;
};

} // namespace bubblefs

#endif // BUBBLEFS_PLATFORM_IO_POSIX_H_