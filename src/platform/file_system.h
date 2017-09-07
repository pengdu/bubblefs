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
// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//

// tensorflow/tensorflow/core/platform/file_statistics.h
// tensorflow/tensorflow/core/platform/file_system.h
// rocksdb/include/rocksdb/env.h

#ifndef BUBBLEFS_PLATFORM_FILE_SYSTEM_H_
#define BUBBLEFS_PLATFORM_FILE_SYSTEM_H_

#include <stdint.h>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "platform/macros.h"
#include "platform/platform.h"
#include "platform/protobuf.h"
#include "platform/types.h"
#include "utils/errors.h"
#include "utils/status.h"
#include "utils/stringpiece.h"

#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#endif

namespace bubblefs {

class RandomAccessFile;
class ReadOnlyMemoryRegion;
class WritableFile;

struct FileStatistics {
  // The length of the file or -1 if finding file length is not supported.
  int64 length = -1;
  // The last modified time in nanoseconds.
  int64 mtime_nsec = 0;
  // True if the file is a directory, otherwise false.
  bool is_directory = false;

  FileStatistics() {}
  FileStatistics(int64 length, int64 mtime_nsec, bool is_directory)
      : length(length), mtime_nsec(mtime_nsec), is_directory(is_directory) {}
  ~FileStatistics() {}
};

/// A generic interface for accessing a file system.  Implementations
/// of custom filesystem adapters must implement this interface,
/// RandomAccessFile, WritableFile, and ReadOnlyMemoryRegion classes.
class FileSystem {
 public:
  /// \brief Creates a brand new random access read-only file with the
  /// specified name.
  ///
  /// On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.  If the file does not exist, returns a non-OK
  /// status.
  ///
  /// The returned file may be concurrently accessed by multiple threads.
  ///
  /// The ownership of the returned RandomAccessFile is passed to the caller
  /// and the object should be deleted when is not used.
  virtual Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) = 0;

  /// \brief Creates an object that writes to a new file with the specified
  /// name.
  ///
  /// Deletes any existing file with the same name and creates a
  /// new file.  On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used.
  virtual Status NewWritableFile(const string& fname,
                                 std::unique_ptr<WritableFile>* result) = 0;

  /// \brief Creates an object that either appends to an existing file, or
  /// writes to a new file (if the file does not exist to begin with).
  ///
  /// On success, stores a pointer to the new file in *result and
  /// returns OK.  On failure stores NULL in *result and returns
  /// non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used.
  virtual Status NewAppendableFile(const string& fname,
                                   std::unique_ptr<WritableFile>* result) = 0;

  /// \brief Creates a readonly region of memory with the file context.
  ///
  /// On success, it returns a pointer to read-only memory region
  /// from the content of file fname. The ownership of the region is passed to
  /// the caller. On failure stores nullptr in *result and returns non-OK.
  ///
  /// The returned memory region can be accessed from many threads in parallel.
  ///
  /// The ownership of the returned ReadOnlyMemoryRegion is passed to the caller
  /// and the object should be deleted when is not used.
  virtual Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) = 0;

  /// Returns OK if the named path exists and NOT_FOUND otherwise.
  virtual Status FileExists(const string& fname) = 0;

  /// Returns true if all the listed files exist, false otherwise.
  /// if status is not null, populate the vector with a detailed status
  /// for each file.
  virtual bool FilesExist(const std::vector<string>& files,
                          std::vector<Status>* status);

  /// \brief Returns the immediate children in the given directory.
  ///
  /// The returned paths are relative to 'dir'.
  virtual Status GetChildren(const string& dir,
                             std::vector<string>* result) = 0;

  /// \brief Given a pattern, stores in *results the set of paths that matches
  /// that pattern. *results is cleared.
  ///
  /// pattern must match all of a name, not just a substring.
  ///
  /// pattern: { term }
  /// term:
  ///   '*': matches any sequence of non-'/' characters
  ///   '?': matches a single non-'/' character
  ///   '[' [ '^' ] { match-list } ']':
  ///        matches any single character (not) on the list
  ///   c: matches character c (c != '*', '?', '\\', '[')
  ///   '\\' c: matches character c
  /// character-range:
  ///   c: matches character c (c != '\\', '-', ']')
  ///   '\\' c: matches character c
  ///   lo '-' hi: matches character c for lo <= c <= hi
  ///
  /// Typical return codes:
  ///  * OK - no errors
  ///  * UNIMPLEMENTED - Some underlying functions (like GetChildren) are not
  ///                    implemented
  /// The default implementation uses a combination of GetChildren, MatchPath
  /// and IsDirectory.
  virtual Status GetMatchingPaths(const string& pattern,
                                  std::vector<string>* results);

  /// \brief Obtains statistics for the given path.
  virtual Status Stat(const string& fname, FileStatistics* stat) = 0;

  /// \brief Deletes the named file.
  virtual Status DeleteFile(const string& fname) = 0;

  /// \brief Creates the specified directory.
  /// Typical return codes:
  ///  * OK - successfully created the directory.
  ///  * ALREADY_EXISTS - directory with name dirname already exists.
  ///  * PERMISSION_DENIED - dirname is not writable.
  virtual Status CreateDir(const string& dirname) = 0;

  /// \brief Creates the specified directory and all the necessary
  /// subdirectories.
  /// Typical return codes:
  ///  * OK - successfully created the directory and sub directories, even if
  ///         they were already created.
  ///  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  virtual Status RecursivelyCreateDir(const string& dirname);

  /// \brief Deletes the specified directory.
  virtual Status DeleteDir(const string& dirname) = 0;

  /// \brief Deletes the specified directory and all subdirectories and files
  /// underneath it. undeleted_files and undeleted_dirs stores the number of
  /// files and directories that weren't deleted (unspecified if the return
  /// status is not OK).
  /// REQUIRES: undeleted_files, undeleted_dirs to be not null.
  /// Typical return codes:
  ///  * OK - dirname exists and we were able to delete everything underneath.
  ///  * NOT_FOUND - dirname doesn't exist
  ///  * PERMISSION_DENIED - dirname or some descendant is not writable
  ///  * UNIMPLEMENTED - Some underlying functions (like Delete) are not
  ///                    implemented
  virtual Status DeleteRecursively(const string& dirname,
                                   int64* undeleted_files,
                                   int64* undeleted_dirs);

  /// \brief Stores the size of `fname` in `*file_size`.
  virtual Status GetFileSize(const string& fname, uint64* file_size) = 0;

  /// \brief Overwrites the target if it exists.
  virtual Status RenameFile(const string& src, const string& target) = 0;

  /// \brief Translate an URI to a filename for the FileSystem implementation.
  ///
  /// The implementation in this class cleans up the path, removing
  /// duplicate /'s, resolving .. and . (more details in
  /// tensorflow::lib::io::CleanPath).
  virtual string TranslateName(const string& name) const;

  /// \brief Returns whether the given path is a directory or not.
  ///
  /// Typical return codes (not guaranteed exhaustive):
  ///  * OK - The path exists and is a directory.
  ///  * FAILED_PRECONDITION - The path exists and is not a directory.
  ///  * NOT_FOUND - The path entry does not exist.
  ///  * PERMISSION_DENIED - Insufficient permissions.
  ///  * UNIMPLEMENTED - The file factory doesn't support directories.
  virtual Status IsDirectory(const string& fname);

  FileSystem() {}

  virtual ~FileSystem();
};

// A file abstraction for reading sequentially through a file
class SequentialFile {
 public:
  SequentialFile() { }
  virtual ~SequentialFile();

  // Read up to "n" bytes from the file.  "scratch[0..n-1]" may be
  // written by this routine.  Sets "*result" to the data that was
  // read (including if fewer than "n" bytes were successfully read).
  // May set "*result" to point at data in "scratch[0..n-1]", so
  // "scratch[0..n-1]" must be live when "*result" is used.
  // If an error was encountered, returns a non-OK status.
  //
  // REQUIRES: External synchronization
  virtual Status Read(size_t n, Slice* result, char* scratch) = 0;

  // Skip "n" bytes from the file. This is guaranteed to be no
  // slower that reading the same data, but may be faster.
  //
  // If end of file is reached, skipping will stop at the end of the
  // file, and Skip will return OK.
  //
  // REQUIRES: External synchronization
  virtual Status Skip(uint64_t n) = 0;

  // Indicates the upper layers if the current SequentialFile implementation
  // uses direct IO.
  virtual bool use_direct_io() const { return false; }

  // Use the returned alignment value to allocate
  // aligned buffer for Direct I/O
  virtual size_t GetRequiredBufferAlignment() const { return kDefaultPageSize; }

  // Remove any kind of caching of data from the offset to offset+length
  // of this file. If the length is 0, then it refers to the end of file.
  // If the system is not caching the file contents, then this is a noop.
  virtual Status InvalidateCache(size_t offset, size_t length) {
    return Status::NotSupported("InvalidateCache not supported.");
  }

  // Positioned Read for direct I/O
  // If Direct I/O enabled, offset, n, and scratch should be properly aligned
  virtual Status PositionedRead(uint64_t offset, size_t n, Slice* result,
                                char* scratch) {
    return Status::NotSupported();
  }
};

/// A file abstraction for randomly reading the contents of a file.
class RandomAccessFile {
 public:
  RandomAccessFile() {}
  virtual ~RandomAccessFile();

  /// \brief Reads up to `n` bytes from the file starting at `offset`.
  ///
  /// `scratch[0..n-1]` may be written by this routine.  Sets `*result`
  /// to the data that was read (including if fewer than `n` bytes were
  /// successfully read).  May set `*result` to point at data in
  /// `scratch[0..n-1]`, so `scratch[0..n-1]` must be live when
  /// `*result` is used.
  ///
  /// On OK returned status: `n` bytes have been stored in `*result`.
  /// On non-OK returned status: `[0..n]` bytes have been stored in `*result`.
  ///
  /// Returns `OUT_OF_RANGE` if fewer than n bytes were stored in `*result`
  /// because of EOF.
  ///
  /// Safe for concurrent use by multiple threads.
  virtual Status Read(uint64 offset, size_t n, StringPiece* result,
                      char* scratch) const = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomAccessFile);
};

// A file abstraction for random reading and writing.
class RandomRWFile {
 public:
  RandomRWFile() {}
  virtual ~RandomRWFile() {}

  // Indicates if the class makes use of direct I/O
  // If false you must pass aligned buffer to Write()
  virtual bool use_direct_io() const { return false; }

  // Use the returned alignment value to allocate
  // aligned buffer for Direct I/O
  virtual size_t GetRequiredBufferAlignment() const { return kDefaultPageSize; }

  // Write bytes in `data` at  offset `offset`, Returns Status::OK() on success.
  // Pass aligned buffer when use_direct_io() returns true.
  virtual Status Write(uint64_t offset, const Slice& data) = 0;

  // Read up to `n` bytes starting from offset `offset` and store them in
  // result, provided `scratch` size should be at least `n`.
  // Returns Status::OK() on success.
  virtual Status Read(uint64_t offset, size_t n, Slice* result,
                      char* scratch) const = 0;

  virtual Status Flush() = 0;

  virtual Status Sync() = 0;

  virtual Status Fsync() { return Sync(); }

  virtual Status Close() = 0;

  // No copying allowed
  RandomRWFile(const RandomRWFile&) = delete;
  RandomRWFile& operator=(const RandomRWFile&) = delete;
};

/// \brief A file abstraction for sequential writing.
///
/// The implementation must provide buffering since callers may append
/// small fragments at a time to the file.
class WritableFile {
 public:
  WritableFile() {}
  virtual ~WritableFile();

  /// \brief Append 'data' to the file.
  virtual Status Append(const StringPiece& data) = 0;

  /// \brief Close the file.
  ///
  /// Flush() and de-allocate resources associated with this file
  ///
  /// Typical return codes (not guaranteed to be exhaustive):
  ///  * OK
  ///  * Other codes, as returned from Flush()
  virtual Status Close() = 0;

  /// \brief Flushes the file and optionally syncs contents to filesystem.
  ///
  /// This should flush any local buffers whose contents have not been
  /// delivered to the filesystem.
  ///
  /// If the process terminates after a successful flush, the contents
  /// may still be persisted, since the underlying filesystem may
  /// eventually flush the contents.  If the OS or machine crashes
  /// after a successful flush, the contents may or may not be
  /// persisted, depending on the implementation.
  virtual Status Flush() = 0;

  /// \brief Syncs contents of file to filesystem.
  ///
  /// This waits for confirmation from the filesystem that the contents
  /// of the file have been persisted to the filesystem; if the OS
  /// or machine crashes after a successful Sync, the contents should
  /// be properly saved.
  virtual Status Sync() = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WritableFile);
};

/// \brief A readonly memmapped file abstraction.
///
/// The implementation must guarantee that all memory is accessible when the
/// object exists, independently from the Env that created it.
class ReadOnlyMemoryRegion {
 public:
  ReadOnlyMemoryRegion() {}
  virtual ~ReadOnlyMemoryRegion() = default;

  /// \brief Returns a pointer to the memory region.
  virtual const void* data() = 0;

  /// \brief Returns the length of the memory region in bytes.
  virtual uint64 length() = 0;
};

// START_SKIP_DOXYGEN

#ifndef SWIG
// Degenerate file system that provides no implementations.
class NullFileSystem : public FileSystem {
 public:
  NullFileSystem() {}

  ~NullFileSystem() override = default;

  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override {
    return errors::Unimplemented("NewRandomAccessFile unimplemented");
  }

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewWritableFile unimplemented");
  }

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewAppendableFile unimplemented");
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return errors::Unimplemented(
        "NewReadOnlyMemoryRegionFromFile unimplemented");
  }

  Status FileExists(const string& fname) override {
    return errors::Unimplemented("FileExists unimplemented");
  }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    return errors::Unimplemented("GetChildren unimplemented");
  }

  Status DeleteFile(const string& fname) override {
    return errors::Unimplemented("DeleteFile unimplemented");
  }

  Status CreateDir(const string& dirname) override {
    return errors::Unimplemented("CreateDir unimplemented");
  }

  Status DeleteDir(const string& dirname) override {
    return errors::Unimplemented("DeleteDir unimplemented");
  }

  Status GetFileSize(const string& fname, uint64* file_size) override {
    return errors::Unimplemented("GetFileSize unimplemented");
  }

  Status RenameFile(const string& src, const string& target) override {
    return errors::Unimplemented("RenameFile unimplemented");
  }

  Status Stat(const string& fname, FileStatistics* stat) override {
    return errors::Unimplemented("Stat unimplemented");
  }
};
#endif

// END_SKIP_DOXYGEN

/// \brief A registry for file system implementations.
///
/// Filenames are specified as an URI, which is of the form
/// [scheme://]<filename>.
/// File system implementations are registered using the REGISTER_FILE_SYSTEM
/// macro, providing the 'scheme' as the key.
class FileSystemRegistry {
 public:
  typedef std::function<FileSystem*()> Factory;

  virtual ~FileSystemRegistry();
  virtual Status Register(const string& scheme, Factory factory) = 0;
  virtual FileSystem* Lookup(const string& scheme) = 0;
  virtual Status GetRegisteredFileSystemSchemes(
      std::vector<string>* schemes) = 0;
};

}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_FILE_SYSTEM_H_