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

// tensorflow/tensorflow/core/platform/env.h
// rocksdb/include/rocksdb/env.h

#ifndef BUBBLEFS_PLATFORM_ENV_H_
#define BUBBLEFS_PLATFORM_ENV_H_

#include <stdint.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "platform/env_time.h"
#include "platform/file_system.h"
#include "platform/macros.h"
#include "platform/mutex.h"
#include "platform/protobuf.h"
#include "platform/types.h"
#include "utils/errors.h"
#include "utils/status.h"
#include "utils/stringpiece.h"

namespace bubblefs {

class Thread;
struct ThreadOptions;

const size_t kDefaultPageSize = 4 * 1024;

// Options while opening a file to read/write
struct EnvOptions {

  // Construct with default Options
  EnvOptions();

  // Construct from Options
  explicit EnvOptions(const DBOptions& options);

   // If true, then use mmap to read data
  bool use_mmap_reads = false;

   // If true, then use mmap to write data
  bool use_mmap_writes = true;

  // If true, then use O_DIRECT for reading data
  bool use_direct_reads = false;

  // If true, then use O_DIRECT for writing data
  bool use_direct_writes = false;

  // If false, fallocate() calls are bypassed
  bool allow_fallocate = true;

  // If true, set the FD_CLOEXEC on open fd.
  bool set_fd_cloexec = true;

  // Allows OS to incrementally sync files to disk while they are being
  // written, in the background. Issue one request for every bytes_per_sync
  // written. 0 turns it off.
  // Default: 0
  uint64_t bytes_per_sync = 0;

  // If true, we will preallocate the file with FALLOC_FL_KEEP_SIZE flag, which
  // means that file size won't change as part of preallocation.
  // If false, preallocation will also change the file size. This option will
  // improve the performance in workloads where you sync the data on every
  // write. By default, we set it to true for MANIFEST writes and false for
  // WAL writes
  bool fallocate_with_keep_size = true;

  // See DBOptions doc
  size_t compaction_readahead_size;

  // See DBOptions doc
  size_t random_access_max_buffer_size;

  // See DBOptions doc
  size_t writable_file_max_buffer_size = 1024 * 1024;

  // If not nullptr, write rate limiting is enabled for flush and compaction
  //RateLimiter* rate_limiter = nullptr;
};

/// \brief An interface used by the tensorflow implementation to
/// access operating system functionality like the filesystem etc.
///
/// Callers may wish to provide a custom Env object to get fine grain
/// control.
///
/// All Env implementations are safe for concurrent access from
/// multiple threads without any external synchronization.
class Env {
 public:
  struct FileAttributes {
    // File name
    string name;
    // Size of file in bytes
    uint64_t size_bytes;
  };
   
  Env();
  virtual ~Env() = default;

  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static Env* Default();

  /// \brief Returns the FileSystem object to handle operations on the file
  /// specified by 'fname'. The FileSystem object is used as the implementation
  /// for the file system related (non-virtual) functions that follow.
  /// Returned FileSystem object is still owned by the Env object and will
  // (might) be destroyed when the environment is destroyed.
  virtual Status GetFileSystemForFile(const string& fname, FileSystem** result);

  /// \brief Returns the file system schemes registered for this Env.
  virtual Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes);

  // \brief Register a file system for a scheme.
  virtual Status RegisterFileSystem(const string& scheme,
                                    FileSystemRegistry::Factory factory);
  
  // Create a brand new sequentially-readable file with the specified name.
  // On success, stores a pointer to the new file in *result and returns OK.
  // On failure stores nullptr in *result and returns non-OK.  If the file does
  // not exist, returns a non-OK status.
  //
  // The returned file will only be accessed by one thread at a time.
  virtual Status NewSequentialFile(const string& fname,
                                   std::unique_ptr<SequentialFile>* result,
                                   const EnvOptions& options)
                                   = 0;

  /// \brief Creates a brand new random access read-only file with the
  /// specified name.

  /// On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.  If the file does not exist, returns a non-OK
  /// status.
  ///
  /// The returned file may be concurrently accessed by multiple threads.
  ///
  /// The ownership of the returned RandomAccessFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  virtual Status NewRandomAccessFile(const string& fname,
                             std::unique_ptr<RandomAccessFile>* result);

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
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  virtual Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result);

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
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  virtual Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result);

  /// \brief Creates a readonly region of memory with the file context.
  ///
  /// On success, it returns a pointer to read-only memory region
  /// from the content of file fname. The ownership of the region is passed to
  /// the caller. On failure stores nullptr in *result and returns non-OK.
  ///
  /// The returned memory region can be accessed from many threads in parallel.
  ///
  /// The ownership of the returned ReadOnlyMemoryRegion is passed to the caller
  /// and the object should be deleted when is not used. The memory region
  /// object shouldn't live longer than the Env object.
  virtual Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result);
  
  // Create an object that writes to a new file with the specified
  // name.  Deletes any existing file with the same name and creates a
  // new file.  On success, stores a pointer to the new file in
  // *result and returns OK.  On failure stores nullptr in *result and
  // returns non-OK.
  //
  // The returned file will only be accessed by one thread at a time.
  virtual Status ReopenWritableFile(const std::string& fname,
                                    std::unique_ptr<WritableFile>* result,
                                    const EnvOptions& options) {
    return Status::NotSupported();
  }
  
  
  // Reuse an existing file by renaming it and opening it as writable.
  virtual Status ReuseWritableFile(const std::string& fname,
                                   const std::string& old_fname,
                                   std::unique_ptr<WritableFile>* result,
                                   const EnvOptions& options);

  // Open `fname` for random read and write, if file doesn't exist the file
  // will be created.  On success, stores a pointer to the new file in
  // *result and returns OK.  On failure returns non-OK.
  //
  // The returned file will only be accessed by one thread at a time.
  virtual Status NewRandomRWFile(const std::string& fname,
                                 std::unique_ptr<RandomRWFile>* result,
                                 const EnvOptions& options) {
    return Status::NotSupported("RandomRWFile is not implemented in this Env");
  }

  /// Returns OK if the named path exists and NOT_FOUND otherwise.
  virtual Status FileExists(const string& fname);

  /// Returns true if all the listed files exist, false otherwise.
  /// if status is not null, populate the vector with a detailed status
  /// for each file.
  virtual bool FilesExist(const std::vector<string>& files,
                  std::vector<Status>* status);

  /// \brief Stores in *result the names of the children of the specified
  /// directory. The names are relative to "dir".
  ///
  /// Original contents of *results are dropped.
  virtual Status GetChildren(const string& dir, std::vector<string>* result);
  
  // Store in *result the attributes of the children of the specified directory.
  // In case the implementation lists the directory prior to iterating the files
  // and files are concurrently deleted, the deleted files will be omitted from
  // result.
  // The name attributes are relative to "dir".
  // Original contents of *results are dropped.
  // Returns OK if "dir" exists and "*result" contains its children.
  //         NotFound if "dir" does not exist, the calling process does not have
  //                  permission to access "dir", or if "dir" is invalid.
  //         IOError if an IO Error was encountered
  virtual Status GetChildrenFileAttributes(const std::string& dir,
                                           std::vector<FileAttributes>* result);


  /// \brief Returns true if the path matches the given pattern. The wildcards
  /// allowed in pattern are described in FileSystem::GetMatchingPaths.
  virtual bool MatchPath(const string& path, const string& pattern) = 0;

  /// \brief Given a pattern, stores in *results the set of paths that matches
  /// that pattern. *results is cleared.
  ///
  /// More details about `pattern` in FileSystem::GetMatchingPaths.
  virtual Status GetMatchingPaths(const string& pattern,
                                  std::vector<string>* results);

  /// Deletes the named file.
  virtual Status DeleteFile(const string& fname);

  /// \brief Deletes the specified directory and all subdirectories and files
  /// underneath it. undeleted_files and undeleted_dirs stores the number of
  /// files and directories that weren't deleted (unspecified if the return
  /// status is not OK).
  /// REQUIRES: undeleted_files, undeleted_dirs to be not null.
  /// Typical return codes
  ///  * OK - dirname exists and we were able to delete everything underneath.
  ///  * NOT_FOUND - dirname doesn't exist
  ///  * PERMISSION_DENIED - dirname or some descendant is not writable
  ///  * UNIMPLEMENTED - Some underlying functions (like Delete) are not
  ///                    implemented
  virtual Status DeleteRecursively(const string& dirname, int64* undeleted_files,
                           int64* undeleted_dirs);

  /// \brief Creates the specified directory and all the necessary
  /// subdirectories. Typical return codes.
  ///  * OK - successfully created the directory and sub directories, even if
  ///         they were already created.
  ///  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  virtual Status RecursivelyCreateDir(const string& dirname);

  /// \brief Creates the specified directory. Typical return codes
  ///  * OK - successfully created the directory.
  ///  * ALREADY_EXISTS - directory already exists.
  ///  * PERMISSION_DENIED - dirname is not writable.
  virtual Status CreateDir(const string& dirname);

  /// Deletes the specified directory.
  virtual Status DeleteDir(const string& dirname);

  /// Obtains statistics for the given path.
  virtual Status Stat(const string& fname, FileStatistics* stat);

  /// \brief Returns whether the given path is a directory or not.
  /// Typical return codes (not guaranteed exhaustive):
  ///  * OK - The path exists and is a directory.
  ///  * FAILED_PRECONDITION - The path exists and is not a directory.
  ///  * NOT_FOUND - The path entry does not exist.
  ///  * PERMISSION_DENIED - Insufficient permissions.
  ///  * UNIMPLEMENTED - The file factory doesn't support directories.
  virtual Status IsDirectory(const string& fname);

  /// Stores the size of `fname` in `*file_size`.
  virtual Status GetFileSize(const string& fname, uint64* file_size);
  
  // Store the last modification time of fname in *file_mtime.
  virtual Status GetFileModificationTime(const string& fname,
                                         uint64_t* file_mtime) = 0;

  /// \brief Renames file src to target. If target already exists, it will be
  /// replaced.
  virtual Status RenameFile(const string& src, const string& target);
  
  // Hard Link file src to target.
  virtual Status LinkFile(const std::string& src, const std::string& target) {
    return Status::NotSupported("LinkFile is not supported for this Env");
  }

  /// \brief Returns the absolute path of the current executable. It resolves
  /// symlinks if there is any.
  virtual string GetExecutablePath();

  /// Creates a local unique temporary file name. Returns true if success.
  virtual bool LocalTempFilename(string* filename);

  // TODO(jeff,sanjay): Add back thread/thread-pool support if needed.
  // TODO(jeff,sanjay): if needed, tighten spec so relative to epoch, or
  // provide a routine to get the absolute time.

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64 NowMicros() { return envTime->NowMicros(); };

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64 NowSeconds() { return envTime->NowSeconds(); }

  /// Sleeps/delays the thread for the prescribed number of micro-seconds.
  virtual void SleepForMicroseconds(int64 micros) = 0;
  
  // Lock the specified file.  Used to prevent concurrent access to
  // the same db by multiple processes.  On failure, stores nullptr in
  // *lock and returns non-OK.
  //
  // On success, stores a pointer to the object that represents the
  // acquired lock in *lock and returns OK.  The caller should call
  // UnlockFile(*lock) to release the lock.  If the process exits,
  // the lock will be automatically released.
  //
  // If somebody else already holds the lock, finishes immediately
  // with a failure.  I.e., this call does not wait for existing locks
  // to go away.
  //
  // May create the named file if it does not already exist.
  virtual Status LockFile(const std::string& fname, FileLock** lock) = 0;

  // Release the lock acquired by a previous successful call to LockFile.
  // REQUIRES: lock was returned by a successful LockFile() call
  // REQUIRES: lock has not already been unlocked.
  virtual Status UnlockFile(FileLock* lock) = 0;

  // Priority for scheduling job in thread pool
  enum Priority { BOTTOM, LOW, HIGH, TOTAL };

  // Priority for requesting bytes in rate limiter scheduler
  enum IOPriority {
    IO_LOW = 0,
    IO_HIGH = 1,
    IO_TOTAL = 2
  };

  // Arrange to run "(*function)(arg)" once in a background thread, in
  // the thread pool specified by pri. By default, jobs go to the 'LOW'
  // priority thread pool.

  // "function" may run in an unspecified thread.  Multiple functions
  // added to the same Env may run concurrently in different threads.
  // I.e., the caller may not assume that background work items are
  // serialized.
  // When the UnSchedule function is called, the unschedFunction
  // registered at the time of Schedule is invoked with arg as a parameter.
  virtual void Schedule(void (*function)(void* arg), void* arg,
                        Priority pri = LOW, void* tag = nullptr,
                        void (*unschedFunction)(void* arg) = 0) = 0;

  // Arrange to remove jobs for given arg from the queue_ if they are not
  // already scheduled. Caller is expected to have exclusive lock on arg.
  virtual int UnSchedule(void* arg, Priority pri) { return 0; }

  // Start a new thread, invoking "function(arg)" within the new thread.
  // When "function(arg)" returns, the thread will be destroyed.
  virtual void StartThread(void (*function)(void* arg), void* arg) = 0;

  // Wait for all threads started by StartThread to terminate.
  virtual void WaitForJoin() {}  
  
  // Get thread pool queue length for specific thread pool.
  virtual unsigned int GetThreadPoolQueueLen(Priority pri = LOW) const {
    return 0;
  }

  // *path is set to a temporary directory that can be used for testing. It may
  // or many not have just been created. The directory may or may not differ
  // between runs of the same process, but subsequent calls will return the
  // same directory.
  virtual Status GetTestDirectory(std::string* path) = 0;
  
 // Get the current host name.
  virtual Status GetHostName(char* name, uint64_t len) = 0;

  // Get the number of seconds since the Epoch, 1970-01-01 00:00:00 (UTC).
  // Only overwrites *unix_time on success.
  virtual Status GetCurrentTime(int64_t* unix_time) = 0;

  // Get full directory name for this db.
  virtual Status GetAbsolutePath(const std::string& db_path,
      std::string* output_path) = 0;

  // The number of background worker threads of a specific thread pool
  // for this environment. 'LOW' is the default pool.
  // default number: 1
  virtual void SetBackgroundThreads(int number, Priority pri = LOW) = 0;
  virtual int GetBackgroundThreads(Priority pri = LOW) = 0;

  // Enlarge number of background worker threads of a specific thread pool
  // for this environment if it is smaller than specified. 'LOW' is the default
  // pool.
  virtual void IncBackgroundThreadsIfNeeded(int number, Priority pri) = 0;

  // Lower IO priority for threads from the specified pool.
  virtual void LowerThreadPoolIOPriority(Priority pool = LOW) {}

  // Converts seconds-since-Jan-01-1970 to a printable string
  virtual std::string TimeToString(uint64_t time) = 0;

  // Generates a unique id that can be used to identify a db
  virtual std::string GenerateUniqueId();  
  
  // Returns the ID of the current thread.
  virtual uint64_t GetThreadID() const;
  
  /// \brief Returns a new thread that is running fn() and is identified
  /// (for debugging/performance-analysis) by "name".
  ///
  /// Caller takes ownership of the result and must delete it eventually
  /// (the deletion will block until fn() stops running).
  virtual Thread* StartThread(const ThreadOptions& thread_options,
                              const string& name,
                              std::function<void()> fn) TF_MUST_USE_RESULT = 0;

  // \brief Schedules the given closure on a thread-pool.
  //
  // NOTE(mrry): This closure may block.
  virtual void SchedClosure(std::function<void()> closure) = 0;

  // \brief Schedules the given closure on a thread-pool after the given number
  // of microseconds.
  //
  // NOTE(mrry): This closure must not block.
  virtual void SchedClosureAfter(int64 micros,
                                 std::function<void()> closure) = 0;

  // \brief Load a dynamic library.
  //
  // Pass "library_filename" to a platform-specific mechanism for dynamically
  // loading a library.  The rules for determining the exact location of the
  // library are platform-specific and are not documented here.
  //
  // On success, returns a handle to the library in "*handle" and returns
  // OK from the function.
  // Otherwise returns nullptr in "*handle" and an error status from the
  // function.
  virtual Status LoadLibrary(const char* library_filename, void** handle) = 0;

  // \brief Get a pointer to a symbol from a dynamic library.
  //
  // "handle" should be a pointer returned from a previous call to LoadLibrary.
  // On success, store a pointer to the located symbol in "*symbol" and return
  // OK from the function. Otherwise, returns nullptr in "*symbol" and an error
  // status from the function.
  virtual Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                      void** symbol) = 0;

  // \brief build the name of dynamic library.
  //
  // "name" should be name of the library.
  // "version" should be the version of the library or NULL
  // returns the name that LoadLibrary() can use
  virtual string FormatLibraryFileName(const string& name,
      const string& version) = 0;

 private:
  // Returns a possible list of local temporary directories.
  void GetLocalTempDirectories(std::vector<string>* list);

  std::unique_ptr<FileSystemRegistry> file_system_registry_;
  TF_DISALLOW_COPY_AND_ASSIGN(Env);
  EnvTime* envTime = EnvTime::Default();
};

/// \brief An implementation of Env that forwards all calls to another Env.
///
/// May be useful to clients who wish to override just part of the
/// functionality of another Env.
class EnvWrapper : public Env {
 public:
  /// Initializes an EnvWrapper that delegates all calls to *t
  explicit EnvWrapper(Env* t) : target_(t) {}
  virtual ~EnvWrapper();

  /// Returns the target to which this Env forwards all calls
  Env* target() const { return target_; }

  Status GetFileSystemForFile(const string& fname,
                              FileSystem** result) override {
    return target_->GetFileSystemForFile(fname, result);
  }

  Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes) override {
    return target_->GetRegisteredFileSystemSchemes(schemes);
  }

  Status RegisterFileSystem(const string& scheme,
                            FileSystemRegistry::Factory factory) override {
    return target_->RegisterFileSystem(scheme, factory);
  }

  bool MatchPath(const string& path, const string& pattern) override {
    return target_->MatchPath(path, pattern);
  }

  uint64 NowMicros() override { return target_->NowMicros(); }
  void SleepForMicroseconds(int64 micros) override {
    target_->SleepForMicroseconds(micros);
  }
  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      std::function<void()> fn) override {
    return target_->StartThread(thread_options, name, fn);
  }
  void SchedClosure(std::function<void()> closure) override {
    target_->SchedClosure(closure);
  }
  void SchedClosureAfter(int64 micros, std::function<void()> closure) override {
    target_->SchedClosureAfter(micros, closure);
  }
  Status LoadLibrary(const char* library_filename, void** handle) override {
    return target_->LoadLibrary(library_filename, handle);
  }
  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              void** symbol) override {
    return target_->GetSymbolFromLibrary(handle, symbol_name, symbol);
  }
  string FormatLibraryFileName(const string& name,
                               const string& version) override {
    return target_->FormatLibraryFileName(name, version);
  }
 private:
  Env* target_;
};

/// Represents a thread used to run a Tensorflow function.
class Thread {
 public:
  Thread() {}

  /// Blocks until the thread of control stops running.
  virtual ~Thread();

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(Thread);
};

/// \brief Options to configure a Thread.
///
/// Note that the options are all hints, and the
/// underlying implementation may choose to ignore it.
struct ThreadOptions {
  /// Thread stack size to use (in bytes).
  size_t stack_size = 0;  // 0: use system default value
  /// Guard area size to use near thread stacks to use (in bytes)
  size_t guard_size = 0;  // 0: use system default value
};

/// A utility routine: reads contents of named file into `*data`
Status ReadFileToString(Env* env, const string& fname, string* data);

/// A utility routine: write contents of `data` to file named `fname`
/// (overwriting existing contents, if any).
Status WriteStringToFile(Env* env, const string& fname,
                         const StringPiece& data);

/// Write binary representation of "proto" to the named file.
Status WriteBinaryProto(Env* env, const string& fname,
                        const protobuf::MessageLite& proto);

/// Reads contents of named file and parse as binary encoded proto data
/// and store into `*proto`.
Status ReadBinaryProto(Env* env, const string& fname,
                       protobuf::MessageLite* proto);

/// Write the text representation of "proto" to the named file.
Status WriteTextProto(Env* env, const string& fname,
                      const protobuf::Message& proto);

/// Read contents of named file and parse as text encoded proto data
/// and store into `*proto`.
Status ReadTextProto(Env* env, const string& fname,
                     protobuf::Message* proto);

// START_SKIP_DOXYGEN

namespace register_file_system {

template <typename Factory>
struct Register {
  Register(Env* env, const string& scheme) {
    // TODO(b/32704451): Don't just ignore the ::bubblefs::Status object!
    env->RegisterFileSystem(scheme, []() -> FileSystem* { return new Factory; })
        .IgnoreError();
  }
};

}  // namespace register_file_system

// END_SKIP_DOXYGEN

}  // namespace bubblefs

// Register a FileSystem implementation for a scheme. Files with names that have
// "scheme://" prefixes are routed to use this implementation.
#define REGISTER_FILE_SYSTEM_ENV(env, scheme, factory) \
  REGISTER_FILE_SYSTEM_UNIQ_HELPER(__COUNTER__, env, scheme, factory)
#define REGISTER_FILE_SYSTEM_UNIQ_HELPER(ctr, env, scheme, factory) \
  REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory)
#define REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory)   \
  static ::bubblefs::register_file_system::Register<factory> \
      register_ff##ctr TF_ATTRIBUTE_UNUSED =                   \
          ::bubblefs::register_file_system::Register<factory>(env, scheme)

#define REGISTER_FILE_SYSTEM(scheme, factory) \
  REGISTER_FILE_SYSTEM_ENV(Env::Default(), scheme, factory);

#endif  // BUBBLEFS_PLATFORM_ENV_H_