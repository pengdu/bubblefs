/*
 * Copyright (c) 2011 The LevelDB Authors.
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// slash/slash/include/env.h

#ifndef BUBBLEFS_PLATFORM_SLASH_ENV_H_
#define BUBBLEFS_PLATFORM_SLASH_ENV_H_

#include <unistd.h>
#include <string>
#include <vector>
#include "platform/mutexlock.h"
#include "utils/slash_random.h"
#include "utils/slash_status.h"

namespace bubblefs {
namespace myslash { 
  
class RandomAccessFile;
class WritableFile;
class SequentialFile;
class RWFile;
class RandomRWFile;
class ThreadPool;

/*
 *  Set the resource limits of a process
 */
int SetMaxFileDescriptorNum(int64_t max_file_descriptor_num);

/*
 * Set size of initial mmap size
 */
void SetMmapBoundSize(size_t size);

extern const size_t kPageSize;

/*
 * File Operations
 */
bool IsDir(const char* path);

// Delete the specified directory.
Status RemoveDir(const char* dir);
Status DeleteDir(const char* path);
bool DeleteDirIfExist(const char* path);

// Create the specified directory.
Status CreateDir(const char* dir);

// Load a directory created by another process.
// This call makes no sense for Env implementations that are backed by a posix
// file system but is usually needed for implementations that are backed by a
// shared object storage service.
// Return OK if the directory is attached and ready to use.
// For Env implementations that are backed by a posix file system, return OK
// if the directory exists.
Status AttachDir(const char* dir);

int CreatePath(const char* path, mode_t mode = 0755);
uint64_t Du(const char* path);

/*
 * Whether the file is exist
 * If exist return true, else return false
 */
bool FileExists(const char* f);

// Delete the named file.
Status DeleteFile(const char* f);

// Rename file src to dst.
Status RenameFile(const char* src, const char* dst);

// Store the size of the named file in *file_size.
Status GetFileSize(const char* f, uint64_t* file_size);
// Copy file src to dst.
Status CopyFile(const char* src, const char* dst);
Status LinuxCopyFile(const char* src, const char* dst);

class FileLock {
  public:
    FileLock() { }
    virtual ~FileLock() {};

    int fd_;
    std::string name_;

  private:

    // No copying allowed
    FileLock(const FileLock&);
    void operator=(const FileLock&);
};

Status LockFile(const std::string& f, FileLock** l);
Status UnlockFile(FileLock* l);

// Store in *r the names of the children of the specified directory.
// The names are relative to "dir".
// Original contents of *r are dropped.
Status GetChildren(const char* dir, std::vector<std::string>* r);
int GetChildren(const std::string& dir, std::vector<std::string>& result);
bool GetDescendant(const std::string& dir, std::vector<std::string>& result);

uint64_t NowMicros();
void SleepForMicroseconds(int micros);

// *path is set to a temporary directory that can be used for testing. It may
// or many not have just been created. The directory may or may not differ
// between runs of the same process, but subsequent calls will return the
// same directory.
Status GetTestDirectory(std::string* path);

std::string RandomString(const int len);
int RandomSeed();

// Obtain the network name of the local machine.
Status FetchHostname(std::string* hostname);

Status NewSequentialFile(const std::string& fname, SequentialFile** result);

Status NewWritableFile(const std::string& fname, WritableFile** result);

Status NewRWFile(const std::string& fname, RWFile** result);

Status AppendSequentialFile(const std::string& fname, SequentialFile** result);

Status AppendWritableFile(const std::string& fname, WritableFile** result, uint64_t write_len = 0);

Status NewRandomRWFile(const std::string& fname, RandomRWFile** result);

// Create a brand new random access read-only file with the specified name.
// On success, stores a pointer to the new file in *r and returns OK.
// On failure stores NULL in *r and returns non-OK.  If the file does not
// exist, returns a non-OK status.
//
// The returned file may be concurrently accessed by multiple threads.
Status NewRandomAccessFile(const char* f, RandomAccessFile** r);

// Arrange to run "(*function)(arg)" once in a background thread.
//
// "function" may run in an unspecified thread.  Multiple functions
// added to the same Env may run concurrently in different threads.
// I.e., the caller may not assume that background work items are
// serialized.
void Schedule(ThreadPool& pool, void (*function)(void*), void* arg);

// Start a new thread, invoking "function(arg)" within the new thread.
// When "function(arg)" returns, the thread will be destroyed.
void StartThread(void* (*function)(void*), void* arg);

// A file abstraction for randomly reading the contents of a file.
class RandomAccessFile {
 public:
  RandomAccessFile() { }
  virtual ~RandomAccessFile() { };

  // Read up to "n" bytes from the file starting at "offset".
  // "scratch[0..n-1]" may be written by this routine.  Sets "*result"
  // to the data that was read (including if fewer than "n" bytes were
  // successfully read).  May set "*result" to point at data in
  // "scratch[0..n-1]", so "scratch[0..n-1]" must be live when
  // "*result" is used.  If an error was encountered, returns a non-OK
  // status.
  //
  // Safe for concurrent use by multiple threads.
  virtual Status Read(uint64_t offset, size_t n, Slice* result,
                      char* scratch) const = 0;

 private:
  // No copying allowed
  RandomAccessFile(const RandomAccessFile&);
  void operator=(const RandomAccessFile&);
};

// A file abstraction for sequential writing.  The implementation
// must provide buffering since callers may append small fragments
// at a time to the file.
class WritableFile {
 public:
  WritableFile() { }
  virtual ~WritableFile() { };

  virtual Status Append(const Slice& data) = 0;
  virtual Status Close() = 0;
  virtual Status Flush() = 0;
  virtual Status Sync() = 0;
  virtual Status Trim(uint64_t offset) { return Status::NotSupported("Trim is not supported"); };
  virtual uint64_t Filesize() { return 0; };

 private:
  // No copying allowed
  WritableFile(const WritableFile&);
  void operator=(const WritableFile&);
};

// A abstract for the sequential readable file
class SequentialFile {
 public:
  SequentialFile() {};
  virtual ~SequentialFile() { };
  //virtual Status Read(size_t n, char *&result, char *scratch) = 0;
  virtual Status Read(size_t n, Slice* result, char* scratch) = 0;
  virtual Status Skip(uint64_t n) = 0;
  //virtual Status Close() = 0;
  virtual char *ReadLine(char *buf, int n) { return nullptr; };
};

class RWFile {
public:
  RWFile() { }
  virtual ~RWFile() { };
  virtual char* GetData() = 0;

private:
  // No copying allowed
  RWFile(const RWFile&);
  void operator=(const RWFile&);
};

// A file abstraction for random reading and writing.
class RandomRWFile {
 public:
  RandomRWFile() {}
  virtual ~RandomRWFile() {}

  // Write data from Slice data to file starting from offset
  // Returns IOError on failure, but does not guarantee
  // atomicity of a write.  Returns OK status on success.
  //
  // Safe for concurrent use.
  virtual Status Write(uint64_t offset, const Slice& data) = 0;
  // Read up to "n" bytes from the file starting at "offset".
  // "scratch[0..n-1]" may be written by this routine.  Sets "*result"
  // to the data that was read (including if fewer than "n" bytes were
  // successfully read).  May set "*result" to point at data in
  // "scratch[0..n-1]", so "scratch[0..n-1]" must be live when
  // "*result" is used.  If an error was encountered, returns a non-OK
  // status.
  //
  // Safe for concurrent use by multiple threads.
  virtual Status Read(uint64_t offset, size_t n, Slice* result,
                      char* scratch) const = 0;
  virtual Status Close() = 0; // closes the file
  virtual Status Sync() = 0; // sync data

  /*
   * Sync data and/or metadata as well.
   * By default, sync only data.
   * Override this method for environments where we need to sync
   * metadata as well.
   */
  virtual Status Fsync() {
    return Sync();
  }

  /*
   * Pre-allocate space for a file.
   */
  virtual Status Allocate(off_t offset, off_t len) {
    (void)offset;
    (void)len;
    return Status::OK();
  }

 private:
  // No copying allowed
  RandomRWFile(const RandomRWFile&);
  void operator=(const RandomRWFile&);
};

// Background execution service.
class ThreadPool {
 public:
  ThreadPool() {}
  virtual ~ThreadPool();
  
  // start a new thread.
  void StartThread(void (*function)(void*), void* arg);

  // Instantiate a new thread pool with a fixed number of threads. The caller
  // should delete the pool to free associated resources.
  // If "eager_init" is true, children threads will be created immediately.
  // A caller may optionally set "attr" to alter default thread behaviour.
  static ThreadPool* NewFixed(int num_threads, bool eager_init = false,
                              void* attr = NULL);

  // Arrange to run "(*function)(arg)" once in one of a pool of
  // background threads.
  //
  // "function" may run in an unspecified thread.  Multiple functions
  // added to the same pool may run concurrently in different threads.
  // I.e., the caller may not assume that background work items are
  // serialized.
  virtual void Schedule(void (*function)(void*), void* arg) = 0;

  // Return a description of the pool implementation.
  virtual std::string ToDebugString() = 0;

  // Stop executing any tasks. Tasks already scheduled will keep running. Tasks
  // not yet scheduled won't be scheduled. Tasks submitted in future will be
  // queued but won't be scheduled.
  virtual void Pause() = 0;

  // Resume executing tasks.
  virtual void Resume() = 0;

 private:
  // No copying allowed
  void operator=(const ThreadPool&);
  ThreadPool(const ThreadPool&);
};

} // namespace myslash
} // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_SLASH_ENV_H_