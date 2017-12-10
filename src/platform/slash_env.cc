/*
 * Copyright (c) 2011 The LevelDB Authors.
 * Copyright (c) 2015-2017 Carnegie Mellon University.
 *
 * All rights reserved.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file. See the AUTHORS file for names of contributors.
 */

// slash/slash/src/env.cc
// pdlfs-common/src/posix_env.cc
// pdlfs-common/src/posix_env.h

#include "platform/slash_env.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <deque>
#include <fstream>
#include <sstream>
#include <vector>
#include "platform/mutexlock.h"
#include "platform/platform.h"
#include "platform/slash_xdebug.h"

namespace bubblefs {
namespace myslash {

static int PthreadCall(const char* label, int result) {
  if (result != 0) {
    fprintf(stderr, "pthreadcall %s: %s\n", label, strerror(result));
    abort();
  }
  return result;
}  
  
/*
 *  Set the resource limits of a process
 */

/*
 *  0: success.
 * -1: set failed.
 * -2: get resource limits failed.
 */
const size_t kPageSize = getpagesize();

int SetMaxFileDescriptorNum(int64_t max_file_descriptor_num) {
  // Try to Set the number of file descriptor
  struct  rlimit limit;
  if (getrlimit(RLIMIT_NOFILE, &limit) != -1) {
    if (limit.rlim_cur < (rlim_t)max_file_descriptor_num) {
      // rlim_cur could be set by any user while rlim_max are
      // changeable only by root.
      limit.rlim_cur = max_file_descriptor_num;
      if(limit.rlim_cur > limit.rlim_max) {
        limit.rlim_max = max_file_descriptor_num;
      }
      if (setrlimit(RLIMIT_NOFILE, &limit) != -1) {
        return 0;
      } else {
        return -1;
      };
    } else {
      return 0;
    }
  } else {
    return -2;
  }
}


/*
 * size of initial mmap size
 */
size_t kMmapBoundSize = 1024 * 1024 * 4;

void SetMmapBoundSize(size_t size) {
  kMmapBoundSize = size;
}

static Status IOError(const std::string& context, int err_number) {
  return Status::IOError(context, strerror(err_number));
}

Status CreateDir(const char* dirname) {
  Status result;
  if (mkdir(dirname, 0755) != 0) {
    result = IOError(dirname, errno);
  }
  return result;
}

Status AttachDir(const char* dirname) {
  Status result;
  DIR* dir = opendir(dirname);
  if (dir == NULL) {
    result = IOError(dirname, errno);
  } else {
    closedir(dir);
  }
  return result;
}

bool FileExists(const char* fname) {
  return access(fname, F_OK) == 0;
}

Status DeleteFile(const char* fname) {
  Status result;
  if (unlink(fname) != 0) {
    result = IOError(fname, errno);
  }
  return result;
}

int DoCreatePath(const char *path, mode_t mode) {
  struct stat st;
  int status = 0;

  if (stat(path, &st) != 0) {
    /* Directory does not exist. EEXIST for race
     * condition */
    if (mkdir(path, mode) != 0 && errno != EEXIST)
      status = -1;
  } else if (!S_ISDIR(st.st_mode)) {
    errno = ENOTDIR;
    status = -1;
  }

  return (status);
}

/**
 ** CreatePath - ensure all directories in path exist
 ** Algorithm takes the pessimistic view and works top-down to ensure
 ** each directory in path exists, rather than optimistically creating
 ** the last element and working backwards.
 */
int CreatePath(const char* path, mode_t mode) {
  char           *pp;
  char           *sp;
  int             status;
  char           *copypath = strdup(path);

  status = 0;
  pp = copypath;
  while (status == 0 && (sp = strchr(pp, '/')) != 0) {
    if (sp != pp) {
      /* Neither root nor double slash in path */
      *sp = '\0';
      status = DoCreatePath(copypath, mode);
      *sp = '/';
    }
    pp = sp + 1;
  }
  if (status == 0)
    status = DoCreatePath(path, mode);
  free(copypath);
  return (status);
}

static int LockOrUnlock(int fd, bool lock) {
  errno = 0;
  struct flock f;
  memset(&f, 0, sizeof(f));
  f.l_type = (lock ? F_WRLCK : F_UNLCK);
  f.l_whence = SEEK_SET;
  f.l_start = 0;
  f.l_len = 0;        // Lock/unlock entire file
  return fcntl(fd, F_SETLK, &f);
}

Status LockFile(const std::string& fname, FileLock** lock) {
  *lock = NULL;
  Status result;
  int fd = open(fname.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd < 0) {
    result = IOError(fname, errno);
  } else if (LockOrUnlock(fd, true) == -1) {
    result = IOError("lock " + fname, errno);
    close(fd);
  } else {
    FileLock* my_lock = new FileLock;
    my_lock->fd_ = fd;
    my_lock->name_ = fname;
    *lock = my_lock;
  }
  return result;
}

Status UnlockFile(FileLock* lock) {
  Status result;
  if (LockOrUnlock(lock->fd_, false) == -1) {
    result = IOError("unlock", errno);
  }
  close(lock->fd_);
  delete lock;
  return result;
}

Status GetChildren(const char* dirname,
                   std::vector<std::string>* result) {
  result->clear();
  DIR* dir = opendir(dirname);
  if (dir != NULL) {
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
      result->push_back(static_cast<const char*>(entry->d_name));
    }
    closedir(dir);
    return Status::OK();
  } else {
    return IOError(dirname, errno);
  }
}

int GetChildren(const std::string& dir, std::vector<std::string>& result) {
  int res = 0;
  result.clear();
  DIR* d = opendir(dir.c_str());
  if (d == NULL) {
    return errno;
  }
  struct dirent* entry;
  while ((entry = readdir(d)) != NULL) {
    if (strcmp(entry->d_name, "..") == 0 || strcmp(entry->d_name, ".") == 0) {
      continue;
    }
    result.push_back(entry->d_name);
  }
  closedir(d);
  return res;
}

bool GetDescendant(const std::string& dir, std::vector<std::string>& result) {
  DIR* d = opendir(dir.c_str());
  if (d == NULL) {
    return false;
  }
  struct dirent* entry;
  std::string fname;
  while ((entry = readdir(d)) != NULL) {
    if (strcmp(entry->d_name, "..") == 0 || strcmp(entry->d_name, ".") == 0) {
      continue;
    }
    fname = dir + "/" + entry->d_name;
    if (IsDir(fname.c_str())) {
      if (!GetDescendant(fname, result)) {
        return false;
      }
    } else {
      result.push_back(fname);
    }
  }
  closedir(d);
  return true;
}

Status RenameFile(const char* src, const char* dst) {
  Status result;
  if (rename(src, dst) != 0) {
    result = IOError(src, errno);
  }
  return result;
}

Status GetFileSize(const char* fname, uint64_t* size) {
  Status s;
  struct stat sbuf;
  if (stat(fname, &sbuf) == 0) {
    *size = static_cast<uint64_t>(sbuf.st_size);
  } else {
    s = IOError(fname, errno);
    *size = 0;
  }
  return s;
}

Status CopyFile(const char* src, const char* dst) {
  Status status;
  int r = -1;
  int w = -1;
  if ((r = open(src, O_RDONLY)) == -1) {
    status = IOError(src, errno);
  }
  if (status.ok()) {
    if ((w = open(dst, O_CREAT | O_TRUNC | O_WRONLY, 0644)) == -1) {
      status = IOError(dst, errno);
    }
  }
  if (status.ok()) {
    ssize_t n;
    char buf[4096];
    while ((n = read(r, buf, 4096)) > 0) {
      ssize_t m = write(w, buf, n);
      if (m != n) {
        status = IOError(dst, errno);
        break;
      }
    }
    if (n == -1) {
      if (status.ok()) {
        status = IOError(src, errno);
      }
    }
  }
  if (r != -1) {
    close(r);
  }
  if (w != -1) {
    close(w);
  }
  return status;
}

#ifdef OS_LINUX
Status LinuxCopyFile(const char* src, const char* dst) {
  Status status;
  int r = -1;
  int w = -1;
  if ((r = open(src, O_RDONLY)) == -1) {
    status = IOError(src, errno);
  }
  if (status.ok()) {
    if ((w = open(dst, O_CREAT | O_TRUNC | O_WRONLY, 0644)) == -1) {
      status = IOError(dst, errno);
    }
  }
  if (status.ok()) {
    int p[2];
    if (pipe(p) == -1) {
      status = IOError("pipe", errno);
    } else {
      const size_t batch_size = 4096;
      while (splice(p[0], 0, w, 0, splice(r, 0, p[1], 0, batch_size, 0), 0) > 0)
        ;
      close(p[0]);
      close(p[1]);
    }
  }
  if (r != -1) {
    close(r);
  }
  if (w != -1) {
    close(w);
  }
  return status;
}
#else
Status LinuxCopyFile(const char* src, const char* dst) { }
#endif

bool IsDir(const char* path) {
  struct stat buf;
  int ret = stat(path, &buf);
  if (0 == ret) {
    if (buf.st_mode & S_IFDIR) {
      //folder
      return true;
    } else {
      //file
      return false;
    }
  }
  return false;
}

Status RemoveDir(const char* dirname) {
  Status result;
  if (rmdir(dirname) != 0) {
    result = IOError(dirname, errno);
  }
  return result;
}

Status DeleteDir(const char* path)
{
  char chBuf[256];
  DIR * dir = NULL;
  struct dirent *ptr;
  int ret = 0;
  dir = opendir(path);
  if (NULL == dir) {
    return IOError(path, errno);
  }
  while((ptr = readdir(dir)) != NULL) {
    ret = strcmp(ptr->d_name, ".");
    if (0 == ret) {
      continue;
    }
    ret = strcmp(ptr->d_name, "..");
    if (0 == ret) {
      continue;
    }
    snprintf(chBuf, 256, "%s/%s", path, ptr->d_name);
    if (IsDir(chBuf)) {
      //is dir
      if (!DeleteDir(chBuf).ok()) {
        return IOError(path, errno);
      }
    } else {
      //is file
      ret = remove(chBuf);
      if(0 != ret) {
        return IOError(path, errno);
      }
    }
  }
  (void)closedir(dir);
  ret = remove(path);
  if (0 != ret) {
    return IOError(path, errno);
  }
  return Status::OK();
}

bool DeleteDirIfExist(const char* path) {
  if (IsDir(path) && !DeleteDir(path).ok()) {
    return false;
  }
  return true;
}

uint64_t Du(const char* filename) {
  struct stat statbuf;
  uint64_t sum;
  if (lstat(filename, &statbuf) != 0) {
    return 0;
  }
  if (S_ISLNK(statbuf.st_mode) && stat(filename, &statbuf) != 0) {
    return 0;
  }
  sum = statbuf.st_size;
  if (S_ISDIR(statbuf.st_mode)) {
    DIR *dir = NULL;
    struct dirent *entry;
    std::string newfile;

    dir = opendir(filename);
    if (!dir) {
      return sum;
    }
    while ((entry = readdir(dir))) {
      if (strcmp(entry->d_name, "..") == 0 || strcmp(entry->d_name, ".") == 0) {
        continue;
      }
      newfile = std::string(filename) + "/" + entry->d_name;
      sum += Du(newfile.c_str());
    }
    closedir(dir);
  }
  return sum;
}

uint64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

void SleepForMicroseconds(int micros) {
  usleep(micros);
}

Status GetTestDirectory(std::string* result) {
  const char* env = getenv("TEST_TMPDIR");
  if (env == NULL || env[0] == '\0') {
    char buf[100];
    snprintf(buf, sizeof(buf), "/tmp/slashtest-%d",
             static_cast<int>(geteuid()));
    *result = buf;
  } else {
    *result = env;
  }
  // Ignore error since directory may exist
  CreateDir((*result).c_str());
  return Status::OK();
}

std::string RandomString(const int len) {
  char buf[len];
  for (int i = 0; i < len; i++) {
    buf[i] = Random::Uniform('z' - 'a') + 'a';
  }
  return std::string(buf, len);
}

int RandomSeed() {
  const char* env = getenv("TEST_RANDOM_SEED");
  int result = (env != NULL ? atoi(env) : 301);
  if (result <= 0) {
    result = 301;
  }
  return result;
}

Status FetchHostname(std::string* hostname) {
  char buf[_POSIX_HOST_NAME_MAX];
  if (gethostname(buf, sizeof(buf)) == -1) {
    return IOError("Cannot get hostname", errno);
  } else {
    *hostname = buf;
    return Status::OK();
  }
}

class PosixRandomAccessFile : public RandomAccessFile {
 private:
  std::string filename_;
  int fd_;

 public:
  PosixRandomAccessFile(const char* fname, int fd)
      : filename_(fname), fd_(fd) {}

  virtual ~PosixRandomAccessFile() { close(fd_); }

  virtual Status Read(uint64_t offset, size_t n, Slice* result,
                      char* scratch) const {
    Status s;
    ssize_t r = pread(fd_, scratch, n, static_cast<off_t>(offset));
    *result = Slice(scratch, static_cast<size_t>(r < 0 ? 0 : r));
    if (r < 0) {
      // An error: return a non-ok status
      s = IOError(filename_, errno);
    }
    return s;
  }
};

class PosixMmapReadableFile : public RandomAccessFile {
 private:
  std::string filename_;
  void* mmapped_region_;
  size_t length_;

 public:
  PosixMmapReadableFile(const char* fname, void* base, size_t length)
      : filename_(fname),
        mmapped_region_(base),
        length_(length) {}

  virtual ~PosixMmapReadableFile() {
    munmap(mmapped_region_, length_);
  }

  virtual Status Read(uint64_t offset, size_t n, Slice* result,
                      char* scratch) const {
    Status s;
    if (offset + n > length_) {
      *result = Slice();
      s = IOError(filename_, EINVAL);
    } else {
      *result = Slice(reinterpret_cast<char*>(mmapped_region_) + offset, n);
    }
    return s;
  }
};

class PosixSequentialFile: public SequentialFile {
 private:
  std::string filename_;
  FILE* file_;

 public:
  virtual void setUnBuffer() {
    setbuf(file_, NULL);
  }

  PosixSequentialFile(const std::string& fname, FILE* f)
      : filename_(fname), file_(f) { setbuf(file_, NULL); }

  virtual ~PosixSequentialFile() {
    if (file_) {
      fclose(file_);
    }
  }

  virtual Status Read(size_t n, Slice* result, char* scratch) override {
    Status s;
    size_t r = fread_unlocked(scratch, 1, n, file_);

    *result = Slice(scratch, r);

    if (r < n) {
      if (feof(file_)) {
        s = Status::EndFile(filename_, "end file");
        // We leave status as ok if we hit the end of the file
      } else {
        // A partial read with an error: return a non-ok status
        s = IOError(filename_, errno);
      }
    }
    return s;
  }

  virtual Status Skip(uint64_t n) override {
    if (fseek(file_, n, SEEK_CUR)) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  virtual char *ReadLine(char* buf, int n) override {
    return fgets(buf, n, file_);
  }

  virtual Status Close() {
    if (fclose(file_) != 0) {
      return IOError(filename_, errno);
    }
    file_ = NULL;
    return Status::OK();
  }
};

class PdlfsSequentialFile : public SequentialFile {
 private:
  std::string filename_;
  int fd_;

 public:
  PdlfsSequentialFile(const char* fname, int fd) : filename_(fname), fd_(fd) {}

  virtual ~PdlfsSequentialFile() { close(fd_); }

  // Read up to "n" bytes from the file.  "scratch[0..n-1]" may be
  // written by this routine.  Sets "*result" to the data that was
  // read (including if fewer than "n" bytes were successfully read).
  // May set "*result" to point at data in "scratch[0..n-1]", so
  // "scratch[0..n-1]" must be live when "*result" is used.
  // If an error was encountered, returns a non-OK status.
  //
  // REQUIRES: External synchronization
  virtual Status Read(size_t n, Slice* result, char* scratch) {
    Status s;
    ssize_t nr = read(fd_, scratch, n);
    if (nr == -1) {
      s = IOError(filename_, errno);
    } else if (nr != 0) {
      *result = Slice(scratch, static_cast<size_t>(nr));
    } else {  // EOF
      *result = Slice(scratch, 0);
    }

    return s;
  }

  // Skip "n" bytes from the file. This is guaranteed to be no
  // slower that reading the same data, but may be faster.
  //
  // If end of file is reached, skipping will stop at the end of the
  // file, and Skip will return OK.
  //
  // REQUIRES: External synchronization
  virtual Status Skip(uint64_t n) {
    off_t r = lseek(fd_, n, SEEK_CUR);
    if (r == -1) {
      return IOError(filename_, errno);
    } else {
      return Status::OK();
    }
  }
};


class PosixBufferedSequentialFile : public SequentialFile {
 private:
  std::string filename_;
  FILE* file_;

 public:
  PosixBufferedSequentialFile(const char* fname, FILE* f)
      : filename_(fname), file_(f) {}

  virtual ~PosixBufferedSequentialFile() { fclose(file_); }

  virtual Status Read(size_t n, Slice* result, char* scratch) {
    Status s;
    size_t r = fread_unlocked(scratch, 1, n, file_);
    *result = Slice(scratch, r);
    if (r < n) {
      if (feof(file_)) {
        // We leave status as ok if we hit the end of the file
      } else {
        // A partial read with an error: return a non-ok status
        s = IOError(filename_, errno);
      }
    }
    return s;
  }

  virtual Status Skip(uint64_t n) {
    if (fseek(file_, n, SEEK_CUR)) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }
};

// We preallocate up to an extra megabyte and use memcpy to append new
// data to the file.  This is safe since we either properly close the
// file before reading from it, or for log files, the reading code
// knows enough to skip zero suffixes.
class PosixMmapFile : public WritableFile
{
 private:
  std::string filename_;
  int fd_;
  size_t page_size_;
  size_t map_size_;       // How much extra memory to map at a time
  char* base_;            // The mapped region
  char* limit_;           // Limit of the mapped region
  char* dst_;             // Where to write next  (in range [base_,limit_])
  char* last_sync_;       // Where have we synced up to
  uint64_t file_offset_;  // Offset of base_ in file
  uint64_t write_len_;    // The data that written in the file


  // Have we done an munmap of unsynced data?
  bool pending_sync_;

  // Roundup x to a multiple of y
  static size_t Roundup(size_t x, size_t y) {
    return ((x + y - 1) / y) * y;
  }

  static size_t TrimDown(size_t x, size_t y) {
    return (x / y) * y;
  }
  size_t TruncateToPageBoundary(size_t s) {
    s -= (s & (page_size_ - 1));
    assert((s % page_size_) == 0);
    return s;
  }

  bool UnmapCurrentRegion() {
    bool result = true;
    if (base_ != NULL) {
      if (last_sync_ < limit_) {
        // Defer syncing this data until next Sync() call, if any
        pending_sync_ = true;
      }
      if (munmap(base_, limit_ - base_) != 0) {
        result = false;
      }
      file_offset_ += limit_ - base_;
      base_ = NULL;
      limit_ = NULL;
      last_sync_ = NULL;
      dst_ = NULL;

      // Increase the amount we map the next time, but capped at 1MB
      if (map_size_ < (1<<20)) {
        map_size_ *= 2;
      }
    }
    return result;
  }

  bool MapNewRegion() {
    assert(base_ == NULL);
    if (posix_fallocate(fd_, file_offset_, map_size_) != 0) {
      log_warn("ftruncate error");
      return false;
    }
    //log_info("map_size %d fileoffset %llu", map_size_, file_offset_);
    void* ptr = mmap(NULL, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED,
                     fd_, file_offset_);
    if (ptr == MAP_FAILED) {
      log_warn("mmap failed");
      return false;
    }
    base_ = reinterpret_cast<char*>(ptr);
    limit_ = base_ + map_size_;
    dst_ = base_ + write_len_;
    write_len_ = 0;
    last_sync_ = base_;
    return true;
  }

 public:
  PosixMmapFile(const std::string& fname, int fd, size_t page_size, uint64_t write_len = 0)
      : filename_(fname),
      fd_(fd),
      page_size_(page_size),
      map_size_(Roundup(kMmapBoundSize, page_size)),
      base_(NULL),
      limit_(NULL),
      dst_(NULL),
      last_sync_(NULL),
      file_offset_(0),
      write_len_(write_len),
      pending_sync_(false) {
        if (write_len_ != 0) {
          while (map_size_ < write_len_) {
            map_size_ += (1024 * 1024);
          }
        }
        assert((page_size & (page_size - 1)) == 0);
      }


  ~PosixMmapFile() {
    if (fd_ >= 0) {
      PosixMmapFile::Close();
    }
  }

  virtual Status Append(const Slice& data) {
    const char* src = data.data();
    size_t left = data.size();
    while (left > 0) {
      assert(base_ <= dst_);
      assert(dst_ <= limit_);
      size_t avail = limit_ - dst_;
      if (avail == 0) {
        if (!UnmapCurrentRegion() || !MapNewRegion()) {
          return IOError(filename_, errno);
        }
      }
      size_t n = (left <= avail) ? left : avail;
      memcpy(dst_, src, n);
      dst_ += n;
      src += n;
      left -= n;
    }
    return Status::OK();
  }

  virtual Status Close() {
    Status s;
    size_t unused = limit_ - dst_;
    if (!UnmapCurrentRegion()) {
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
    base_ = NULL;
    limit_ = NULL;
    return s;
  }

  virtual Status Flush() {
    return Status::OK();
  }

  virtual Status Sync() {
    Status s;

    if (pending_sync_) {
      // Some unmapped data was not synced
      pending_sync_ = false;
      if (fdatasync(fd_) < 0) {
        s = IOError(filename_, errno);
      }
    }

    if (dst_ > last_sync_) {
      // Find the beginnings of the pages that contain the first and last
      // bytes to be synced.
      size_t p1 = TruncateToPageBoundary(last_sync_ - base_);
      size_t p2 = TruncateToPageBoundary(dst_ - base_ - 1);
      last_sync_ = dst_;
      if (msync(base_ + p1, p2 - p1 + page_size_, MS_SYNC) < 0) {
        s = IOError(filename_, errno);
      }
    }

    return s;
  }

  virtual Status Trim(uint64_t target) {
    if (!UnmapCurrentRegion()) {
      return IOError(filename_, errno);
    }

    file_offset_ = target;

    if (!MapNewRegion()) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  virtual uint64_t Filesize() {
    return write_len_ + file_offset_ + (dst_ - base_);
  }
};

class PosixBufferedWritableFile : public WritableFile {
 private:
  std::string filename_;
  FILE* file_;

 public:
  PosixBufferedWritableFile(const char* fname, FILE* f)
      : filename_(fname), file_(f) {}

  virtual ~PosixBufferedWritableFile() {
    if (file_ != NULL) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  virtual Status Append(const Slice& data) {
    size_t r = fwrite_unlocked(data.data(), 1, data.size(), file_);
    if (r != data.size()) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  virtual Status Close() {
    Status result;
    if (fclose(file_) != 0) {
      result = IOError(filename_, errno);
    }
    file_ = NULL;
    return result;
  }

  virtual Status Flush() {
    if (fflush_unlocked(file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status SyncDirIfManifest() {
    const char* f = filename_.c_str();
    const char* sep = strrchr(f, '/');
    Slice basename;
    std::string dir;
    if (sep == NULL) {
      dir = ".";
      basename = f;
    } else {
      dir = std::string(f, sep - f);
      basename = sep + 1;
    }
    Status s;
    if (basename.starts_with("MANIFEST")) {
      int fd = open(dir.c_str(), O_RDONLY);
      if (fd < 0) {
        s = IOError(dir, errno);
      } else {
        if (fsync(fd) < 0) {
          s = IOError(dir, errno);
        }
        close(fd);
      }
    }
    return s;
  }

  virtual Status Sync() {
    // Ensure new files referred to by the manifest are in the file system.
    Status s = SyncDirIfManifest();
    if (!s.ok()) {
      return s;
    }
    if (fflush_unlocked(file_) != 0 || fdatasync(fileno(file_)) != 0) {
      s = Status::IOError(filename_, strerror(errno));
    }
    return s;
  }
};

class PosixWritableFile : public WritableFile {
 private:
  std::string filename_;
  int fd_;

 public:
  PosixWritableFile(const char* fname, int fd) : filename_(fname), fd_(fd) {}

  virtual ~PosixWritableFile() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

  virtual Status Append(const Slice& buf) {
    ssize_t nw = write(fd_, buf.data(), buf.size());
    if (nw != buf.size()) {
      return IOError(filename_, errno);
    } else {
      return Status::OK();
    }
  }

  virtual Status Close() {
    close(fd_);
    fd_ = -1;
    return Status::OK();
  }

  virtual Status Flush() {
    // Do nothing since we never buffer any data
    return Status::OK();
  }

  virtual Status SyncDirIfManifest() {
    const char* f = filename_.c_str();
    const char* sep = strrchr(f, '/');
    Slice basename;
    std::string dir;
    if (sep == NULL) {
      dir = ".";
      basename = f;
    } else {
      dir = std::string(f, sep - f);
      basename = sep + 1;
    }
    Status s;
    if (basename.starts_with("MANIFEST")) {
      int fd = open(dir.c_str(), O_RDONLY);
      if (fd < 0) {
        s = IOError(dir, errno);
      } else {
        if (fsync(fd) < 0) {
          s = IOError(dir, errno);
        }
        close(fd);
      }
    }
    return s;
  }

  virtual Status Sync() {
    // Ensure new files referred to by the manifest are in the file system.
    Status s = SyncDirIfManifest();
    if (!s.ok()) {
      return s;
    } else {
      int r = fdatasync(fd_);
      if (r != 0) {
        s = IOError(filename_, errno);
      }
    }

    return s;
  }
};

class MmapRWFile : public RWFile
{
 public:
   MmapRWFile(const std::string& fname, int fd, size_t page_size)
     : filename_(fname),
     fd_(fd),
     //page_size_(page_size),
     map_size_(Roundup(65536, page_size)),
     base_(NULL) {
       DoMapRegion();
     }

   ~MmapRWFile() {
     if (fd_ >= 0) {
       munmap(base_, map_size_);
     }
   }

   bool DoMapRegion() {
     if (posix_fallocate(fd_, 0, map_size_) != 0) {
       return false;
     }
     void* ptr = mmap(NULL, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
     if (ptr == MAP_FAILED) {
       return false;
     }
     base_ = reinterpret_cast<char*>(ptr);
     return true;
   }

   char* GetData() { return base_; }
   char* base() { return base_; }

 private:
   static size_t Roundup(size_t x, size_t y) {
     return ((x + y - 1) / y) * y;
   }
   std::string filename_;
   int fd_;
   //size_t page_size_;
   size_t map_size_;
   char* base_;
};

class PosixRandomRWFile : public RandomRWFile {
 private:
   const std::string filename_;
   int fd_;
   bool pending_sync_;
   bool pending_fsync_;
   //bool fallocate_with_keep_size_;

 public:
   PosixRandomRWFile(const std::string& fname, int fd)
     : filename_(fname),
     fd_(fd),
     pending_sync_(false),
     pending_fsync_(false) {
       //fallocate_with_keep_size_ = options.fallocate_with_keep_size;
     }

   ~PosixRandomRWFile() {
     if (fd_ >= 0) {
       Close();
     }
   }

   virtual Status Write(uint64_t offset, const Slice& data) override {
     const char* src = data.data();
     size_t left = data.size();
     Status s;
     pending_sync_ = true;
     pending_fsync_ = true;

     while (left != 0) {
       ssize_t done = pwrite(fd_, src, left, offset);
       if (done < 0) {
         if (errno == EINTR) {
         continue;
       }
       return IOError(filename_, errno);
     }

     left -= done;
     src += done;
     offset += done;
   }

   return Status::OK();
 }

 virtual Status Read(uint64_t offset, size_t n, Slice* result,
                     char* scratch) const override {
   Status s;
   ssize_t r = -1;
   size_t left = n;
   char* ptr = scratch;
   while (left > 0) {
     r = pread(fd_, ptr, left, static_cast<off_t>(offset));
     if (r <= 0) {
       if (errno == EINTR) {
         continue;
       }
       break;
     }
     ptr += r;
     offset += r;
     left -= r;
   }
   *result = Slice(scratch, (r < 0) ? 0 : n - left);
   if (r < 0) {
     s = IOError(filename_, errno);
   }
   return s;
 }

 virtual Status Close() override {
   Status s = Status::OK();
   if (fd_ >= 0 && close(fd_) < 0) {
     s = IOError(filename_, errno);
   }
   fd_ = -1;
   return s;
 }

 virtual Status Sync() override {
   if (pending_sync_ && fdatasync(fd_) < 0) {
     return IOError(filename_, errno);
   }
   pending_sync_ = false;
   return Status::OK();
 }

 virtual Status Fsync() override {
   if (pending_fsync_ && fsync(fd_) < 0) {
     return IOError(filename_, errno);
   }
   pending_fsync_ = false;
   pending_sync_ = false;
   return Status::OK();
 }

//  virtual Status Allocate(off_t offset, off_t len) override {
//    TEST_KILL_RANDOM(rocksdb_kill_odds);
//    int alloc_status = fallocate(
//        fd_, fallocate_with_keep_size_ ? FALLOC_FL_KEEP_SIZE : 0, offset, len);
//    if (alloc_status == 0) {
//      return Status::OK();
//    } else {
//      return IOError(filename_, errno);
//    }
//  }
};

Status NewSequentialFile(const std::string& fname, SequentialFile** result) {
  FILE* f = fopen(fname.c_str(), "r");
  if (f == NULL) {
    *result = NULL;
    return IOError(fname, errno);
  } else {
    *result = new PosixSequentialFile(fname, f);
    return Status::OK();
  }
}

Status NewWritableFile(const std::string& fname, WritableFile** result) {
  Status s;
  const int fd = open(fname.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
  if (fd < 0) {
    *result = NULL;
    s = IOError(fname, errno);
  } else {
    *result = new PosixMmapFile(fname, fd, kPageSize);
  }
  return s;
}

Status NewRWFile(const std::string& fname, RWFile** result) {
  Status s;
  const int fd = open(fname.c_str(), O_CREAT | O_RDWR, 0644);
  if (fd < 0) {
    *result = NULL;
    s = IOError(fname, errno);
  } else {
    *result = new MmapRWFile(fname, fd, kPageSize);
  }
  return s;
}

Status AppendWritableFile(const std::string& fname, WritableFile** result, uint64_t write_len) {
  Status s;
  const int fd = open(fname.c_str(), O_RDWR, 0644);
  if (fd < 0) {
    *result = NULL;
    s = IOError(fname, errno);
  } else {
    *result = new PosixMmapFile(fname, fd, kPageSize, write_len);
  }
  return s;
}

Status NewRandomRWFile(const std::string& fname, RandomRWFile** result) {
  Status s;
  const int fd = open(fname.c_str(), O_CREAT | O_RDWR, 0644);
  if (fd < 0) {
    *result = NULL;
    s = IOError(fname, errno);
  } else {
    *result = new PosixRandomRWFile(fname, fd);
  }
  return s;
}

Status NewRandomAccessFile(const char* fname, RandomAccessFile** r) {
  int fd = open(fname, O_RDONLY);
  if (fd != -1) {
    *r = new PosixRandomAccessFile(fname, fd);
    return Status::OK();
  } else {
    *r = NULL;
    return IOError(fname, errno);
  }
}

static pthread_t Pthread(void* (*func)(void*), void* arg, void* attr) {
  pthread_t th;
  pthread_attr_t* ta = reinterpret_cast<pthread_attr_t*>(attr);
  PthreadCall("pthread_create", pthread_create(&th, ta, func, arg));
  PthreadCall("pthread_detach", pthread_detach(th));
  return th;
}

class PosixFixedThreadPool : public ThreadPool {
 public:
  explicit PosixFixedThreadPool(int max_threads, bool eager_init = false,
                                void* attr = NULL)
      : bg_cv_(&mu_),
        num_pool_threads_(0),
        max_threads_(max_threads),
        shutting_down_(false),
        paused_(false) {
    if (eager_init) {
      // Create pool threads immediately
      MutexLock ml(&mu_);
      InitPool(attr);
    }
  }

  virtual ~PosixFixedThreadPool();
  virtual void Schedule(void (*function)(void*), void* arg);
  virtual std::string ToDebugString();
  virtual void Resume();
  virtual void Pause();

  void StartThread(void (*function)(void*), void* arg);
  void InitPool(void* attr);

 private:
  // BGThread() is the body of the background thread
  void BGThread();

  static void* BGWrapper(void* arg) {
    reinterpret_cast<PosixFixedThreadPool*>(arg)->BGThread();
    return NULL;
  }

  port::Mutex mu_;
  port::CondVar bg_cv_;
  int num_pool_threads_;
  int max_threads_;

  bool shutting_down_;
  bool paused_;

  // Entry per Schedule() call
  struct BGItem {
    void* arg;
    void (*function)(void*);
  };
  typedef std::deque<BGItem> BGQueue;
  BGQueue queue_;

  struct StartThreadState {
    void (*user_function)(void*);
    void* arg;
  };

  static void* StartThreadWrapper(void* arg) {
    StartThreadState* state = reinterpret_cast<StartThreadState*>(arg);
    state->user_function(state->arg);
    delete state;
    return NULL;
  }
};

std::string PosixFixedThreadPool::ToDebugString() {
  char tmp[100];
  snprintf(tmp, sizeof(tmp), "POSIX fixed thread pool: num_threads=%d",
           max_threads_);
  return tmp;
}

PosixFixedThreadPool::~PosixFixedThreadPool() {
  mu_.Lock();
  shutting_down_ = true;
  bg_cv_.SignalAll();
  while (num_pool_threads_ != 0) {
    bg_cv_.Wait();
  }
  mu_.Unlock();
}

void PosixFixedThreadPool::InitPool(void* attr) {
  mu_.AssertHeld();
  while (num_pool_threads_ < max_threads_) {
    num_pool_threads_++;
    Pthread(BGWrapper, this, attr);
  }
}

void PosixFixedThreadPool::Schedule(void (*function)(void*), void* arg) {
  MutexLock ml(&mu_);
  if (shutting_down_) return;
  InitPool(NULL);  // Start background threads if necessary

  // If the queue is currently empty, the background threads
  // may be waiting.
  if (queue_.empty()) bg_cv_.SignalAll();

  // Add to priority queue
  queue_.push_back(BGItem());
  queue_.back().function = function;
  queue_.back().arg = arg;
}

void PosixFixedThreadPool::BGThread() {
  void (*function)(void*) = NULL;
  void* arg;

  while (true) {
    {
      MutexLock l(&mu_);
      // Wait until there is an item that is ready to run
      while (!shutting_down_ && (paused_ || queue_.empty())) {
        bg_cv_.Wait();
      }
      if (shutting_down_) {
        assert(num_pool_threads_ > 0);
        num_pool_threads_--;
        bg_cv_.SignalAll();
        return;
      }

      assert(!queue_.empty());
      function = queue_.front().function;
      arg = queue_.front().arg;
      queue_.pop_front();
    }

    assert(function != NULL);
    function(arg);
  }
}

void PosixFixedThreadPool::Resume() {
  MutexLock ml(&mu_);
  paused_ = false;
  bg_cv_.SignalAll();
}

void PosixFixedThreadPool::Pause() {
  MutexLock ml(&mu_);
  paused_ = true;
}

void PosixFixedThreadPool::StartThread(void (*function)(void*), void* arg) {
  StartThreadState* state = new StartThreadState;
  state->user_function = function;
  state->arg = arg;
  Pthread(StartThreadWrapper, state, NULL);
}

ThreadPool* ThreadPool::NewFixed(int num_threads, bool eager_init, void* attr) {
  return new PosixFixedThreadPool(num_threads, eager_init, attr);
}

void Schedule(ThreadPool& pool, void (*function)(void*), void* arg) {
  pool.Schedule(function, arg);
}

void StartThread(void* (*function)(void*), void* arg) {
  Pthread(function, arg, NULL);
}

} // namespace myslash
} // namespace bubblefs