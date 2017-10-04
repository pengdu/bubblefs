
#ifndef  BUBBLEFS_API_BFS_H_
#define  BUBBLEFS_API_BFS_H_

#include <fcntl.h>
#include <stdint.h>
#include <map>
#include <string>
#include <vector>
#include "api/error.h"
#include "platform/macros.h"

namespace bubblefs {
namespace bfs {

enum class WriteMode {
  kWriteDefault,      // use write strategy specified by flag file by default
  kWriteChains,
  kWriteFanout,
};

struct WriteOptions {
  int flush_timeout;  // in ms, <= 0 means do not timeout, == 0 means do not wait
  int sync_timeout;   // in ms, <= 0 means do not timeout, == 0 means do not wait
  int close_timeout;  // in ms, <= 0 means do not timeout, == 0 means do not wait
  int replica;
  WriteMode write_mode;
  bool sync_on_close; // flush data to disk when closing the file
    
  WriteOptions() : flush_timeout(-1), sync_timeout(-1),
                   close_timeout(-1), replica(-1), write_mode(WriteMode::kWriteDefault),
                   sync_on_close(false) { }
};

struct ReadOptions {
  int timeout;    // in ms, <= 0 means do not timeout, == 0 means do not wait
  ReadOptions() : timeout(-1) {}
};

struct FSOptions {
  const char* username;
  const char* passwd;
  FSOptions() : username(nullptr), passwd(nullptr) {}
};

/// Bfs File interface
class File {
public:
  File() {}
  virtual ~File() {}
  virtual int32_t Pread(char* buf, int32_t read_size, int64_t offset, bool reada = false) = 0;
  //for files opened with O_WRONLY, only support Seek(0, SEEK_CUR)
  virtual int64_t Seek(int64_t offset, int32_t whence) = 0;
  virtual int32_t Read(char* buf, int32_t read_size) = 0;
  virtual int32_t Write(const char* buf, int32_t write_size) = 0;
  virtual int32_t Flush() = 0;
  virtual int32_t Sync() = 0;
  virtual int32_t Close() = 0;
private:
  DISALLOW_COPY_AND_ASSIGN(File);
};

struct BfsFileInfo {
  int64_t size;
  uint32_t ctime;
  uint32_t mode;
  char name[1024];
  char link[1024];
};

// Bfs fileSystem interface
class FS {
public:
  FS() { }
  virtual ~FS() { }
  // Open filesystem with nameserver address (host:port)
  static bool OpenFileSystem(const char* nameserver, FS** fs, const FSOptions&);
  // Create directory
  virtual int32_t CreateDirectory(const char* path) = 0;
  // List Directory
  virtual int32_t ListDirectory(const char* path, BfsFileInfo** filelist, int *num) = 0;
  // Delete Directory
  virtual int32_t DeleteDirectory(const char* path, bool recursive) = 0;
  // Lock Directory
  virtual int32_t LockDirectory(const char* path) = 0;
  // Unlock Directory
  virtual int32_t UnlockDirectory(const char* path) = 0;
  // Du
  virtual int32_t DiskUsage(const char* path, int64_t* du_size) = 0;
  // Access
  virtual int32_t Access(const char* path, int32_t mode) = 0;
  // Stat
  virtual int32_t Stat(const char* path, BfsFileInfo* fileinfo) = 0;
  // GetFileSize: get real file size
  virtual int32_t GetFileSize(const char* path, int64_t* file_size) = 0;
  // Open file for read or write, flags: O_WRONLY or O_RDONLY
  virtual int32_t OpenFile(const char* path, int32_t flags, File** file,
                           const ReadOptions& options) = 0;
  virtual int32_t OpenFile(const char* path, int32_t flags, File** file,
                           const WriteOptions& options) = 0;
  virtual int32_t OpenFile(const char* path, int32_t flags, int32_t mode,
                           File** file, const WriteOptions& options) = 0;
  virtual int32_t CloseFile(File* file) = 0;
  virtual int32_t DeleteFile(const char* path) = 0;
  virtual int32_t Rename(const char* oldpath, const char* newpath) = 0;
  virtual int32_t Chmod(int32_t mode, const char* path) = 0;
  virtual int32_t ChangeReplicaNum(const char* file_name, int32_t replica_num) = 0;
  // Create symlink
  virtual int32_t Symlink(const char* oldpath, const char* newpath) = 0;
  // Show system status
  virtual int32_t SysStat(const std::string& stat_name, std::string* result) = 0;
  // GetFileLocation: get file locate, return chunkserver address and port
  virtual int32_t GetFileLocation(const std::string& path,
                                  std::map<int64_t, std::vector<std::string> >* locations) = 0;
  virtual int32_t ShutdownChunkServer(const std::vector<std::string>& cs_address) = 0;
  virtual int32_t ShutdownChunkServerStat() = 0;
    
private:
  DISALLOW_COPY_AND_ASSIGN(FS);
};

} // namespace bfs
} // namespace bubblefs

#endif  // BUBBLEFS_API_BFS_H_