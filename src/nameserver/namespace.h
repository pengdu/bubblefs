
#ifndef BUBBLEFS_NAMESERVER_NAMESPACE_H_
#define BUBBLEFS_NAMESERVER_NAMESPACE_H_

#include <stdint.h>
#include <string>
#include <functional>
#include "db/db.h"
#include "platform/macros.h"
#include "platform/mutexlock.h"

#include "proto/file.pb.h"
#include "proto/nameserver.pb.h"
#include "proto/status_code.pb.h"

namespace bubblefs {
namespace bfs {
  
class Namespace {
 public:
  NameSpace(bool standalone = true);
  ~NameSpace();
  void Activate(std::function<void (const FileInfo&)> rebuild_callback, 
                nameserver::NameServerLog* log = nullptr);
  /// fs ops
  int64_t Version() const;
  void EraseNamespace();
  static std::string NormalizePath(const std::string& path);
  StatusCode ListDirectory(const std::string& path,
                           google::protobuf::RepeatedPtrField<FileInfo>* outputs);
  StatusCode CreateFile(const std::string& file_name, int flags, int mode,
                        int replica_num, std::vector<int64_t>* blocks_to_remove,
                        nameserver::NameServerLog* log = nullptr);
  StatusCode RemoveFile(const std::string& path, FileInfo* file_removed, 
                        nameserver::NameServerLog* log = nullptr);
  StatusCode DeleteDirectory(const std::string& path, bool recursive, std::vector<FileInfo>* files_removed, 
                             nameserver::NameServerLog* log = nullptr);
  StatusCode DiskUsage(const std::string& path, uint64_t* du_size);
  StatusCode Rename(const std::string& old_path, const std::string& new_path,
                    bool* need_unlink, FileInfo* remove_file,
                    nameserver::NameServerLog* log = nullptr);
  StatusCode Symlink(const std::string& src,
                     const std::string& dst,
                     nameserver::NameServerLog* log = nullptr);
  bool GetFileInfo(const std::string& path, FileInfo* file_info);
  bool UpdateFileInfo(const FileInfo& file_info,
                      nameserver::NameServerLog* log = nullptr);
  bool DeleteFileInfo(const std::string file_key,
                      nameserver::NameServerLog* log = nullptr);
  StatusCode GetDirLockStatus(const std::string& path);
  void SetDirLockStatus(const std::string& path, StatusCode status,
                        const std::string& uuid = "");
  /// block ops
  bool RebuildBlockMap(std::function<void (const FileInfo&)> callback);
  int64_t GetNewBlockId();
  void ListAllBlocks(const std::string& path, std::vector<int64_t>* result);
  /// ha - tail log from leader/master
  void TailLog(const std::string& log);
  void TailSnapshot(int32_t ns_id, std::string* logstr);
                 
 private:
   enum FileType {
     kDefault = 0,
     kDir = 1,
     kSymlink = 2,
   };
   
   static void EncodingStoreKey(int64_t entry_id,
                                const std::string& path,
                                std::string* key_str);
   static void DecodingStoreKey(const std::string& key_str,
                                int64_t* entry_id,
                                std::string* path);
   bool GetFromStore(const std::string& key, FileInfo* info);
   void SetupRoot();
   FileType GetFileType(int type) const;
   StatusCode BuildPath(const std::string& path, FileInfo* file_info, std::string* fname,
                        nameserver::NameServerLog* log = nullptr);
   bool LookUp(const std::string& path, FileInfo* info);
   bool LookUp(int64_t pid, const std::string& name, FileInfo* info);
   StatusCode InternalDeleteDirectory(const FileInfo& dir_info, bool recursive, 
                                      std::vector<FileInfo>* files_removed,
                                      nameserver::NameServerLog* log);
   StatusCode InternalComputeDiskUsage(const FileInfo& info, uint64_t* du_size);
   bool GetLinkSrcPath(const FileInfo& info, FileInfo* src_info);
   uint32_t EncodeLog(nameserver::NameServerLog* log, int32_t type,
                      const std::string& key, const std::string& value);
   void InitBlockIdUpbound(nameserver::NameServerLog* log);
   void UpdateBlockIdUpbound(nameserver::NameServerLog* log);
   
 private:
   leveldb::DB* db_;   /// NameSpace storage
   leveldb::Cache* db_cache_;  /// block cache for leveldb
   int64_t version_;   /// Namespace version
   volatile int64_t last_entry_id_;
   FileInfo root_path_;
   int64_t block_id_upbound_;
   int64_t next_block_id_;
   port::Mutex mu_;
   std::map<int32_t, leveldb::Iterator*> snapshot_tasks_; /// HA module
   
 private:
   DISALLOW_COPY_AND_ASSIGN(Namespace);
  
};  
  
} // namespace bfs  
} // namespace bubblefs

#endif // BUBBLEFS_NAMESERVER_NAMESPACE_H_