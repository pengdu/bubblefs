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
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// tensorflow/tensorflow/core/platform/env.cc
// tensorflow/tensorflow/core/platform/file_system.cc
// rocksdb/env/env.cc

#include "platform/env.h"
#include <sys/stat.h>
#include <unistd.h>
#include <deque>
#include <utility>
#include <vector>
#include "platform/port.h"
#include "platform/protobuf.h"
#include "utils/errors.h"
#include "utils/map_util.h"
#include "utils/path.h"
#include "utils/stl_util.h"
#include "utils/stringprintf.h"
#include "utils/threadpool.h"

namespace bubblefs {

namespace {

constexpr int kNumThreads = 8;

// Run a function in parallel using a ThreadPool, but skip the ThreadPool
// on the iOS platform due to its problems with more than a few threads.
void ForEach(int first, int last, const std::function<void(int)>& f) {
#if TARGET_OS_IPHONE
  for (int i = first; i < last; i++) {
    f(i);
  }
#else
  int num_threads = std::min(kNumThreads, last - first);
  thread::ThreadPool threads(Env::Default(), "ForEach", num_threads);
  for (int i = first; i < last; i++) {
    threads.Schedule([f, i] { f(i); });
  }
#endif
}

}  // anonymous namespace

string Env::TranslateName(const string& name) const {
  return io::CleanPath(name);
}

uint64_t Env::GetThreadID() const {
  std::hash<std::thread::id> hasher;
  return hasher(std::this_thread::get_id());
}

Status Env::ReuseWritableFile(const string& fname,
                              const string& old_fname,
                              std::unique_ptr<WritableFile>* result,
                              const EnvOptions& options) {
  Status s = RenameFile(old_fname, fname);
  if (!s.ok()) {
    return s;
  }
  return NewWritableFile(fname, result, options);
}

Status Env::GetChildrenFileAttributes(const string& dir,
                                      std::vector<FileAttributes>* result) {
  assert(result != nullptr);
  std::vector<string> child_fnames;
  Status s = GetChildren(dir, &child_fnames);
  if (!s.ok()) {
    return s;
  }
  result->resize(child_fnames.size());
  size_t result_size = 0;
  for (size_t i = 0; i < child_fnames.size(); ++i) {
    const string path = dir + "/" + child_fnames[i];
    s = GetFileSize(path, &(*result)[result_size].size_bytes);
    if (!s.ok()) {
      if (FileExists(path).IsNotFound()) {
        // The file may have been deleted since we listed the directory
        continue;
      }
      return s;
    }
    (*result)[result_size].name = std::move(child_fnames[i]);
    result_size++;
  }
  result->resize(result_size);
  return Status::OK();
}

bool Env::FilesExist(const std::vector<string>& files,
                     std::vector<Status>* status) {
  bool result = true;
  for (const auto& file : files) {
    Status s = FileExists(file);
    result &= s.ok();
    if (status != nullptr) {
      status->push_back(s);
    } else if (!result) {
      // Return early since there is no need to check other files.
      return false;
    }
  }
  return result;
}

Status Env::GetMatchingPaths(const string& pattern,
                             std::vector<string>* results) {
  results->clear();
  // Find the fixed prefix by looking for the first wildcard.
  string fixed_prefix = pattern.substr(0, pattern.find_first_of("*?[\\"));
  string eval_pattern = pattern;
  std::vector<string> all_files;
  string dir = io::Dirname(fixed_prefix).ToString();
  // If dir is empty then we need to fix up fixed_prefix and eval_pattern to
  // include . as the top level directory.
  if (dir.empty()) {
    dir = ".";
    fixed_prefix = io::JoinPath(dir, fixed_prefix);
    eval_pattern = io::JoinPath(dir, pattern);
  }

  // Setup a BFS to explore everything under dir.
  std::deque<string> dir_q;
  dir_q.push_back(dir);
  Status ret;  // Status to return.
  // children_dir_status holds is_dir status for children. It can have three
  // possible values: OK for true; FAILED_PRECONDITION for false; CANCELLED
  // if we don't calculate IsDirectory (we might do that because there isn't
  // any point in exploring that child path).
  std::vector<Status> children_dir_status;
  while (!dir_q.empty()) {
    string current_dir = dir_q.front();
    dir_q.pop_front();
    std::vector<string> children;
    Status s = GetChildren(current_dir, &children);
    ret.Update(s);
    if (children.empty()) continue;
    // This IsDirectory call can be expensive for some FS. Parallelizing it.
    children_dir_status.resize(children.size());
    ForEach(0, children.size(), [this, &current_dir, &children, &fixed_prefix,
                                 &children_dir_status](int i) {
      const string child_path = io::JoinPath(current_dir, children[i]);
      // In case the child_path doesn't start with the fixed_prefix then
      // we don't need to explore this path.
      if (!StringPiece(child_path).starts_with(fixed_prefix)) {
        children_dir_status[i] =
            Status(error::CANCELLED, "Operation not needed");
      } else {
        children_dir_status[i] = IsDirectory(child_path);
      }
    });
    for (std::size_t i = 0; i < children.size(); ++i) {
      const string child_path = io::JoinPath(current_dir, children[i]);
      // If the IsDirectory call was cancelled we bail.
      if (children_dir_status[i].code() == error::CANCELLED) {
        continue;
      }
      // If the child is a directory add it to the queue.
      if (children_dir_status[i].ok()) {
        dir_q.push_back(child_path);
      }
      all_files.push_back(child_path);
    }
  }

  // Match all obtained files to the input pattern.
  for (const auto& f : all_files) {
    if (Env::Default()->MatchPath(f, eval_pattern)) {
      results->push_back(f);
    }
  }
  return ret;
}

Status Env::DeleteRecursively(const string& dirname,
                              int64* undeleted_files,
                              int64* undeleted_dirs) {
  CHECK_NOTNULL(undeleted_files);
  CHECK_NOTNULL(undeleted_dirs);

  *undeleted_files = 0;
  *undeleted_dirs = 0;
  // Make sure that dirname exists;
  Status exists_status = FileExists(dirname);
  if (!exists_status.ok()) {
    (*undeleted_dirs)++;
    return exists_status;
  }
  std::deque<string> dir_q;      // Queue for the BFS
  std::vector<string> dir_list;  // List of all dirs discovered
  dir_q.push_back(dirname);
  Status ret;  // Status to be returned.
  // Do a BFS on the directory to discover all the sub-directories. Remove all
  // children that are files along the way. Then cleanup and remove the
  // directories in reverse order.;
  while (!dir_q.empty()) {
    string dir = dir_q.front();
    dir_q.pop_front();
    dir_list.push_back(dir);
    std::vector<string> children;
    // GetChildren might fail if we don't have appropriate permissions.
    Status s = GetChildren(dir, &children);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
      continue;
    }
    for (const string& child : children) {
      const string child_path = io::JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (IsDirectory(child_path).ok()) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        Status del_status = DeleteFile(child_path);
        ret.Update(del_status);
        if (!del_status.ok()) {
          (*undeleted_files)++;
        }
      }
    }
  }
  // Now reverse the list of directories and delete them. The BFS ensures that
  // we can delete the directories in this order.
  std::reverse(dir_list.begin(), dir_list.end());
  for (const string& dir : dir_list) {
    // Delete dir might fail because of permissions issues or might be
    // unimplemented.
    Status s = DeleteDir(dir);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
    }
  }
  return ret;
}

Status Env::RecursivelyCreateDir(const string& dirname) {
  StringPiece scheme, host, remaining_dir;
  io::ParseURI(dirname, &scheme, &host, &remaining_dir);
  std::vector<StringPiece> sub_dirs;
  while (!remaining_dir.empty()) {
    Status status = FileExists(io::CreateURI(scheme, host, remaining_dir));
    if (status.ok()) {
      break;
    }
    if (status.code() != error::Code::NOT_FOUND) {
      return status;
    }
    // Basename returns "" for / ending dirs.
    if (!remaining_dir.ends_with("/")) {
      sub_dirs.push_back(io::Basename(remaining_dir));
    }
    remaining_dir = io::Dirname(remaining_dir);
  }

  // sub_dirs contains all the dirs to be created but in reverse order.
  std::reverse(sub_dirs.begin(), sub_dirs.end());

  // Now create the directories.
  string built_path = remaining_dir.ToString();
  for (const StringPiece sub_dir : sub_dirs) {
    built_path = io::JoinPath(built_path, sub_dir);
    Status status = CreateDir(io::CreateURI(scheme, host, built_path));
    if (!status.ok() && status.code() != error::ALREADY_EXISTS) {
      return status;
    }
  }
  return Status::OK();
}

Status Env::IsDirectory(const string& name) {
  // Check if path exists.
  TF_RETURN_IF_ERROR(FileExists(name));
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(name, &stat));
  if (stat.is_directory) {
    return Status::OK();
  }
  return Status(error::FAILED_PRECONDITION, "Not a directory");
}

string Env::GetExecutablePath() {
  char exe_path[PATH_MAX] = {0};
#ifdef __APPLE__
  uint32_t buffer_size(0U);
  _NSGetExecutablePath(nullptr, &buffer_size);
  char unresolved_path[buffer_size];
  _NSGetExecutablePath(unresolved_path, &buffer_size);
  CHECK(realpath(unresolved_path, exe_path));
#elif defined(PLATFORM_WINDOWS)
  HMODULE hModule = GetModuleHandleW(NULL);
  WCHAR wc_file_path[MAX_PATH] = {0};
  GetModuleFileNameW(hModule, wc_file_path, MAX_PATH);
  string file_path = WindowsFileSystem::WideCharToUtf8(wc_file_path);
  std::copy(file_path.begin(), file_path.end(), exe_path);
#else
  CHECK_NE(-1, readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1));
#endif
  // Make sure it's null-terminated:
  exe_path[sizeof(exe_path) - 1] = 0;

  return exe_path;
}

bool Env::LocalTempFilename(string* filename) {
  std::vector<string> dirs;
  GetLocalTempDirectories(&dirs);

  // Try each directory, as they might be full, have inappropriate
  // permissions or have different problems at times.
  // return std::string(".org.chromium.Chromium.XXXXXX");

  for (const string& dir : dirs) {
#ifdef __APPLE__
    uint64_t tid64;
    pthread_threadid_np(nullptr, &tid64);
    int32 tid = static_cast<int32>(tid64);
    int32 pid = static_cast<int32>(getpid());
#elif defined(PLATFORM_WINDOWS)
    int32 tid = static_cast<int32>(GetCurrentThreadId());
    int32 pid = static_cast<int32>(GetCurrentProcessId());
#else
    int32 tid = static_cast<int32>(pthread_self());
    int32 pid = static_cast<int32>(getpid());
#endif
    uint64 now_microsec = NowMicros();

    *filename = io::JoinPath(
        dir, strings::Printf("tempfile-%s-%x-%d-%lx", port::Hostname().c_str(),
                             tid, pid, now_microsec));
    if (FileExists(*filename).ok()) {
      filename->clear();
    } else {
      return true;
    }
  }
  return false;
}

Status ReadFileToString(Env* env, const string& fname, string* data) {
  uint64 file_size;
  Status s = env->GetFileSize(fname, &file_size);
  if (!s.ok()) {
    return s;
  }
  std::unique_ptr<RandomAccessFile> file;
  s = env->NewRandomAccessFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  gtl::STLStringResizeUninitialized(data, file_size);
  char* p = gtl::string_as_array(data);
  StringPiece result;
  s = file->Read(0, file_size, &result, p);
  if (!s.ok()) {
    data->clear();
  } else if (result.size() != file_size) {
    s = errors::Aborted("File ", fname, " changed while reading: ", file_size,
                        " vs. ", result.size());
    data->clear();
  } else if (result.data() == p) {
    // Data is already in the correct location
  } else {
    memmove(p, result.data(), result.size());
  }
  return s;
}

Status WriteStringToFile(Env* env, const string& fname,
                         const StringPiece& data) {
  std::unique_ptr<WritableFile> file;
  Status s = env->NewWritableFile(fname, &file);
  if (!s.ok()) {
    return s;
  }
  s = file->Append(data);
  if (s.ok()) {
    s = file->Close();
  }
  return s;
}

// A ZeroCopyInputStream on a RandomAccessFile.
namespace {
class FileStream : public protobuf::io::ZeroCopyInputStream {
 public:
  explicit FileStream(RandomAccessFile* file) : file_(file), pos_(0) {}

  void BackUp(int count) override { pos_ -= count; }
  bool Skip(int count) override {
    pos_ += count;
    return true;
  }
  protobuf_int64 ByteCount() const override { return pos_; }
  Status status() const { return status_; }

  bool Next(const void** data, int* size) override {
    StringPiece result;
    Status s = file_->Read(pos_, kBufSize, &result, scratch_);
    if (result.empty()) {
      status_ = s;
      return false;
    }
    pos_ += result.size();
    *data = result.data();
    *size = result.size();
    return true;
  }

 private:
  static const int kBufSize = 512 << 10;

  RandomAccessFile* file_;
  int64 pos_;
  Status status_;
  char scratch_[kBufSize];
};

}  // namespace

Status WriteBinaryProto(Env* env, const string& fname,
                        const protobuf::MessageLite& proto) {
  string serialized;
  proto.AppendToString(&serialized);
  return WriteStringToFile(env, fname, serialized);
}

Status ReadBinaryProto(Env* env, const string& fname,
                       protobuf::MessageLite* proto) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
  std::unique_ptr<FileStream> stream(new FileStream(file.get()));

  // TODO(jiayq): the following coded stream is for debugging purposes to allow
  // one to parse arbitrarily large messages for MessageLite. One most likely
  // doesn't want to put protobufs larger than 64MB on Android, so we should
  // eventually remove this and quit loud when a large protobuf is passed in.
  protobuf::io::CodedInputStream coded_stream(stream.get());
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively.
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);

  if (!proto->ParseFromCodedStream(&coded_stream)) {
    TF_RETURN_IF_ERROR(stream->status());
    return errors::DataLoss("Can't parse ", fname, " as binary proto");
  }
  return Status::OK();
}

Status WriteTextProto(Env* env, const string& fname,
                      const protobuf::Message& proto) {
  string serialized;
  if (!protobuf::TextFormat::PrintToString(proto, &serialized)) {
    return errors::FailedPrecondition("Unable to convert proto to text.");
  }
  return WriteStringToFile(env, fname, serialized);
}

Status ReadTextProto(Env* env, const string& fname,
                     protobuf::Message* proto) {
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(fname, &file));
  std::unique_ptr<FileStream> stream(new FileStream(file.get()));

  if (!protobuf::TextFormat::Parse(stream.get(), proto)) {
    TF_RETURN_IF_ERROR(stream->status());
    return errors::DataLoss("Can't parse ", fname, " as text proto");
  }
  return Status::OK();
}

}  // namespace bubblefs