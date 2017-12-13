/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Paddle/paddle/utils/Util.cpp

#include "utils/paddle_util.h"

#include <dirent.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <mutex>

#include "platform/paddle_custom_stacktrace.h"
#include "platform/paddle_threadlocal.h"
#include "utils/paddle_string_util.h"
#include "utils/paddle_thread.h"

#include "gflags/gflags.h"

DEFINE_int32(seed, 1, "random number seed. 0 for srand(time)");

namespace bubblefs {
namespace mypaddle {

pid_t getTID() {
#if defined(__APPLE__) || defined(__OSX__)
  // syscall is deprecated: first deprecated in macOS 10.12.
  // syscall is unsupported;
  // syscall pid_t tid = syscall(SYS_thread_selfid);
  uint64_t tid;
  pthread_threadid_np(NULL, &tid);
#else
#ifndef __NR_gettid
#define __NR_gettid 224
#endif
  pid_t tid = syscall(__NR_gettid);
#endif
  PANIC_ENFORCE_NE((int)tid, -1);
  return tid;
}

static bool g_initialized = false;
typedef std::pair<int, std::function<void()>> PriorityFuncPair;
typedef std::vector<PriorityFuncPair> InitFuncList;
static InitFuncList* g_initFuncs = nullptr;
static std::once_flag g_onceFlag;
void registerInitFunction(std::function<void()> func, int priority) {
  if (g_initialized) {
    PANIC("registerInitFunction() should only called before initMain()");
  }
  if (!g_initFuncs) {
    g_initFuncs = new InitFuncList();
  }
  g_initFuncs->push_back(std::make_pair(priority, func));
}

void runInitFunctions() {
  std::call_once(g_onceFlag, []() {
    //VLOG(3) << "Calling runInitFunctions";
    if (g_initFuncs) {
      std::sort(g_initFuncs->begin(),
                g_initFuncs->end(),
                [](const PriorityFuncPair& x, const PriorityFuncPair& y) {
                  return x.first > y.first;
                });
      for (auto& f : *g_initFuncs) {
        f.second();
      }
      delete g_initFuncs;
      g_initFuncs = nullptr;
    }
    g_initialized = true;
    //VLOG(3) << "Call runInitFunctions done.";
  });
}

/*
void initMain(int argc, char** argv) {
  installLayerStackTracer();
  std::string line;
  for (int i = 0; i < argc; ++i) {
    line += argv[i];
    line += ' ';
  }

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  initializeLogging(argc, argv);
  LOG(INFO) << "commandline: " << line;
  CHECK_EQ(argc, 1) << "Unknown commandline argument: " << argv[1];

  installProfilerSwitch();

#ifdef __SSE__
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif
#ifdef __SSE3__
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

  if (FLAGS_seed == 0) {
    unsigned int t = time(NULL);
    srand(t);
    ThreadLocalRand::initSeed(t);
    LOG(INFO) << "random number seed=" << t;
  } else {
    srand(FLAGS_seed);
    ThreadLocalRand::initSeed(FLAGS_seed);
  }

  if (FLAGS_use_gpu) {
    // This is the initialization of the CUDA environment,
    // need before runInitFunctions.
    // TODO(hedaoyuan) Can be considered in the runInitFunctions,
    // but to ensure that it is the first to initialize.
    hl_start();
    hl_init(FLAGS_gpu_id);
  }

  version::printVersion();
  checkCPUFeature().check();
  runInitFunctions();
}
*/

std::string readFile(const std::string& fileName) {
  std::ifstream is(fileName);

  // get length of file:
  is.seekg(0, is.end);
  size_t length = is.tellg();
  is.seekg(0, is.beg);
  std::string str(length, (char)0);
  PANIC_ENFORCE(is.read(&str[0], length), "Fail to read file: %s", fileName.c_str());
  return str;
}

namespace path {

std::string basename(const std::string& path) {
  size_t pos = path.rfind(sep);
  ++pos;
  return path.substr(pos, std::string::npos);
}

std::string dirname(const std::string& path) {
  size_t pos = path.rfind(sep);
  if (pos == std::string::npos) return std::string();
  return path.substr(0, pos);
}

std::string join(const std::string& part1, const std::string& part2) {
  if (!part2.empty() && part2.front() == sep) {
    return part2;
  }
  std::string ret;
  ret.reserve(part1.size() + part2.size() + 1);
  ret = part1;
  if (!ret.empty() && ret.back() != sep) {
    ret += sep;
  }
  ret += part2;
  return ret;
}

}  // namespace path

void copyFileToPath(const std::string& file, const std::string& dir) {
  //VLOG(3) << "copy " << file << " to " << dir;
  std::string fileName = path::basename(file);
  std::string dst = path::join(dir, fileName);
  std::ifstream source(file, std::ios_base::binary);
  std::ofstream dest(dst, std::ios_base::binary);
  //CHECK(source) << "Fail to open " << file;
  //CHECK(dest) << "Fail to open " << dst;
  dest << source.rdbuf();
  source.close();
  dest.close();
}

bool fileExist(const char* filename) { return (access(filename, 0) == 0); }

void touchFile(const char* filename) {
  if (!fileExist(filename)) {
    std::ofstream os(filename);
  }
}

int isDir(const char* path) {
  struct stat s_buf;
  if (stat(path, &s_buf)) {
    return 0;
  }
  return S_ISDIR(s_buf.st_mode);
}

void rmDir(const char* folderName) {
  if (isDir(folderName)) {
    DIR* dp;
    struct dirent* ep;
    std::string buf;
    dp = opendir(folderName);
    while ((ep = readdir(dp)) != NULL) {
      if (strcmp(ep->d_name, ".") && strcmp(ep->d_name, "..")) {
        buf = std::string(folderName) + "/" + std::string(ep->d_name);
        if (isDir(buf.c_str())) {
          rmDir(buf.c_str());
        } else {
          remove(buf.c_str());
        }
      }
    }
    closedir(dp);
    rmdir(folderName);
  }
}

void mkDir(const char* filename) {
  if (mkdir(filename, 0755)) {
    PANIC_ENFORCE(errno == EEXIST, "%s mkdir failed!", filename);
  }
}

void mkDirRecursively(const char* dir) {
  struct stat sb;

  if (*dir == 0) return;  // empty string
  if (!stat(dir, &sb)) return;

  mkDirRecursively(path::dirname(dir).c_str());

  mkDir(dir);
}

void loadFileList(const std::string& fileListFileName,
                  std::vector<std::string>& fileList) {
  std::ifstream is(fileListFileName);
  PANIC_ENFORCE(is, "Fail to open %s", fileListFileName.c_str());
  std::string line;
  while (is) {
    if (!getline(is, line)) break;
    fileList.push_back(line);
  }
}

double getMemoryUsage() {
#if defined(__ANDROID__)
  return 0.0;
#else
  FILE* fp = fopen("/proc/meminfo", "r");
  PANIC_ENFORCE(fp, "failed to fopen /proc/meminfo");
  size_t bufsize = 256 * sizeof(char);
  char* buf = new (std::nothrow) char[bufsize];
  PANIC_ENFORCE(buf, "new buf fail");
  int totalMem = -1;
  int freeMem = -1;
  int bufMem = -1;
  int cacheMem = -1;
  while (getline(&buf, &bufsize, fp) >= 0) {
    if (0 == strncmp(buf, "MemTotal", 8)) {
      if (1 != sscanf(buf, "%*s%d", &totalMem)) {
        PANIC("failed to get MemTotal from string: [%s]", buf);
      }
    } else if (0 == strncmp(buf, "MemFree", 7)) {
      if (1 != sscanf(buf, "%*s%d", &freeMem)) {
        PANIC("failed to get MemFree from string: [%s]", buf);
      }
    } else if (0 == strncmp(buf, "Buffers", 7)) {
      if (1 != sscanf(buf, "%*s%d", &bufMem)) {
        PANIC("failed to get Buffers from string: [%s]", buf);
      }
    } else if (0 == strncmp(buf, "Cached", 6)) {
      if (1 != sscanf(buf, "%*s%d", &cacheMem)) {
        PANIC("failed to get Cached from string: [%s]", buf);
      }
    }
    if (totalMem != -1 && freeMem != -1 && bufMem != -1 && cacheMem != -1) {
      break;
    }
  }
  PANIC_ENFORCE(totalMem != -1 && freeMem != -1 && bufMem != -1 && cacheMem != -1,
                "failed to get all information");
  fclose(fp);
  delete[] buf;
  double usedMem = 1.0 - 1.0 * (freeMem + bufMem + cacheMem) / totalMem;
  return usedMem;
#endif
}

SyncThreadPool* getGlobalSyncThreadPool(int trainer_count) {
  static std::unique_ptr<SyncThreadPool> syncThreadPool;
  if (syncThreadPool &&
      syncThreadPool->getNumThreads() != (size_t)trainer_count) {
    PRINTF_WARN("trainer_count changed in training process!\n");
    syncThreadPool.reset(nullptr);
  }
  if (!syncThreadPool) {
    syncThreadPool.reset(new SyncThreadPool(trainer_count));
  }
  return syncThreadPool.get();
}

size_t calculateServiceNum(const std::string& pservers, int ports_num) {
  std::vector<std::string> hosts;
  str::split(pservers, ',', &hosts);
  return hosts.size() * ports_num;
}

void memcpyWithCheck(void* dest,
                     const void* src,
                     size_t num,
                     const void* srcEnd) {
  int minus = (char*)srcEnd - (char*)src - num;
  PANIC_ENFORCE_LE(0, minus);
  memcpy(dest, src, num);
}

}  // namespace mypaddle
}  // namespace bubblefs