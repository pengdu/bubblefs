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
// Copyright (c) 2014 The IndexFS Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

// Paddle/paddle/utils/Logging.cpp
// indexfs/common/logging.cc

#include "platform/logging.h"

namespace bubblefs {
  
void InitializeLogging(int argc, char** argv) {
  (void)(argc);
#if TF_USE_GLOG
  if (!getenv("GLOG_logtostderr")) {
    google::LogToStderr();
  }
  google::InstallFailureSignalHandler();
  google::InitGoogleLogging(argv[0]);
#endif
}


void SetupLog(const std::string& name) {
    // log info/warning/error/fatal to tera.log
    // log warning/error/fatal to tera.wf

    std::string program_name = "program_name";
    if (!name.empty()) {
        program_name = name;
    }

    std::string log_filename = FLAGS_log_dir + "/" + program_name + ".INFO.";
    std::string wf_filename = FLAGS_log_dir + "/" + program_name + ".WARNING.";
    google::SetLogDestination(google::INFO, log_filename.c_str());
    google::SetLogDestination(google::WARNING, wf_filename.c_str());
    google::SetLogDestination(google::ERROR, "");
    google::SetLogDestination(google::FATAL, "");

    google::SetLogSymlink(google::INFO, program_name.c_str());
    google::SetLogSymlink(google::WARNING, program_name.c_str());
    google::SetLogSymlink(google::ERROR, "");
    google::SetLogSymlink(google::FATAL, "");
}

namespace logging {

void SetMinLogLevel(int level) {
#if TF_USE_GLOG
  FLAGS_minloglevel = level; 
#endif
}

void InstallFailureFunction(void (*callback)()) {
#if TF_USE_GLOG
  google::InstallFailureFunction(callback);
#endif
}

void InstallFailureWriter(void (*callback)(const char*, int)) {
#if TF_USE_GLOG
  google::InstallFailureWriter(callback);
#endif
}

}  // namespace logging

void LoggerUtil::FlushLogFiles() { google::FlushLogFiles(google::INFO); }

namespace {
static const char* NULL_LOG_FILE = "/dev/null";
static inline
void InternalLogOpen(
    const char* log_fname) {
#ifndef NDEBUG
  FLAGS_minloglevel = 0;
  FLAGS_logbuflevel = -1;
#else
  FLAGS_minloglevel = 1;
  FLAGS_logbuflevel = 0;
#endif
  if (log_fname == NULL) {
    FLAGS_logtostderr = true;
    log_fname = NULL_LOG_FILE;
  }
  google::InitGoogleLogging(log_fname);
}
}

void LoggerUtil::Initialize(const char* log_fname) {
  InternalLogOpen(log_fname);
  google::InstallFailureSignalHandler();
}

void LoggerUtil::Shutdown() { google::ShutdownGoogleLogging(); }

}  // namespace bubblefs