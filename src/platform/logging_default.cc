
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

// tensorflow/tensorflow/core/platform/default/logging.cc

#include "platform/logging_default.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include "platform/env_time.h"
#include "platform/macros.h"
#include "utils/stringpiece.h"

namespace bubblefs {
namespace internal {

LogMessage::LogMessage(const char* fname, int line, int severity)
    : fname_(fname), line_(line), severity_(severity) {}

#if defined(PLATFORM_POSIX_ANDROID)
void LogMessage::GenerateLogMessage() {
  int android_log_level;
  switch (severity_) {
    case INFO:
      android_log_level = ANDROID_LOG_INFO;
      break;
    case WARNING:
      android_log_level = ANDROID_LOG_WARN;
      break;
    case ERROR:
      android_log_level = ANDROID_LOG_ERROR;
      break;
    case FATAL:
      android_log_level = ANDROID_LOG_FATAL;
      break;
    default:
      if (severity_ < INFO) {
        android_log_level = ANDROID_LOG_VERBOSE;
      } else {
        android_log_level = ANDROID_LOG_ERROR;
      }
      break;
  }

  std::stringstream ss;
  const char* const partial_name = strrchr(fname_, '/');
  ss << (partial_name != nullptr ? partial_name + 1 : fname_) << ":" << line_
     << " " << str();
  __android_log_write(android_log_level, "native", ss.str().c_str());

  // Also log to stderr (for standalone Android apps).
  std::cerr << "native : " << ss.str() << std::endl;

  // Android logging at level FATAL does not terminate execution, so abort()
  // is still required to stop the program.
  if (severity_ == FATAL) {
    abort();
  }
}

#else

void LogMessage::GenerateLogMessage() {
  static EnvTime* env_time = EnvTime::Default();
  uint64 now_micros = env_time->NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32 micros_remainder = static_cast<int32>(now_micros % 1000000);
  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];
  strftime(time_buffer, time_buffer_size, "%Y-%m-%d %H:%M:%S",
           localtime(&now_seconds));

  // TODO(jeff,sanjay): Replace this with something that logs through the env.
  fprintf(stderr, "%s.%06d: %c %s:%d] %s\n", time_buffer, micros_remainder,
          "IWEF"[severity_], fname_, line_, str().c_str());
}
#endif


namespace {

// Parse log level (int64) from environment variable (char*)
int64 LogLevelStrToInt(const char* tf_env_var_val) {
  if (tf_env_var_val == nullptr) {
    return 0;
  }

  // Ideally we would use env_var / safe_strto64, but it is
  // hard to use here without pulling in a lot of dependencies,
  // so we use std:istringstream instead
  string min_log_level(tf_env_var_val);
  std::istringstream ss(min_log_level);
  int64 level;
  if (!(ss >> level)) {
    // Invalid vlog level setting, set level to default (0)
    level = 0;
  }

  return level;
}

int64 MinLogLevelFromEnv() {
  const char* tf_env_var_val = getenv("TF_CPP_MIN_LOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
}

int64 MinVLogLevelFromEnv() {
  const char* tf_env_var_val = getenv("TF_CPP_MIN_VLOG_LEVEL");
  return LogLevelStrToInt(tf_env_var_val);
}

using VmoduleMap = std::unordered_map<StringPiece, int, StringPiece::Hasher>;

// Returns a mapping from module name to VLOG level, derived from the
// TF_CPP_VMOUDLE environment variable; ownership is transferred to the caller.
VmoduleMap* VmoduleRecordsFromEnv() {
  // The value of the env var is supposed to be of the form:
  //    "foo=1,bar=2,baz=3"
  const char* tf_env_var_val = getenv("TF_CPP_VMODULE");
  auto* result = new VmoduleMap();
  if (tf_env_var_val == nullptr) return result;
  while (true) {
    const char* eq = strchr(tf_env_var_val, '=');
    if (eq == nullptr) break;
    const char* after_eq = eq + 1;

    // Comma either points at the next comma delimiter, or at a null terminator.
    // We check that the integer we parse ends at this delimiter.
    const char* comma = strchr(after_eq, ',');
    const char* new_tf_env_var_val;
    if (comma == nullptr) {
      comma = strchr(after_eq, '\0');
      new_tf_env_var_val = comma;
    } else {
      new_tf_env_var_val = comma + 1;
    }

    char* endptr = nullptr;
    int level = strtol(after_eq, &endptr, 10);
    if (endptr != comma) {
      fprintf(stderr,
              "warning: could not parse integer in vmodule specification in "
              "\"%s\".\n",
              after_eq);
      break;
    }
    StringPiece module(tf_env_var_val, eq - tf_env_var_val);
    tf_env_var_val = new_tf_env_var_val;
    (*result)[module] = level;
  }
  return result;
}

}  // namespace

LogMessage::~LogMessage() {
  // Read the min log level once during the first call to logging.
  static int64 min_log_level = MinLogLevelFromEnv();
  if (PREDICT_TRUE(severity_ >= min_log_level)) GenerateLogMessage();
}

int64 LogMessage::MinVLogLevel() {
  static int64 min_vlog_level = MinVLogLevelFromEnv();
  return min_vlog_level;
}

bool LogMessage::VmoduleActivated(const char* fname, int lvl) {
  static VmoduleMap* vmodule_records = VmoduleRecordsFromEnv();
  const char* last_slash = strrchr(fname, '/');
  const char* module_start = last_slash == nullptr ? fname : last_slash + 1;
  const char* dot_after = strchr(module_start, '.');
  const char* module_limit =
      dot_after == nullptr ? strchr(fname, '\0') : dot_after;
  StringPiece module(module_start, module_limit - module_start);
  auto it = vmodule_records->find(module);
  if (it == vmodule_records->end()) return false;
  return it->second >= lvl;
}

LogMessageFatal::LogMessageFatal(const char* file, int line)
    : LogMessage(file, line, FATAL) {}
LogMessageFatal::~LogMessageFatal() {
  // abort() ensures we don't return (we promised we would not via
  // ATTRIBUTE_NORETURN).
  GenerateLogMessage();
  abort();
}

void LogString(const char* fname, int line, int severity,
               const string& message) {
  LogMessage(fname, line, severity) << message;
}

template <>
void MakeCheckOpValueString(std::ostream* os, const char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "char value " << static_cast<short>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const signed char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "signed char value " << static_cast<short>(v);
  }
}

template <>
void MakeCheckOpValueString(std::ostream* os, const unsigned char& v) {
  if (v >= 32 && v <= 126) {
    (*os) << "'" << v << "'";
  } else {
    (*os) << "unsigned char value " << static_cast<unsigned short>(v);
  }
}

#if LANG_CXX11
template <>
void MakeCheckOpValueString(std::ostream* os, const std::nullptr_t& p) {
  (*os) << "nullptr";
}
#endif

CheckOpMessageBuilder::CheckOpMessageBuilder(const char* exprtext)
    : stream_(new std::ostringstream) {
  *stream_ << "Check failed: " << exprtext << " (";
}

CheckOpMessageBuilder::~CheckOpMessageBuilder() { delete stream_; }

std::ostream* CheckOpMessageBuilder::ForVar2() {
  *stream_ << " vs. ";
  return stream_;
}

string* CheckOpMessageBuilder::NewString() {
  *stream_ << ")";
  return new string(stream_->str());
}

}  // namespace internal
}  // namespace bubblefs