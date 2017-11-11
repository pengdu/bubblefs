/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// caffe2/caffe2/core/flags.h

/**
 * @file flags.h
 * @brief Commandline flags support for Caffe2.
 *
 * This is a portable commandline flags tool for caffe2, so we can optionally
 * choose to use gflags or a lightweighted custom implementation if gflags is
 * not possible on a certain platform. If you have gflags installed, set the
 * macro CAFFE2_USE_GFLAGS will seamlessly route everything to gflags.
 *
 * To define a flag foo of type bool default to true, do the following in the
 * *global* namespace:
 *     CAFFE2_DEFINE_bool(foo, true, "An example.");
 *
 * To use it in another .cc file, you can use CAFFE2_DECLARE_* as follows:
 *     CAFFE2_DECLARE_bool(foo);
 *
 * In both cases, you can then access the flag via caffe2::FLAGS_foo.
 */

#ifndef BUBBLEFS_PLATFORM_CAFFE2_FLAGS_H_
#define BUBBLEFS_PLATFORM_CAFFE2_FLAGS_H_

#include "gflags/gflags.h"

namespace bubblefs {
namespace mycaffe2 {
  
/**
 * Sets the usage message when a commandline tool is called with "--help".
 */
void SetUsageMessage(const std::string& str);

/**
 * Returns the usage message for the commandline tool set by SetUsageMessage.
 */
const char* UsageMessage();

/**
 * Parses the commandline flags.
 *
 * This command parses all the commandline arguments passed in via pargc
 * and argv. Once it is finished, partc and argv will contain the remaining
 * commandline args that caffe2 does not deal with. Note that following
 * convention, argv[0] contains the binary name and is not parsed.
 */
bool ParseCaffeCommandLineFlags(int* pargc, char*** pargv);
/**
 * Checks if the commandline flags has already been passed.
 */
bool CommandLineFlagsHasBeenParsed();

}  // namespace mycaffe2
}  // namespace bubblefs

// gflags before 2.0 uses namespace google and after 2.1 uses namespace gflags.
// Using GFLAGS_GFLAGS_H_ to capture this change.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

#define MYCAFFE2_GFLAGS_DEF_WRAPPER(type, name, default_value, help_str)         \
  DEFINE_##type(name, default_value, help_str);                                \
  namespace bubblefs {                                                         \
  namespace mycaffe2 {                                                           \
    using ::FLAGS_##name;                                                      \
  }                                                                            \
  } // namespace bubblefs.mycaffe2

#define MYCAFFE2_DEFINE_int(name, default_value, help_str)                       \
  MYCAFFE2_GFLAGS_DEF_WRAPPER(int32, name, default_value, help_str)
#define MYCAFFE2_DEFINE_int64(name, default_value, help_str)                     \
  MYCAFFE2_GFLAGS_DEF_WRAPPER(int64, name, default_value, help_str)              
#define MYCAFFE2_DEFINE_double(name, default_value, help_str)                    \
  MYCAFFE2_GFLAGS_DEF_WRAPPER(double, name, default_value, help_str)
#define MYCAFFE2_DEFINE_bool(name, default_value, help_str)                      \
  MYCAFFE2_GFLAGS_DEF_WRAPPER(bool, name, default_value, help_str)
#define MYCAFFE2_DEFINE_string(name, default_value, help_str) \
  MYCAFFE2_GFLAGS_DEF_WRAPPER(string, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define MYCAFFE2_GFLAGS_DECLARE_WRAPPER(type, name)                              \
  DECLARE_##type(name);                                                        \
  namespace bubblefs {                                                         \
  namespace mycaffe2 {                                                           \
    using ::FLAGS_##name;                                                      \
  }                                                                            \
  } // namespace bubblefs.mycaffe2

#define MYCAFFE2_DECLARE_int(name) MYCAFFE2_GFLAGS_DECLARE_WRAPPER(int32, name)
#define MYCAFFE2_DECLARE_int64(name) MYCAFFE2_GFLAGS_DECLARE_WRAPPER(int64, name)
#define MYCAFFE2_DECLARE_double(name) MYCAFFE2_GFLAGS_DECLARE_WRAPPER(double, name)
#define MYCAFFE2_DECLARE_bool(name) MYCAFFE2_GFLAGS_DECLARE_WRAPPER(bool, name)
#define MYCAFFE2_DECLARE_string(name) MYCAFFE2_GFLAGS_DECLARE_WRAPPER(string, name)

#endif  // BUBBLEFS_PLATFORM_CAFFE2_FLAGS_H_