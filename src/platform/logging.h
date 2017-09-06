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

// Paddle/paddle/utils/Logging.h
// tensorflow/tensorflow/core/platform/default/logging.h

#ifndef BUBBLEFS_PLATFORM_LOGGING_H_ 
#define BUBBLEFS_PLATFORM_LOGGING_H_

#include "platform/base.h"

#if TF_USE_GLOG
#include "glog/logging.h"
#else
// use a light version of glog
#include "platform/logging_default.h"
#endif // TF_USE_GLOG

namespace bubblefs {
  
void InitializeLogging(int argc, char** argv);

namespace logging {

void SetMinLogLevel(int level);
void InstallFailureFunction(void (*callback)());
void InstallFailureWriter(void (*callback)(const char*, int));

}  // namespace logging

namespace internal {
// Emit "message" as a log message to the log for the specified
// "severity" as if it came from a LOG call at "fname:line"
void LogString(const char* fname, int line, int severity,
               const string& message);
}  // namespace internal

}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_LOGGING_H_