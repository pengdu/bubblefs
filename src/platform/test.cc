/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

// tensorflow/tensorflow/core/platform/test.cc
// tensorflow/tensorflow/core/platform/posix/test.cc

#include "platform/test.h"
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "platform/logging.h"
#include "platform/port.h"
#include "platform/subprocess.h"
#include "platform/types.h"

namespace bubblefs {
namespace testing {

#if defined(PLATFORM_GOOGLE)
string TmpDir() { return FLAGS_test_tmpdir; }
int RandomSeed() { return FLAGS_test_random_seed; }
#else
string TmpDir() {
  // 'bazel test' sets TEST_TMPDIR
  const char* env = getenv("TEST_TMPDIR");
  if (env && env[0] != '\0') {
    return env;
  }
  env = getenv("TMPDIR");
  if (env && env[0] != '\0') {
    return env;
  }
  return "/tmp";
}
string SrcDir() {
  // Bazel makes data dependencies available via a relative path.
  return "";
}
int RandomSeed() {
  const char* env = getenv("TEST_RANDOM_SEED");
  int result;
  if (env && sscanf(env, "%d", &result) == 1) {
    return result;
  }
  return 301;
}
#endif  
  
std::unique_ptr<SubProcess> CreateSubProcess(const std::vector<string>& argv) {
  std::unique_ptr<SubProcess> proc(new SubProcess());
  proc->SetProgram(argv[0], argv);
  proc->SetChannelAction(CHAN_STDERR, ACTION_DUPPARENT);
  proc->SetChannelAction(CHAN_STDOUT, ACTION_DUPPARENT);
  return proc;
}

int PickUnusedPortOrDie() { return internal::PickUnusedPortOrDie(); }

}  // namespace testing
}  // namespace bubblefs