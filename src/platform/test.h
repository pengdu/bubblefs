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

// tensorflow/tensorflow/core/platform/test.h

#ifndef BUBBLEFS_PLATFORM_TEST_H_
#define BUBBLEFS_PLATFORM_TEST_H_

#include <memory>
#include <vector>
#include "platform/macros.h"
#include "platform/platform.h"
#include "platform/subprocess.h"
#include "platform/types.h"

// As of September 2016, we continue to attempt to avoid the use of gmock aka
// googlemock included in the test framework
// (https://github.com/google/googletest) to discourage over-eager use of mocks
// that lead to cumbersome class hierarchies and tests that might end up not
// testing real code in important ways.
#include "gtest/gtest.h"

namespace bubblefs {
namespace testing {

// Return a temporary directory suitable for temporary testing files.
string TmpDir();

// Return a random number generator seed to use in randomized tests.
// Returns the same value for the lifetime of the process.
int RandomSeed();

// Returns an object that represents a child process that will be
// launched with the given command-line arguments `argv`. The process
// must be explicitly started by calling the Start() method on the
// returned object.
std::unique_ptr<SubProcess> CreateSubProcess(const std::vector<string>& argv);

// Returns an unused port number, for use in multi-process testing.
// NOTE: This function is not thread-safe.
int PickUnusedPortOrDie();

}  // namespace testing
}  // namespace bubblefs

#endif  // BUBBLEFS_PLATFORM_TEST_H_