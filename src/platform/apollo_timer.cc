/******************************************************************************
 * Copyright 2017 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

// apollo/modules/perception/lib/base/timer.cc

#include "platform/apollo_timer.h"

namespace bubblefs {
namespace myapollo {
namespace common {
namespace time {

using std::string;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

void Timer::Start() {
  start_time_ = std::chrono::system_clock::now();
}

uint64_t Timer::End(const string &msg) {
  end_time_ = std::chrono::system_clock::now();
  uint64_t elapsed_time =
      duration_cast<milliseconds>(end_time_ - start_time_).count();

  // start new timer.
  start_time_ = end_time_;
  return elapsed_time;
}

}  // namespace time
}  // namespace common
}  // namespace myapollo
}  // namespace bubblefs