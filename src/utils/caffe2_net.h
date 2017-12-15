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

// caffe2/caffe2/core/net.h

#ifndef BUBBLEFS_UTILS_CAFFE2_NET_H_
#define BUBBLEFS_UTILS_CAFFE2_NET_H_

#include <atomic>
#include <climits>
#include <cstddef>
#include <thread> // NOLINT
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "platform/base_error.h"
#include "platform/types.h"
#include "utils/caffe2_blob.h"
#include "utils/caffe2_observer.h"
#include "utils/caffe2_operator_schema.h"
#include "utils/caffe2_tensor.h"
#include "utils/caffe2_workspace.h"
#include "utils/caffe2_proto_caffe2.h"
#include "utils/caffe2_simple_queue.h"

namespace bubblefs {
namespace mycaffe2 {

class NetBase;
typedef ObserverBase<NetBase> NetObserver;
typedef std::function<std::unique_ptr<NetObserver>(NetBase*)>
    NetObserverCreator;

class OperatorBase;
class Workspace;

// Net is a thin struct that owns all the operators together with the operator
// contexts.
class NetBase : public Observable<NetBase> {
 public:
  NetBase(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  virtual ~NetBase() noexcept {}

  virtual bool SupportsAsync() = 0;
  inline const std::vector<const Event*>& events() const {
    return events_;
  }

  virtual void Wait() {
    // by default just wait till all events are finished
    for (const auto& event : events_) {
      event->Finish();
    }
  }

  virtual bool Run() {
    if (!RunAsync()) {
      PRINTF_ERROR("Failed to execute async run\n");
      return false;
    }
    Wait();
    for (const Event* event : events_) {
      if (event->Query() != EventStatus::EVENT_SUCCESS) {
        //CAFFE_THROW(event->ErrorMessage());
        PANIC("%s", event->ErrorMessage().c_str());
      }
    }
    return true;
  }

  virtual bool RunAsync();

  /**
   * Benchmarks a network.
   *
   * This function returns a vector of float recording the number of milli-
   * seconds spent during the benchmark. The 0-th item is the time spent per
   * each network run, and if a net instantiation supports run_individual,
   * the remainder of the vector returns the number of milliseconds spent per
   * opeartor.
   */
  virtual std::vector<float> TEST_Benchmark(
      const int /*warmup_runs*/,
      const int /*main_runs*/,
      const bool /*run_individual*/) {
    PRINTF_ERROR("Benchmark not implemented for this net type.\n");
    return std::vector<float>();
  }

  inline const std::vector<string>& external_output() const {
    return external_output_;
  }

  inline const std::vector<string>& external_input() const {
    return external_input_;
  }

  /* Used to attach Observers to operators of a Net
   *
   * Returns pointers to objects owned with unique_ptrs.
   * Use with caution.
   */
  virtual std::vector<OperatorBase*> GetOperators() const = 0;

  const string& Name() const {
    return name_;
  }

  inline const NetDef& debug_def() const {
    PANIC_ENFORCE(has_debug_def(), "net_def was null!");
    return *net_def_;
  }

  inline bool has_debug_def() const {
    return net_def_ != nullptr;
  }

 protected:
  virtual bool DoRunAsync() {
    //CAFFE_THROW("Not implemented");
  };

  std::vector<string> external_input_;
  std::vector<string> external_output_;
  string name_;
  std::vector<const Event*> events_;
  std::shared_ptr<const NetDef> net_def_;
  DISALLOW_COPY_AND_ASSIGN(NetBase);
};

/**
 * @brief Creates a network, accessing / creating blobs in the given workspace.
 *
 * Note that this is different from Workspace::CreateNet. The latter adds the
 * created net object to the workspace's net map, while this function returns
 * a standalone net object.
 */
unique_ptr<NetBase> CreateNet(const NetDef& net_def, Workspace* ws);
unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws);

void SetGlobalNetObserverCreator(NetObserverCreator creator);

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_NET_H_