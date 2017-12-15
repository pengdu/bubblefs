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
// caffe2/caffe2/core/net.cc

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
#include "platform/caffe2_timer.h"
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

NetBase::NetBase(
    const std::shared_ptr<const NetDef>& def,
    Workspace* /* unused */)
    : external_input_(
          def->external_input().begin(),
          def->external_input().end()),
      external_output_(
          def->external_output().begin(),
          def->external_output().end()),
      name_(def->name()),
      net_def_(def) {
  // Check that node_name is empty for all ops
  for (const OperatorDef& op : def->op()) {
    if (op.has_device_option()) {
      PANIC_ENFORCE(
          !op.device_option().has_node_name(),
          "node_name must be empty for all operators at execution time.");
    }
  }

  // Go through the operators and make sure that blobs are correctly made.
  std::set<string> known_blobs(
      external_input_.begin(), external_input_.end());
  std::set<string> remaining_output(
      external_output_.begin(), external_output_.end());
  for (const auto& blob : known_blobs) {
    remaining_output.erase(blob);
  }
  for (const OperatorDef& op : def->op()) {
    for (const string& in : op.input()) {
      if (!known_blobs.count(in)) {
        if (external_input_.size()) {
          //CAFFE_THROW(
          //   "op ",
          //   op.type(),
          //   ": Source for input ",
          //   in,
          //   " is unknown for net ",
          //  def->name(),
          //  ", operator ",
          //ProtoDebugString(op));
        } else {
          // If we are not declaring input and output, we will simply VLOG it
          // for debugging purposes.
          //VLOG(1) << "op " << op.type() << ": input " << in << " is unknown.";
        }
      }
    }
    for (const string& out : op.output()) {
      known_blobs.insert(out);
      remaining_output.erase(out);
    }
  }
  // Finally, check if all declared outputs are being created.
  //CAFFE_ENFORCE(
  //    remaining_output.size() == 0,
  //    "Some of the blobs are declared as output but never produced by the "
  //    "net ",
  //    def->name(),
  //    ", the first one is ",
  //    *remaining_output.begin());
}

bool NetBase::RunAsync() {
  for (auto& op : GetOperators()) {
    op->ResetEvent();
  }
  return DoRunAsync();
}

static NetObserverCreator GlobalNetObserverCreator = [](NetBase* net) {
  // A no-op ObserverBase<NetBase> observer
  return std::unique_ptr<NetObserver>(new NetObserver(net));
};

void SetGlobalNetObserverCreator(NetObserverCreator creator) {
  GlobalNetObserverCreator = creator;
  //VLOG(1) << "Have set custom GlobalNetObserverCreator";
}

unique_ptr<NetBase> CreateNet(const NetDef& net_def, Workspace* ws) {
  std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
  return CreateNet(tmp_net_def, ws);
}

unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws) {
  // In default, we will return a simple network that just runs all operators
  // sequentially.
  unique_ptr<NetBase> net;
  if (!net_def->has_type()) {
    net = std::unique_ptr<NetBase>(new SimpleNet(net_def, ws));
  } else {
    //net = NetRegistry()->Create(net_def->type(), net_def, ws);
  }
  //VLOG(1) << "Adding a global observer to a net";
  if (net) {
    net->AttachObserver(GlobalNetObserverCreator(net.get()));
  }
  return net;
}

} // namespace mycaffe2
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_CAFFE2_NET_H_