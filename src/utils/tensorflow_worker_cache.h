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

// tensorflow/tensorflow/core/distributed_runtime/worker_cache.h

#ifndef BUBBLEFS_UTILS_TENSORFLOW_WORKER_CACHE_H_
#define BUBBLEFS_UTILS_TENSORFLOW_WORKER_CACHE_H_

#include <string>
#include <vector>
#include "utils/tensorflow_worker_interface.h"
#include "utils/status.h"

namespace bubblefs {
namespace mytensorflow {
  
typedef std::function<void(const Status&)> StatusCallback;

class ChannelCache;
class StepStats;

class WorkerCacheInterface {
 public:
  virtual ~WorkerCacheInterface() {}

  // Updates *workers with strings naming the remote worker tasks to
  // which open channels have been established.
  virtual void ListWorkers(std::vector<string>* workers) const = 0;

  // If "target" names a remote task for which an RPC channel exists
  // or can be constructed, returns a pointer to a WorkerInterface object
  // wrapping that channel. The returned value must be destroyed by
  // calling `this->ReleaseWorker(target, ret)`
  // TODO(mrry): rename this to GetOrCreateWorker() or something that
  // makes it more obvious that this method returns a potentially
  // shared object.
  virtual WorkerInterface* CreateWorker(const string& target) = 0;

  // Release a worker previously returned by this->CreateWorker(target).
  //
  // TODO(jeff,sanjay): Consider moving target into WorkerInterface.
  // TODO(jeff,sanjay): Unify all worker-cache impls and factor out a
  //                    per-rpc-subsystem WorkerInterface creator.
  virtual void ReleaseWorker(const string& target, WorkerInterface* worker) {
    // Subclasses may override to reuse worker objects.
    delete worker;
  }

  // Set *locality with the DeviceLocality of the specified remote device
  // within its local environment.  Returns true if *locality
  // was set, using only locally cached data.  Returns false
  // if status data for that device was not available.  Never blocks.
  virtual bool GetDeviceLocalityNonBlocking(const string& device,
                                            DeviceLocality* locality) = 0;

  // Set *locality with the DeviceLocality of the specified remote device
  // within its local environment.  Callback gets Status::OK if *locality
  // was set.
  virtual void GetDeviceLocalityAsync(const string& device,
                                      DeviceLocality* locality,
                                      StatusCallback done) = 0;

  // Start/stop logging activity.
  virtual void SetLogging(bool active) {}

  // Discard any saved log data.
  virtual void ClearLogs() {}

  // Return logs for the identified step in *ss.  Any returned data will no
  // longer be stored.
  virtual bool RetrieveLogs(int64 step_id, StepStats* ss) { return false; }
};

}  // namespace mytensorflow
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_TENSORFLOW_WORKER_CACHE_H_