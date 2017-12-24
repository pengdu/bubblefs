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

// tensorflow/tensorflow/core/distributed_runtime/session_mgr.h

#ifndef BUBBLEFS_UTILS_TENSORFLOW_SESSION_MGR_H_
#define BUBBLEFS_UTILS_TENSORFLOW_SESSION_MGR_H_

#include <functional>
#include "platform/mutex.h"
#include "tensorflow_worker_session.h"
#include "utils/status.h"
#include "utils/tensorflow_protobuf_proto.h"

namespace bubblefs {
namespace mytensorflow {

class WorkerCacheInterface;
struct WorkerEnv;

// SessionMgr keeps track of information related to a given session.
//
// SessionMgr runs on the workers.
//
// SessionMgr is threadsafe.
class SessionMgr {
 public:
  typedef std::function<Status(const ServerDef&, WorkerCacheInterface**)>
      WorkerCacheFactory;

  explicit SessionMgr(
      WorkerEnv* worker_env, const string& default_worker_name,
      std::unique_ptr<WorkerCacheInterface> default_worker_cache,
      WorkerCacheFactory worker_cache_factory);
  ~SessionMgr() {}

  // Allocates state for a new session.
  Status CreateSession(const string& session, const ServerDef& server_def,
                       bool isolate_session_state);

  // Locates the worker session for a given session handle
  WorkerSession* WorkerSessionForSession(const string& session);
  WorkerSession* LegacySession();

  Status DeleteSession(const string& session);

  static string WorkerNameFromServerDef(const ServerDef& server_def);

 private:
  const WorkerEnv* const worker_env_;  // Not owned.

  // A note about destruction:
  // We must delete graph_mgr before device_mgr, due to shared
  // ownership of OpKernels in the executors. (The graph_mgr will
  // free all stateless OpKernels, and pass over borrowed stateful
  // OpKernels, which are also held in their respective devices'
  // OpSegments.)
  //
  // legacy_session_ owns the worker_env_.device_mgr, and so we must ensure
  // that sessions_'s WorkerSessions are deleted (which do not own the
  // underlying devices, but instead own RenamedDevices) before
  // legacy_session_ is deleted. Further, we must ensure that WorkerSession's
  // device_mgr is deleted after WorkerSession's graph_mgr.

  std::unique_ptr<WorkerCacheInterface> default_worker_cache_;
  WorkerSession legacy_session_;

  const WorkerCacheFactory worker_cache_factory_;

  WorkerSession* WorkerSessionForSessionUnlocked(const string& session)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutex mu_;
  // A map from session identifier to internal session structure.
  std::map<string, std::unique_ptr<WorkerSession>> sessions_ GUARDED_BY(mu_);
};

}  // namespace mytensorflow
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_TENSORFLOW_SESSION_MGR_H_