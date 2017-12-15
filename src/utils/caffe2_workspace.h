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

// caffe2/caffe2/core/workspace.h

#ifndef BUBBLEFS_UTILS_CAFFE2_WORKSPACE_H_
#define BUBBLEFS_UTILS_CAFFE2_WORKSPACE_H_

#include "platform/types.h"
#include "utils/caffe2_observer.h"

#include <climits>
#include <cstddef>
#include <atomic>
#include <mutex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "utils/caffe2_blob.h"
#include "utils/caffe2_proto_caffe2.h"

namespace bubblefs {
namespace mycaffe2 {

class NetBase;

bool FLAGS_caffe2_print_blob_sizes_at_exit;

/**
 * Workspace is a class that holds all the related objects created during
 * runtime: (1) all blobs, and (2) all instantiated networks. It is the owner of
 * all these objects and deals with the scaffolding logistics.
 */
class Workspace {
 public:
  typedef std::function<bool(int)> ShouldContinue;
  typedef CaffeMap<string, unique_ptr<Blob> > BlobMap;
  typedef CaffeMap<string, unique_ptr<NetBase> > NetMap;
  /**
   * Initializes an empty workspace.
   */
  Workspace() : root_folder_("."), shared_(nullptr) {}

  /**
   * Initializes an empty workspace with the given root folder.
   *
   * For any operators that are going to interface with the file system, such
   * as load operators, they will write things under this root folder given
   * by the workspace.
   */
  explicit Workspace(const string& root_folder)
      : root_folder_(root_folder), shared_(nullptr) {}

  /**
   * Initializes a workspace with a shared workspace.
   *
   * When we access a Blob, we will first try to access the blob that exists
   * in the local workspace, and if not, access the blob that exists in the
   * shared workspace. The caller keeps the ownership of the shared workspace
   * and is responsible for making sure that its lifetime is longer than the
   * created workspace.
   */
  explicit Workspace(const Workspace* shared)
      : root_folder_("."), shared_(shared) {}

  /**
   * Initializes workspace with parent workspace, blob name remapping
   * (new name -> parent blob name), no other blobs are inherited from
   * parent workspace
   */
  Workspace(
      const Workspace* shared,
      const std::unordered_map<string, string>& forwarded_blobs)
      : root_folder_("."), shared_(nullptr) {
    PANIC_ENFORCE(shared, "Parent workspace must be specified");
    for (const auto& forwarded : forwarded_blobs) {
      PANIC_ENFORCE(
          shared->HasBlob(forwarded.second), "Invalid parent workspace blob");
      forwarded_blobs_[forwarded.first] =
          std::make_pair(shared, forwarded.second);
    }
  }

  /**
   * Initializes a workspace with a root folder and a shared workspace.
   */
  Workspace(const string& root_folder, Workspace* shared)
      : root_folder_(root_folder), shared_(shared) {}

  ~Workspace() {
    if (FLAGS_caffe2_print_blob_sizes_at_exit) {
      PrintBlobSizes();
    }
  }

  /**
   * Add blob mappings from another workspace
   */
  void AddBlobMapping(
      const Workspace* parent,
      const std::unordered_map<string, string>& forwarded_blobs);

  /**
   * Return list of blobs owned by this Workspace, not including blobs
   * shared from parent workspace.
   */
  std::vector<string> LocalBlobs() const;

  /**
   * Return a list of blob names. This may be a bit slow since it will involve
   * creation of multiple temp variables. For best performance, simply use
   * HasBlob() and GetBlob().
   */
  std::vector<string> Blobs() const;

  /**
   * Return the root folder of the workspace.
   */
  const string& RootFolder() { return root_folder_; }
  /**
   * Checks if a blob with the given name is present in the current workspace.
   */
  inline bool HasBlob(const string& name) const {
    // First, check the local workspace,
    // Then, check the forwarding map, then the parent workspace
    if (blob_map_.count(name)) {
      return true;
    } else if (forwarded_blobs_.count(name)) {
      const auto parent_ws = forwarded_blobs_.at(name).first;
      const auto& parent_name = forwarded_blobs_.at(name).second;
      return parent_ws->HasBlob(parent_name);
    } else if (shared_) {
      return shared_->HasBlob(name);
    }
    return false;
  }

  void PrintBlobSizes();

  /**
   * Creates a blob of the given name. The pointer to the blob is returned, but
   * the workspace keeps ownership of the pointer. If a blob of the given name
   * already exists, the creation is skipped and the existing blob is returned.
   */
  Blob* CreateBlob(const string& name);
  /**
   * Similar to CreateBlob(), but it creates a blob in the local workspace even
   * if another blob with the same name already exists in the parent workspace
   * -- in such case the new blob hides the blob in parent workspace. If a blob
   * of the given name already exists in the local workspace, the creation is
   * skipped and the existing blob is returned.
   */
  Blob* CreateLocalBlob(const string& name);
  /**
   * Remove the blob of the given name. Return true if removed and false if
   * not exist.
   * Will NOT remove from the shared workspace.
   */
  bool RemoveBlob(const string& name);
  /**
   * Gets the blob with the given name as a const pointer. If the blob does not
   * exist, a nullptr is returned.
   */
  const Blob* GetBlob(const string& name) const;
  /**
   * Gets the blob with the given name as a mutable pointer. If the blob does
   * not exist, a nullptr is returned.
   */
  Blob* GetBlob(const string& name);

  /**
   * Renames a local workspace blob. If blob is not found in the local blob list
   * or if the target name is already present in local or any parent blob list
   * the function will through.
   */
  Blob* RenameBlob(const string& old_name, const string& new_name);

  /**
   * Creates a network with the given NetDef, and returns the pointer to the
   * network. If there is anything wrong during the creation of the network, a
   * nullptr is returned. The Workspace keeps ownership of the pointer.
   *
   * If there is already a net created in the workspace with the given name,
   * CreateNet will overwrite it if overwrite=true is specified. Otherwise, an
   * exception is thrown.
   */
  NetBase* CreateNet(const NetDef& net_def, bool overwrite = false);
  NetBase* CreateNet(
      const std::shared_ptr<const NetDef>& net_def,
      bool overwrite = false);
  /**
   * Gets the pointer to a created net. The workspace keeps ownership of the
   * network.
   */
  NetBase* GetNet(const string& net_name);
  /**
   * Deletes the instantiated network with the given name.
   */
  void DeleteNet(const string& net_name);
  /**
   * Finds and runs the instantiated network with the given name. If the network
   * does not exist or there are errors running the network, the function
   * returns false.
   */
  bool RunNet(const string& net_name);

  /**
   * Returns a list of names of the currently instantiated networks.
   */
  std::vector<string> Nets() const {
    std::vector<string> names;
    for (auto& entry : net_map_) {
      names.push_back(entry.first);
    }
    return names;
  }

  /**
   * Runs a plan that has multiple nets and execution steps.
   */
  bool RunPlan(const PlanDef& plan_def,
               ShouldContinue should_continue);

#if CAFFE2_MOBILE
  /*
   * Returns a CPU threadpool instace for parallel execution of
   * work. The threadpool is created lazily; if no operators use it,
   * then no threadpool will be created.
   */
  ThreadPool* GetThreadPool();
#endif

  // RunOperatorOnce and RunNetOnce runs an operator or net once. The difference
  // between RunNet and RunNetOnce lies in the fact that RunNet allows you to
  // have a persistent net object, while RunNetOnce creates a net and discards
  // it on the fly - this may make things like database read and random number
  // generators repeat the same thing over multiple calls.
  bool RunOperatorOnce(const OperatorDef& op_def);
  bool RunNetOnce(const NetDef& net_def);

 public:
  std::atomic<int> last_failed_op_net_position;

 private:
  BlobMap blob_map_;
  NetMap net_map_;
  const string root_folder_;
  const Workspace* shared_;
  std::unordered_map<string, std::pair<const Workspace*, string>>
      forwarded_blobs_;
#if CAFFE2_MOBILE
  std::unique_ptr<ThreadPool> thread_pool_;
  std::mutex thread_pool_creation_mutex_;
#endif // CAFFE2_MOBILE

  DISALLOW_COPY_AND_ASSIGN(Workspace);
};

}  // namespace mycaffe2
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_CAFFE2_WORKSPACE_H_