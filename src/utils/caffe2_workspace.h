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
// caffe2/caffe2/core/workspace.cc

#ifndef BUBBLEFS_UTILS_CAFFE2_WORKSPACE_H_
#define BUBBLEFS_UTILS_CAFFE2_WORKSPACE_H_

#include "platform/types.h"
#include "utils/caffe2_observer.h"


#include <algorithm>
#include <atomic>
#include <climits>
#include <cstddef>
#include <ctime>
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

  DISALLOW_COPY_AND_ASSIGN(Workspace);
};

std::vector<string> Workspace::LocalBlobs() const {
  std::vector<string> names;
  names.reserve(blob_map_.size());
  for (auto& entry : blob_map_) {
    names.push_back(entry.first);
  }
  return names;
}

std::vector<string> Workspace::Blobs() const {
  std::vector<string> names;
  names.reserve(blob_map_.size());
  for (auto& entry : blob_map_) {
    names.push_back(entry.first);
  }
  for (const auto& forwarded : forwarded_blobs_) {
    const auto parent_ws = forwarded.second.first;
    const auto& parent_name = forwarded.second.second;
    if (parent_ws->HasBlob(parent_name)) {
      names.push_back(forwarded.first);
    }
  }
  if (shared_) {
    const auto& shared_blobs = shared_->Blobs();
    names.insert(names.end(), shared_blobs.begin(), shared_blobs.end());
  }
  return names;
}

Blob* Workspace::CreateBlob(const string& name) {
  if (HasBlob(name)) {
    //VLOG(1) << "Blob " << name << " already exists. Skipping.";
  } else if (forwarded_blobs_.count(name)) {
    // possible if parent workspace deletes forwarded blob
    //VLOG(1) << "Blob " << name << " is already forwarded from parent workspace "
    //        << "(blob " << forwarded_blobs_[name].second << "). Skipping.";
  } else {
    //VLOG(1) << "Creating blob " << name;
    blob_map_[name] = unique_ptr<Blob>(new Blob());
  }
  return GetBlob(name);
}

Blob* Workspace::CreateLocalBlob(const string& name) {
  if (blob_map_.count(name)) {
    //VLOG(1) << "Blob " << name << " already exists. Skipping.";
  } else {
    //VLOG(1) << "Creating blob " << name;
    blob_map_[name] = unique_ptr<Blob>(new Blob());
  }
  return GetBlob(name);
}

Blob* Workspace::RenameBlob(const string& old_name, const string& new_name) {
  // We allow renaming only local blobs for API clarity purpose
  auto it = blob_map_.find(old_name);
  PANIC_ENFORCE(
      it != blob_map_.end(),
      "Blob is not in the local blob list");

  // New blob can't be in any parent either, otherwise it will hide a parent
  // blob
  PANIC_ENFORCE(
      !HasBlob(new_name), "Blob is already in the workspace");

  // First delete the old record
  auto value = std::move(it->second);
  blob_map_.erase(it);

  auto* raw_ptr = value.get();
  blob_map_[new_name] = std::move(value);
  return raw_ptr;
}

bool Workspace::RemoveBlob(const string& name) {
  auto it = blob_map_.find(name);
  if (it != blob_map_.end()) {
    //VLOG(1) << "Removing blob " << name << " from this workspace.";
    blob_map_.erase(it);
    return true;
  }

  // won't go into shared_ here
  //VLOG(1) << "Blob " << name << " not exists. Skipping.";
  return false;
}

const Blob* Workspace::GetBlob(const string& name) const {
  if (blob_map_.count(name)) {
    return blob_map_.at(name).get();
  } else if (forwarded_blobs_.count(name)) {
    const auto parent_ws = forwarded_blobs_.at(name).first;
    const auto& parent_name = forwarded_blobs_.at(name).second;
    return parent_ws->GetBlob(parent_name);
  } else if (shared_ && shared_->HasBlob(name)) {
    return shared_->GetBlob(name);
  }
  //LOG(WARNING) << "Blob " << name << " not in the workspace.";
  // TODO(Yangqing): do we want to always print out the list of blobs here?
  // LOG(WARNING) << "Current blobs:";
  // for (const auto& entry : blob_map_) {
  //   LOG(WARNING) << entry.first;
  // }
  return nullptr;
}

void Workspace::AddBlobMapping(
    const Workspace* parent,
    const std::unordered_map<string, string>& forwarded_blobs) {
  PANIC_ENFORCE(parent, "Parent workspace must be specified");
  for (const auto& forwarded : forwarded_blobs) {
    PANIC_ENFORCE(
        parent->HasBlob(forwarded.second),
        "Invalid parent workspace blob");
    if (forwarded_blobs_.count(forwarded.first)) {
      const auto& ws_blob = forwarded_blobs_[forwarded.first];
      PANIC_ENFORCE_EQ(
          ws_blob.first, parent); // "Redefinition of blob " + forwarded.first);
      PANIC_ENFORCE_EQ(
          ws_blob.second,
          forwarded.second); // "Redefinition of blob " + forwarded.first);
    } else {
      PANIC_ENFORCE(
          !HasBlob(forwarded.first), "Redefinition of blob");
      // Lazy blob resolution - store the parent workspace and
      // blob name, blob value might change in the parent workspace
      forwarded_blobs_[forwarded.first] =
          std::make_pair(parent, forwarded.second);
    }
  }
}

Blob* Workspace::GetBlob(const string& name) {
  return const_cast<Blob*>(static_cast<const Workspace*>(this)->GetBlob(name));
}

NetBase* Workspace::GetNet(const string& name) {
  if (!net_map_.count(name)) {
    return nullptr;
  } else {
    return net_map_[name].get();
  }
}

void Workspace::DeleteNet(const string& name) {
  if (net_map_.count(name)) {
    net_map_.erase(name);
  }
}

bool Workspace::RunNet(const string& name) {
  if (!net_map_.count(name)) {
    //LOG(ERROR) << "Network " << name << " does not exist yet.";
    return false;
  }
  return net_map_[name]->Run();
}

bool Workspace::RunOperatorOnce(const OperatorDef& op_def) {
  std::unique_ptr<OperatorBase> op(CreateOperator(op_def, this));
  if (op.get() == nullptr) {
    //LOG(ERROR) << "Cannot create operator of type " << op_def.type();
    return false;
  }
  if (!op->Run()) {
    //LOG(ERROR) << "Error when running operator " << op_def.type();
    return false;
  }
  return true;
}
bool Workspace::RunNetOnce(const NetDef& net_def) {
  std::unique_ptr<NetBase> net(CreateNet(net_def, this));
  if (net == nullptr) {
    //CAFFE_THROW(
    //    "Could not create net: " + net_def.name() + " of type " +
    //    net_def.type());
  }
  if (!net->Run()) {
    //LOG(ERROR) << "Error when running network " << net_def.name();
    return false;
  }
  return true;
}

}  // namespace mycaffe2
}  // namespace bubblefs

#endif  // BUBBLEFS_UTILS_CAFFE2_WORKSPACE_H_