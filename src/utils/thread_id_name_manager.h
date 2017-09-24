// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// brpc/src/butil/threading/thread_id_name_manager.h

#ifndef BUBBLEFS_UTILS_THREAD_ID_NAME_MANAGER_H_
#define BUBBLEFS_UTILS_THREAD_ID_NAME_MANAGER_H_

#include <map>
#include <string>
#include "platform/base_export.h"
#include "platform/condition_variable.h"
#include "platform/platform_thread.h"
#include "platform/types.h"

template <typename T> struct DefaultSingletonTraits;

namespace bubblefs {
namespace base {

class BASE_EXPORT ThreadIdNameManager {
 public:
  static ThreadIdNameManager* GetInstance();

  static const char* GetDefaultInternedString();

  // Register the mapping between a thread |id| and |handle|.
  void RegisterThread(PlatformThreadHandle::Handle handle, PlatformThreadId id);

  // Set the name for the given id.
  void SetName(PlatformThreadId id, const char* name);

  // Get the name for the given id.
  const char* GetName(PlatformThreadId id);

  // Remove the name for the given id.
  void RemoveName(PlatformThreadHandle::Handle handle, PlatformThreadId id);

  ThreadIdNameManager();
  ~ThreadIdNameManager();
  
 private:
  friend struct DefaultSingletonTraits<ThreadIdNameManager>;

  typedef std::map<PlatformThreadId, PlatformThreadHandle::Handle>
      ThreadIdToHandleMap;
  typedef std::map<PlatformThreadHandle::Handle, std::string*>
      ThreadHandleToInternedNameMap;
  typedef std::map<std::string, std::string*> NameToInternedNameMap;

  // lock_ protects the name_to_interned_name_, thread_id_to_handle_ and
  // thread_handle_to_interned_name_ maps.
  concurrent::Lock lock_;

  NameToInternedNameMap name_to_interned_name_;
  ThreadIdToHandleMap thread_id_to_handle_;
  ThreadHandleToInternedNameMap thread_handle_to_interned_name_;

  // Treat the main process specially as there is no PlatformThreadHandle.
  std::string* main_process_name_;
  PlatformThreadId main_process_id_;

  DISALLOW_COPY_AND_ASSIGN(ThreadIdNameManager);
};

}  // namespace base
}  // namespce bubblefs

#endif  // BUBBLEFS_UTILS_THREAD_ID_NAME_MANAGER_H_