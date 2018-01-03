// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: likang01(com@baidu.com)

// tera/src/zk/zk_adapter.cc

#include "utils/tera_zk_adapter.h"
#include <errno.h>
#include <functional>
#include "platform/base_error.h"
#include "utils/bdcom_this_thread.h"

namespace bubblefs {
namespace mytera {
namespace zk {

const int32_t kMaxNodeDataLen = 10240;

FILE* ZooKeeperAdapter::lib_log_output_ = NULL;
port::Mutex ZooKeeperAdapter::lib_log_mutex_;

struct ZooKeeperWatch {
    pthread_mutex_t mutex;
    bool watch_value;
    bool watch_exist;
    bool watch_child;

    ZooKeeperWatch()
        : watch_value(false), watch_exist(false), watch_child(false) {
        pthread_mutex_init(&mutex, NULL);
    }

    ~ZooKeeperWatch() {
        pthread_mutex_destroy(&mutex);
    }
};

ZooKeeperAdapter::ZooKeeperAdapter()
    : handle_(NULL), state_(ZS_DISCONN), session_id_(-1),
      state_cond_(&state_mutex_), session_timeout_(0), session_timer_id_(0),
      thread_pool_(1) {
}

ZooKeeperAdapter::~ZooKeeperAdapter() {
    Finalize();
}

bool ZooKeeperAdapter::Init(const std::string& server_list,
                            const std::string& root_path,
                            uint32_t session_timeout,
                            const std::string& id,
                            int* zk_errno) {
    MutexLock mutex(&state_mutex_);

    if (NULL != handle_) {
        SetZkAdapterCode(ZE_INITED, zk_errno);
        return false;
    }

    server_list_ = server_list;
    root_path_ = root_path;
    if (*root_path_.end() == '/') {
        root_path_.resize(root_path_.size() - 1);
    }
    id_ = id;

    handle_ = zookeeper_init((server_list_ + root_path_).c_str(),
                              EventCallBack, session_timeout, NULL, this, 0);
    if (NULL == handle_) {
        PRINTF_ERROR("zookeeper_init fail : %s\n", zerror(errno));
        SetZkAdapterCode(ZE_SESSION, zk_errno);
        return false;
    }

    while (state_ == ZS_DISCONN || state_ == ZS_CONNECTING) {
        state_cond_.Wait();
    }

    int code = ZE_OK;
    // succe
    if (state_ == ZS_CONNECTED) {
        pthread_rwlock_init(&watcher_lock_, NULL);
        pthread_rwlock_init(&locks_lock_, NULL);

        PRINTF_INFO("zookeeper_init success\n");
        SetZkAdapterCode(code, zk_errno);
        return true;
    }

    // fail
    if (state_ == ZS_TIMEOUT) {
        code = ZE_SESSION;
    } else if (state_ == ZS_AUTH) {
        code = ZE_AUTH;
    } else {
        code = ZE_UNKNOWN;
    }
    zookeeper_close(handle_);
    handle_ = NULL;
    state_ = ZS_DISCONN;

    PRINTF_ERROR("zookeeper_init fail : %s\n", ZkErrnoToString(code).c_str());
    SetZkAdapterCode(code, zk_errno);
    return false;
}

void ZooKeeperAdapter::Finalize() {
    zhandle_t* old_handle;
    {
        MutexLock mutex(&state_mutex_);
        if (NULL == handle_) {
            return;
        }
        old_handle = handle_;
        handle_ = NULL;
    }
    int ret = zookeeper_close(old_handle);
    if (ret == ZOK) {
        PRINTF_INFO("zookeeper_close success\n");
    } else {
        PRINTF_ERROR("zookeeper_close fail : %s\n", zerror(ret));
    }
    {
        MutexLock mutex(&state_mutex_);
        pthread_rwlock_destroy(&locks_lock_);
        pthread_rwlock_destroy(&watcher_lock_);
        locks_.clear();
        watchers_.clear();
        state_ = ZS_DISCONN;
        thread_pool_.CancelTask(session_timer_id_);
        session_timer_id_ = 0;
        PRINTF_INFO("zookeeper_session_timeout_timer has gone, safe to finalize.\n");
    }
}

bool ZooKeeperAdapter::CreatePersistentNode(const std::string& path,
                                            const std::string& value,
                                            int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    return Create(path, value, 0, NULL, zk_errno);
}

bool ZooKeeperAdapter::CreateEphemeralNode(const std::string& path,
                                           const std::string& value,
                                           int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    return Create(path, value, ZOO_EPHEMERAL, NULL, zk_errno);
}

bool ZooKeeperAdapter::CreateSequentialEphemeralNode(const std::string& path,
                                                     const std::string& value,
                                                     std::string* ret_path,
                                                     int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    return Create(path, value, ZOO_EPHEMERAL | ZOO_SEQUENCE, ret_path, zk_errno);
}

bool ZooKeeperAdapter::Create(const std::string& path, const std::string& value,
                              int flag, std::string* ret_path, int* zk_errno) {
    state_mutex_.AssertHeld();
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    int value_len = value.size();
    if (value_len == 0) {
        value_len = -1;
    }

    size_t root_path_len = root_path_.size();
    size_t path_len = path.size();
    char * ret_path_buf = NULL;
    size_t ret_path_size = 0;
    if (ret_path != NULL) {
        ret_path_size = root_path_len + path_len + 11;
        ret_path_buf = new char[ret_path_size];
    }

    int ret = zoo_create(handle_, path.c_str(), value.c_str(), value_len,
                         &ZOO_OPEN_ACL_UNSAFE, flag, ret_path_buf,
                         ret_path_size);
    if (ZOK == ret) {
        if (NULL != ret_path) {
            size_t ret_path_len = strlen(ret_path_buf);
            if (((flag & ZOO_SEQUENCE) == 1 &&
                ret_path_len == root_path_len + path_len + 10) ||
                ((flag & ZOO_SEQUENCE) == 0 &&
                ret_path_len == root_path_len + path_len)) {
                // compatible to zk 3.3.x
                *ret_path = ret_path_buf + root_path_len;
            } else {
                *ret_path = ret_path_buf;
            }
        }
        PRINTF_INFO("zoo_create success\n");
    } else {
        PRINTF_WARN("zoo_create fail : %s\n", zerror(ret));
    }

    if (NULL != ret_path_buf) {
        delete[] ret_path_buf;
    }

    switch (ret) {
        case ZOK:
            SetZkAdapterCode(ZE_OK, zk_errno);
            return true;
        case ZNONODE:
            SetZkAdapterCode(ZE_NO_PARENT, zk_errno);
            return false;
        case ZNODEEXISTS:
            SetZkAdapterCode(ZE_EXIST, zk_errno);
            return false;
        case ZNOAUTH:
            SetZkAdapterCode(ZE_AUTH, zk_errno);
            return false;
        case ZNOCHILDRENFOREPHEMERALS:
            SetZkAdapterCode(ZE_ENTITY_PARENT, zk_errno);
            return false;
        case ZBADARGUMENTS:
            SetZkAdapterCode(ZE_ARG, zk_errno);
            return false;
        case ZINVALIDSTATE:
            SetZkAdapterCode(ZE_SESSION, zk_errno);
            return false;
        case ZMARSHALLINGERROR:
            SetZkAdapterCode(ZE_SYSTEM, zk_errno);
            return false;
        default:
            SetZkAdapterCode(ZE_UNKNOWN, zk_errno);
            return false;
    }
}

bool ZooKeeperAdapter::DeleteNode(const std::string& path, int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    int ret = zoo_delete(handle_, path.c_str(), -1);
    if (ZOK == ret) {
        PRINTF_INFO("zoo_delete success\n");
    } else {
        PRINTF_WARN("zoo_delete fail : %s\n", zerror(ret));
    }

    switch (ret) {
        case ZOK:
            SetZkAdapterCode(ZE_OK, zk_errno);
            return true;
        case ZNONODE:
            SetZkAdapterCode(ZE_NOT_EXIST, zk_errno);
            return false;
        case ZNOAUTH:
            SetZkAdapterCode(ZE_AUTH, zk_errno);
            return false;
        case ZBADVERSION: // impossible
            SetZkAdapterCode(ZE_UNKNOWN, zk_errno);
            return false;
        case ZNOTEMPTY:
            SetZkAdapterCode(ZE_HAS_CHILD, zk_errno);
            return false;
        case ZBADARGUMENTS:
            SetZkAdapterCode(ZE_ARG, zk_errno);
            return false;
        case ZINVALIDSTATE:
            SetZkAdapterCode(ZE_SESSION, zk_errno);
            return false;
        case ZMARSHALLINGERROR:
            SetZkAdapterCode(ZE_SYSTEM, zk_errno);
            return false;
        default:
            SetZkAdapterCode(ZE_UNKNOWN, zk_errno);
            return false;
    }
}

bool ZooKeeperAdapter::ReadNode(const std::string& path, std::string* value,
                                int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    int ret = GetWrapper(path, false, value);
    SetZkAdapterCode(ret, zk_errno);
    return (ZE_OK == ret);
}

bool ZooKeeperAdapter::ReadAndWatchNode(const std::string& path,
                                        std::string* value, int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    pthread_rwlock_wrlock(&watcher_lock_);
    std::pair<WatcherMap::iterator, bool> insert_ret = watchers_.insert(
        std::pair<std::string, ZooKeeperWatch*>(path, NULL));
    struct ZooKeeperWatch*& watch = insert_ret.first->second;
    if (NULL == watch) {
        watch = new ZooKeeperWatch;
    }
    pthread_mutex_lock(&watch->mutex);
    pthread_rwlock_unlock(&watcher_lock_);

    bool is_watch = false;
    if (!watch->watch_value) {
        is_watch = true;
    } else {
        pthread_mutex_unlock(&watch->mutex);
        PRINTF_INFO("watch has been set before\n");
    }

    int ret = GetWrapper(path, is_watch, value);
    if (ZE_OK == ret) {
        if (is_watch) {
            watch->watch_value = true;
            pthread_mutex_unlock(&watch->mutex);
        }
        SetZkAdapterCode(ZE_OK, zk_errno);
        return true;
    } else {
        if (is_watch) {
            pthread_mutex_unlock(&watch->mutex);
        }
        SetZkAdapterCode(ret, zk_errno);
        return false;
    }
}

bool ZooKeeperAdapter::ListChildren(const std::string& path,
                                    std::vector<std::string>* child_list,
                                    std::vector<std::string>* value_list,
                                    int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    int ret = GetChildrenWrapper(path, false, child_list, value_list);
    SetZkAdapterCode(ret, zk_errno);
    return (ZE_OK == ret);
}

bool ZooKeeperAdapter::ListAndWatchChildren(const std::string& path,
                                            std::vector<std::string>* child_list,
                                            std::vector<std::string>* value_list,
                                            int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    pthread_rwlock_wrlock(&watcher_lock_);
    std::pair<WatcherMap::iterator, bool> insert_ret = watchers_.insert(
        std::pair<std::string, ZooKeeperWatch*>(path, NULL));
    struct ZooKeeperWatch*& watch = insert_ret.first->second;
    if (NULL == watch) {
        watch = new ZooKeeperWatch;
    }
    pthread_mutex_lock(&watch->mutex);
    pthread_rwlock_unlock(&watcher_lock_);

    bool is_watch = false;
    if (!watch->watch_child) {
        is_watch = true;
    } else {
        pthread_mutex_unlock(&watch->mutex);
        PRINTF_INFO("is_watch has been set before\n");
    }

    int ret = GetChildrenWrapper(path, is_watch, child_list, value_list);
    if (ZE_OK == ret) {
        if (is_watch) {
            watch->watch_child = true;
            pthread_mutex_unlock(&watch->mutex);
        }
        SetZkAdapterCode(ret, zk_errno);
        return true;
    } else {
        if (is_watch) {
            pthread_mutex_unlock(&watch->mutex);
        }
        SetZkAdapterCode(ret, zk_errno);
        return false;
    }
}

bool ZooKeeperAdapter::CheckExist(const std::string&path, bool* is_exist,
                                  int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    int ret = ExistsWrapper(path, false, is_exist);
    SetZkAdapterCode(ret, zk_errno);
    return (ZE_OK == ret);
}

bool ZooKeeperAdapter::CheckAndWatchExist(const std::string& path, bool* is_exist,
                                          int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    bool is_watch = false;
    pthread_rwlock_wrlock(&watcher_lock_);
    std::pair<WatcherMap::iterator, bool> insert_ret = watchers_.insert(
        std::pair<std::string, ZooKeeperWatch*>(path, NULL));
    struct ZooKeeperWatch*& watch = insert_ret.first->second;
    if (NULL == watch) {
        watch = new ZooKeeperWatch;
    }
    pthread_mutex_lock(&watch->mutex);
    pthread_rwlock_unlock(&watcher_lock_);
    if (!watch->watch_exist) {
        is_watch = true;
    } else {
        pthread_mutex_unlock(&watch->mutex);
        PRINTF_INFO("is_watch has been set before\n");
    }

    int ret = ExistsWrapper(path, is_watch, is_exist);
    if (ZE_OK == ret) {
        if (is_watch) {
            watch->watch_exist = true;
            pthread_mutex_unlock(&watch->mutex);
        }
    } else {
        if (is_watch) {
            pthread_mutex_unlock(&watch->mutex);
        }
    }
    SetZkAdapterCode(ret, zk_errno);
    return (ZE_OK == ret);
}


bool ZooKeeperAdapter::CheckAndWatchExistForLock(const std::string& path,
                                                 bool* is_exist, int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    int ret = ExistsWrapperForLock(path, is_exist);
    SetZkAdapterCode(ret, zk_errno);
    return (ZE_OK == ret);
}

bool ZooKeeperAdapter::WriteNode(const std::string& path,
                                 const std::string& value, int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    int ret = zoo_set(handle_, path.c_str(), value.c_str(), value.size(), -1);
    if (ZOK == ret) {
        PRINTF_INFO("zoo_set success\n");
    } else {
        PRINTF_WARN("zoo_set fail : %s\n", zerror(ret));
    }

    switch (ret) {
        case ZOK:
            SetZkAdapterCode(ZE_OK, zk_errno);
            return true;
        case ZNONODE:
            SetZkAdapterCode(ZE_NOT_EXIST, zk_errno);
            return false;
        case ZNOAUTH:
            SetZkAdapterCode(ZE_AUTH, zk_errno);
            return false;
        case ZBADVERSION:  // impossible
            SetZkAdapterCode(ZE_UNKNOWN, zk_errno);
            return false;
        case ZBADARGUMENTS:
            SetZkAdapterCode(ZE_ARG, zk_errno);
            return false;
        case ZINVALIDSTATE:
            SetZkAdapterCode(ZE_SESSION, zk_errno);
            return false;
        case ZMARSHALLINGERROR:
            SetZkAdapterCode(ZE_SYSTEM, zk_errno);
            return false;
        default:
            SetZkAdapterCode(ZE_UNKNOWN, zk_errno);
            return false;
    }
}

void ZooKeeperAdapter::EventCallBack(zhandle_t * zh, int type, int state,
                                     const char * node_path, void * watch_ctx) {
    PRINTF_INFO("recv event: type=%s, state=%s, path=%s\n",
                ZooTypeToString(type).c_str(), ZooStateToString(state).c_str(), node_path);

    if (NULL == watch_ctx) {
        return;
    }
    ZooKeeperAdapter* zk_adapter = (ZooKeeperAdapter*)watch_ctx;

    MutexLock mutex(&zk_adapter->state_mutex_);
    if (zh != zk_adapter->handle_) {
        PRINTF_WARN("zhandle not match\n");
        return;
    }
    // handle_ is guaranteed (by zk lib) to be valid within callback func.
    // no need to check it.

    if (ZOO_SESSION_EVENT == type) {
        zk_adapter->SessionEventCallBack(state);
        return;
    }

    if (NULL == node_path) {
        PRINTF_WARN("path is missing\n");
        return;
    }

    std::string path = node_path;
    if (!ZooKeeperUtil::IsValidPath(path)) {
        PRINTF_WARN("path is invalid\n");
        return;
    }

    if (ZOO_CREATED_EVENT == type) {
        zk_adapter->CreateEventCallBack(path);
    } else if (ZOO_DELETED_EVENT == type) {
        zk_adapter->DeleteEventCallBack(path);
    } else if (ZOO_CHANGED_EVENT == type) {
        zk_adapter->ChangeEventCallBack(path);
    } else if (ZOO_CHILD_EVENT == type) {
        zk_adapter->ChildEventCallBack(path);
    } else if (ZOO_NOTWATCHING_EVENT == type) {
        zk_adapter->WatchLostEventCallBack(state, path);
    } else {
        PRINTF_WARN("unknown event type: %d\n", type);
    }
}

void ZooKeeperAdapter::CreateEventCallBack(std::string path) {
    PRINTF_INFO("CreateEventCallBack: path=[%s]\n", path.c_str());

    pthread_rwlock_wrlock(&watcher_lock_);
    WatcherMap::iterator itor = watchers_.find(path);
    if (itor == watchers_.end()) {
        pthread_rwlock_unlock(&watcher_lock_);
        PRINTF_INFO("watch not match\n");
        return;
    }

    ZooKeeperWatch * watch = itor->second;
    pthread_mutex_lock(&watch->mutex);
    pthread_rwlock_unlock(&watcher_lock_);
    if (!watch->watch_exist) {
        pthread_mutex_unlock(&watch->mutex);
        PRINTF_WARN("watch not match\n");
        return;
    }

    bool is_exist;
    int ret = ExistsWrapper(path, true, &is_exist);
    if (ZE_OK == ret) {
        pthread_mutex_unlock(&watch->mutex);
        state_mutex_.Unlock();
        OnNodeCreated(path);
        if (!is_exist) {
            OnNodeDeleted(path);
        }
        state_mutex_.Lock();
    } else {
        watch->watch_exist = false;
        pthread_mutex_unlock(&watch->mutex);
        TryCleanWatch(path);
        state_mutex_.Unlock();
        OnWatchFailed(path, ZT_WATCH_EXIST, ret);
        state_mutex_.Lock();
    }
}

void ZooKeeperAdapter::DeleteEventCallBack(std::string path) {
    PRINTF_INFO("DeleteEventCallBack: path=[%s]\n", path.c_str());

    pthread_rwlock_wrlock(&watcher_lock_);
    WatcherMap::iterator itor = watchers_.find(path);
    if (itor == watchers_.end()) {
        pthread_rwlock_unlock(&watcher_lock_);
        PRINTF_INFO("watch not match\n");
        return;
    }

    ZooKeeperWatch * watch = itor->second;
    pthread_mutex_lock(&watch->mutex);
    pthread_rwlock_unlock(&watcher_lock_);

    if (!watch->watch_exist && !watch->watch_value && !watch->watch_child) {
        pthread_mutex_unlock(&watch->mutex);
        PRINTF_WARN("watch not match\n");
        return;
    }

    bool is_watch_exist = watch->watch_exist;
    bool is_exist;
    int ret = ExistsWrapper(path, true, &is_exist);
    if (ZE_OK == ret) {
        watch->watch_value = false;
        watch->watch_child = false;
        pthread_mutex_unlock(&watch->mutex);
        if (!is_watch_exist) {
            TryCleanWatch(path);
        }
        state_mutex_.Unlock();
        OnNodeDeleted(path);
        if (is_exist && is_watch_exist) {
            OnNodeCreated(path);
        }
        state_mutex_.Lock();
    } else {
        watch->watch_exist = false;
        watch->watch_value = false;
        watch->watch_child = false;
        pthread_mutex_unlock(&watch->mutex);
        TryCleanWatch(path);
        state_mutex_.Unlock();
        OnNodeDeleted(path);
        if (is_watch_exist) {
            OnWatchFailed(path, ZT_WATCH_EXIST, ret);
        }
        state_mutex_.Lock();
    }
}

void ZooKeeperAdapter::ChangeEventCallBack(std::string path) {
    PRINTF_INFO("ChangeEventCallBack: path=[%s]\n", path.c_str());

    pthread_rwlock_wrlock(&watcher_lock_);
    WatcherMap::iterator itor = watchers_.find(path);
    if (itor == watchers_.end()) {
        pthread_rwlock_unlock(&watcher_lock_);
        PRINTF_INFO("watch not match\n");
        return;
    }

    ZooKeeperWatch * watch = itor->second;
    pthread_mutex_lock(&watch->mutex);
    pthread_rwlock_unlock(&watcher_lock_);

    if (!watch->watch_value) {
        pthread_mutex_unlock(&watch->mutex);
        PRINTF_WARN("watch not match\n");
        return;
    }

    std::string value;
    int ret = GetWrapper(path, true, &value);
    if (ZE_OK == ret) {
        pthread_mutex_unlock(&watch->mutex);
        state_mutex_.Unlock();
        OnNodeValueChanged(path, value);
        state_mutex_.Lock();
    } else if (ZE_NOT_EXIST == ret) {
        watch->watch_value = false;
        watch->watch_child = false;
        pthread_mutex_unlock(&watch->mutex);
        TryCleanWatch(path);
        state_mutex_.Unlock();
        OnNodeDeleted(path);
        state_mutex_.Lock();
    } else {
        watch->watch_value = false;
        pthread_mutex_unlock(&watch->mutex);
        TryCleanWatch(path);
        state_mutex_.Unlock();
        OnWatchFailed(path, ZT_WATCH_VALUE, ret);
        state_mutex_.Lock();
    }
}

void ZooKeeperAdapter::ChildEventCallBack(std::string path) {
    PRINTF_INFO("ChildEventCallBack: path=[%s]\n", path.c_str());

    pthread_rwlock_wrlock(&watcher_lock_);
    WatcherMap::iterator itor = watchers_.find(path);
    if (itor == watchers_.end()) {
        pthread_rwlock_unlock(&watcher_lock_);
        PRINTF_INFO("watch not match\n");
        return;
    }

    ZooKeeperWatch * watch = itor->second;
    pthread_mutex_lock(&watch->mutex);
    pthread_rwlock_unlock(&watcher_lock_);

    if (!watch->watch_child) {
        pthread_mutex_unlock(&watch->mutex);
        PRINTF_WARN("watch not match\n");
        return;
    }

    std::vector<std::string> child_list;
    std::vector<std::string> value_list;
    int ret = GetChildrenWrapper(path, true, &child_list, &value_list);
    if (ZE_OK == ret) {
        pthread_mutex_unlock(&watch->mutex);
        state_mutex_.Unlock();
        OnChildrenChanged(path, child_list, value_list);
        state_mutex_.Lock();
    } else if (ZE_NOT_EXIST == ret) {
        watch->watch_child = false;
        watch->watch_value = false;
        pthread_mutex_unlock(&watch->mutex);
        TryCleanWatch(path);
        state_mutex_.Unlock();
        OnNodeDeleted(path);
        state_mutex_.Lock();
    } else {
        watch->watch_child = false;
        pthread_mutex_unlock(&watch->mutex);
        TryCleanWatch(path);
        state_mutex_.Unlock();
        OnWatchFailed(path, ZT_WATCH_CHILD, ret);
        state_mutex_.Lock();
    }
}

void ZooKeeperAdapter::SessionTimeoutWrapper() {
    this->OnSessionTimeout();
    MutexLock mutex(&state_mutex_);
    session_timer_id_ = 0;
}

void ZooKeeperAdapter::SessionEventCallBack(int state) {
    if (ZOO_CONNECTED_STATE == state) {
        if (ZS_CONNECTING == state_) {
            if (!thread_pool_.CancelTask(session_timer_id_)) {
                PRINTF_WARN("session timeout timer is triggered\n");
                return;
            }
            session_timer_id_ = 0;
        }
        const clientid_t *cid = zoo_client_id(handle_);
        if (cid == NULL) {
            PRINTF_WARN("zoo_client_id fail\n");
            return;
        }
        session_id_ = cid->client_id;
        state_ = ZS_CONNECTED;
        state_cond_.Signal();
        session_timeout_ = zoo_recv_timeout(handle_);
        PRINTF_INFO("connected to zk server, session timeout: %d ms\n",
                    session_timeout_);
    } else if (ZOO_CONNECTING_STATE == state || ZOO_ASSOCIATING_STATE == state) {
        if (ZS_CONNECTED == state_) {
            PRINTF_INFO("disconnect from zk server, enable timer: %d ms\n",
                        session_timeout_);
            mybdcom::ThreadPool::Task task =
                std::bind(&ZooKeeperAdapter::SessionTimeoutWrapper, this);
            session_timer_id_ = thread_pool_.DelayTask(session_timeout_, task);
        }
        session_id_ = -1;
        state_ = ZS_CONNECTING;
        state_cond_.Signal();
    } else if (ZOO_AUTH_FAILED_STATE == state) {
        session_id_ = -1;
        state_ = ZS_AUTH;
        state_cond_.Signal();
    } else if (ZOO_EXPIRED_SESSION_STATE == state) {
        session_id_ = -1;
        state_ = ZS_TIMEOUT;
        state_cond_.Signal();
        state_mutex_.Unlock();
        OnSessionTimeout();
        state_mutex_.Lock();
    }
}

void ZooKeeperAdapter::WatchLostEventCallBack(int state, std::string path) {
    // shit...
}

bool ZooKeeperAdapter::SyncLock(const std::string& path, int* zk_errno,
                                int32_t timeout) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }
    bool ret_val;

    pthread_rwlock_wrlock(&locks_lock_);
    std::pair<LockMap::iterator, bool> insert_ret = locks_.insert(
        std::pair<std::string, ZooKeeperLock*>(path, NULL));
    if (!insert_ret.second) {
        ZooKeeperLock * lock = insert_ret.first->second;
        if (lock == NULL || !lock->IsAcquired()) {
            PRINTF_INFO("lock exists but is not acquired\n");
        } else {
            PRINTF_INFO("lock has been acquired\n");
        }
        pthread_rwlock_unlock(&locks_lock_);
        SetZkAdapterCode(ZE_LOCK_EXIST, zk_errno);
        return false;
    }
    pthread_rwlock_unlock(&locks_lock_);

    timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    end_time.tv_sec = start_time.tv_sec + timeout;
    end_time.tv_usec = start_time.tv_usec;

    LockCompletion * callback_param = new LockCompletion();
    ZooKeeperLock * lock = new ZooKeeperLock(this, path, SyncLockCallback,
                                             callback_param);
    callback_param->SetLock(lock);

    state_mutex_.Unlock();
    if (!lock->BeginLock(zk_errno)) {
        state_mutex_.Lock();
        delete callback_param;
        delete lock;
        pthread_rwlock_wrlock(&locks_lock_);
        locks_.erase(path);
        pthread_rwlock_unlock(&locks_lock_);
        return false;
    }
    state_mutex_.Lock();

    pthread_rwlock_wrlock(&locks_lock_);
    locks_[path] = lock;
    pthread_rwlock_unlock(&locks_lock_);

    timeval now_time;
    gettimeofday(&now_time, NULL);
    if (timeout > 0 && (now_time.tv_sec > end_time.tv_sec
        || (now_time.tv_sec == end_time.tv_sec && now_time.tv_usec
            > end_time.tv_usec))) {
        if (lock->IsAcquired()) {
            SetZkAdapterCode(ZE_OK, zk_errno);
            return true;
        } else {
            SetZkAdapterCode(ZE_LOCK_TIMEOUT, zk_errno);
            return false;
        }
    }

    state_mutex_.Unlock();
    if (timeout > 0) {
        ret_val = callback_param->Wait(zk_errno, &end_time);
    } else {
        ret_val = callback_param->Wait(zk_errno);
    }
    state_mutex_.Lock();
    return ret_val;
}

bool ZooKeeperAdapter::AsyncLock(const std::string& path,
                                 LOCK_CALLBACK callback_func,
                                 void * callback_param, int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    pthread_rwlock_wrlock(&locks_lock_);
    std::pair<LockMap::iterator, bool> insert_ret = locks_.insert(
        std::pair<std::string, ZooKeeperLock*>(path, NULL));
    if (!insert_ret.second) {
        ZooKeeperLock * lock = insert_ret.first->second;
        if (lock == NULL || !lock->IsAcquired()) {
            PRINTF_INFO("lock exists but is not acquired\n");
        } else {
            PRINTF_INFO("lock has been acquired\n");
        }
        pthread_rwlock_unlock(&locks_lock_);
        SetZkAdapterCode(ZE_LOCK_EXIST, zk_errno);
        return false;
    }
    pthread_rwlock_unlock(&locks_lock_);
    ZooKeeperLock * lock = new ZooKeeperLock(this, path, callback_func,
                                             callback_param);
    state_mutex_.Unlock();
    if (!lock->BeginLock(zk_errno)) {
        state_mutex_.Lock();
        pthread_rwlock_wrlock(&locks_lock_);
        locks_.erase(path);
        pthread_rwlock_unlock(&locks_lock_);
        delete lock;
        return false;
    } else {
        state_mutex_.Lock();
        pthread_rwlock_wrlock(&locks_lock_);
        locks_[path] = lock;
        pthread_rwlock_unlock(&locks_lock_);
        return true;
    }
}

void ZooKeeperAdapter::SyncLockCallback(const std::string& path, int err,
                                        void * param) {
    LockCompletion * comp = (LockCompletion *) param;
    comp->Signal(err);
}

bool ZooKeeperAdapter::CancelLock(const std::string& path, int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    pthread_rwlock_wrlock(&locks_lock_);
    LockMap::iterator itor = locks_.find(path);
    if (itor == locks_.end()) {
        pthread_rwlock_unlock(&locks_lock_);
        PRINTF_WARN("lock not exist\n");
        SetZkAdapterCode(ZE_LOCK_NOT_EXIST, zk_errno);
        return false;
    }

    ZooKeeperLock * lock = itor->second;
    state_mutex_.Unlock();
    if (!lock->CancelLock(zk_errno)) {
        state_mutex_.Lock();
        delete lock;
        locks_.erase(itor);
        pthread_rwlock_unlock(&locks_lock_);
        return false;
    } else {
        state_mutex_.Lock();
        pthread_rwlock_unlock(&locks_lock_);
        return true;
    }
}

bool ZooKeeperAdapter::Unlock(const std::string& path, int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (!ZooKeeperUtil::IsValidPath(path)) {
        SetZkAdapterCode(ZE_ARG, zk_errno);
        return false;
    }
    if (NULL == handle_) {
        SetZkAdapterCode(ZE_NOT_INIT, zk_errno);
        return false;
    }

    pthread_rwlock_wrlock(&locks_lock_);
    LockMap::iterator itor = locks_.find(path);
    if (itor == locks_.end() || itor->second == NULL) {
        pthread_rwlock_unlock(&locks_lock_);
        PRINTF_WARN("lock not exist\n");
        SetZkAdapterCode(ZE_LOCK_NOT_EXIST, zk_errno);
        return false;
    }

    ZooKeeperLock * lock = itor->second;
    state_mutex_.Unlock();
    if (lock->Unlock(zk_errno)) {
        state_mutex_.Lock();
        delete lock;
        locks_.erase(itor);
        pthread_rwlock_unlock(&locks_lock_);
        return true;
    } else {
        state_mutex_.Lock();
        pthread_rwlock_unlock(&locks_lock_);
        return false;
    }
}

void ZooKeeperAdapter::GetId(std::string* id) {
    MutexLock mutex(&state_mutex_);
    *id = id_;
}

void ZooKeeperAdapter::TryCleanWatch(const std::string& path) {
    state_mutex_.AssertHeld();
    pthread_rwlock_wrlock(&watcher_lock_);
    WatcherMap::iterator itor = watchers_.find(path);
    if (itor == watchers_.end()) {
        pthread_rwlock_unlock(&watcher_lock_);
        return;
    }

    ZooKeeperWatch * watch = itor->second;
    pthread_mutex_lock(&watch->mutex);
    if (!watch->watch_child && !watch->watch_exist && !watch->watch_value) {
        pthread_mutex_unlock(&watch->mutex);
        delete watch;
        watchers_.erase(itor);
    } else {
        pthread_mutex_unlock(&watch->mutex);
    }
    pthread_rwlock_unlock(&watcher_lock_);
}

void ZooKeeperAdapter::LockEventCallBack(zhandle_t * zh, int type, int state,
                                         const char * node_path, void * watch_ctx) {
    PRINTF_INFO("recv lock event: type=%s, state=%s, path=[%s]\n",
                ZooTypeToString(type).c_str(), ZooStateToString(state).c_str(), node_path);

    if (ZOO_DELETED_EVENT != type) {
        PRINTF_WARN("only allow DELETE_EVENT for lock\n");
        return;
    }

    if (NULL == watch_ctx) {
        return;
    }
    ZooKeeperAdapter* zk_adapter = (ZooKeeperAdapter*)watch_ctx;

    {
        MutexLock mutex(&zk_adapter->state_mutex_);
        if (zh != zk_adapter->handle_) {
            PRINTF_WARN("zhandle not match\n");
            return;
        }
    }

    if (NULL == node_path) {
        PRINTF_WARN("path is missing\n");
        return;
    }

    std::string path = node_path;
    if (!ZooKeeperUtil::IsValidPath(path)) {
        PRINTF_WARN("path is invalid\n");
        return;
    }

    zk_adapter->LockEventCallBack(path);
}

void ZooKeeperAdapter::LockEventCallBack(std::string path) {
    PRINTF_INFO("LockEventCallBack: path=[%s]\n", path.c_str());
    MutexLock mutex(&state_mutex_);

    std::string lock_path;
    ZooKeeperUtil::GetParentPath(path, &lock_path);

    pthread_rwlock_wrlock(&locks_lock_);
    LockMap::iterator itor = locks_.find(lock_path);
    if (itor == locks_.end()) {
        pthread_rwlock_unlock(&locks_lock_);
        PRINTF_WARN("lock [%s] not exist\n", lock_path.c_str());
        return;
    }
    ZooKeeperLock* lock = itor->second;
    if (lock == NULL) {
        pthread_rwlock_unlock(&locks_lock_);
        return;
    }
    state_mutex_.Unlock();
    lock->OnWatchNodeDeleted(path);
    state_mutex_.Lock();
    pthread_rwlock_unlock(&locks_lock_);
}

bool ZooKeeperAdapter::GetSessionId(int64_t* session_id, int* zk_errno) {
    MutexLock mutex(&state_mutex_);
    if (ZS_CONNECTED == state_) {
        *session_id = session_id_;
        SetZkAdapterCode(ZE_OK, zk_errno);
        return true;
    }
    SetZkAdapterCode(ZE_SESSION, zk_errno);
    return false;
}

bool ZooKeeperAdapter::SetLibraryLogOutput(const std::string& file) {
    MutexLock mutex(&lib_log_mutex_);
    FILE* new_log = fopen(file.c_str(), "a");
    if (NULL == new_log) {
        PRINTF_WARN("fail to open file [%s]:%s\n", file.c_str(), strerror(errno));
        return false;
    }
    zoo_set_log_stream(new_log);
    if (NULL != lib_log_output_) {
        fclose(lib_log_output_);
    }
    lib_log_output_ = new_log;
    return true;
}

int ZooKeeperAdapter::ExistsWrapper(const std::string& path, bool is_watch,
                                    bool* is_exist) {
    state_mutex_.AssertHeld();
    struct Stat stat;
    int ret = zoo_exists(handle_, path.c_str(), is_watch, &stat);
    if (ZOK == ret) {
        *is_exist = true;
        PRINTF_INFO("zoo_exists success\n");
    } else if (ZNONODE == ret) {
        *is_exist = false;
        PRINTF_INFO("zoo_exists success\n");
    } else {
        PRINTF_WARN("zoo_exists fail: %s\n", zerror(ret));
    }

    switch (ret) {
        case ZOK:
        case ZNONODE:
            return ZE_OK;
        case ZNOAUTH:
            return ZE_AUTH;
        case ZBADARGUMENTS:
            return ZE_ARG;
        case ZINVALIDSTATE:
            return ZE_SESSION;
        case ZMARSHALLINGERROR:
            return ZE_SYSTEM;
        default:
            return ZE_UNKNOWN;
    }
}

int ZooKeeperAdapter::ExistsWrapperForLock(const std::string& path,
                                           bool* is_exist) {
    state_mutex_.AssertHeld();
    struct Stat stat;
    int ret = zoo_wexists(handle_, path.c_str(), LockEventCallBack, this, &stat);
    if (ZOK == ret) {
        *is_exist = true;
        PRINTF_INFO("zoo_exists success\n");
    } else if (ZNONODE == ret) {
        *is_exist = false;
        PRINTF_INFO("zoo_exists success\n");
    } else {
        PRINTF_WARN("zoo_exists fail: %s\n", zerror(ret));
    }

    switch (ret) {
        case ZOK:
        case ZNONODE:
            return ZE_OK;
        case ZNOAUTH:
            return ZE_AUTH;
        case ZBADARGUMENTS:
            return ZE_ARG;
        case ZINVALIDSTATE:
            return ZE_SESSION;
        case ZMARSHALLINGERROR:
            return ZE_SYSTEM;
        default:
            return ZE_UNKNOWN;
    }
}

int ZooKeeperAdapter::GetWrapper(const std::string& path, bool is_watch,
                                 std::string* value) {
    state_mutex_.AssertHeld();
    char* buffer = new char[kMaxNodeDataLen];
    int buffer_len = kMaxNodeDataLen;
    int ret = zoo_get(handle_, path.c_str(), is_watch, buffer, &buffer_len,
                      NULL);
    if (ZOK == ret) {
        if (buffer_len < 0) {
            buffer_len = 0;
        } else if (buffer_len >= kMaxNodeDataLen) {
            buffer_len = kMaxNodeDataLen - 1;
        }
        buffer[buffer_len] = '\0';
        *value = buffer;
        PRINTF_INFO("zoo_get success\n");
    } else {
        PRINTF_WARN("zoo_get fail: %s\n", zerror(ret));
    }
    delete[] buffer;

    switch (ret) {
        case ZOK:
            return ZE_OK;
        case ZNONODE:
            return ZE_NOT_EXIST;
        case ZNOAUTH:
            return ZE_AUTH;
        case ZBADARGUMENTS:
            return ZE_ARG;
        case ZINVALIDSTATE:
            return ZE_SESSION;
        case ZMARSHALLINGERROR:
            return ZE_SYSTEM;
        default:
            return ZE_UNKNOWN;
    }
}

int ZooKeeperAdapter::GetChildrenWrapper(const std::string& path, bool is_watch,
                                         std::vector<std::string>* child_list,
                                         std::vector<std::string>* value_list) {
    state_mutex_.AssertHeld();
    struct String_vector str_vec;
    allocate_String_vector(&str_vec, 0);
    int ret = zoo_get_children(handle_, path.c_str(), is_watch, &str_vec);
    if (ZOK == ret) {
        child_list->clear();
        value_list->clear();
        for (int i = 0; i < str_vec.count; i++) {
            child_list->push_back(str_vec.data[i]);
            std::string child_path = path + '/' + str_vec.data[i];
            std::string value;
            int ret2 = GetWrapper(child_path, false, &value);
            if (ZE_OK != ret2) {
                value = "";
                PRINTF_WARN("read node fail: %d\n", ret2);
            }
            value_list->push_back(value);
        }
        PRINTF_INFO("zoo_get_children success\n");
    } else {
        PRINTF_WARN("zoo_get_children fail: %s\n", zerror(ret));
    }
    deallocate_String_vector(&str_vec);

    switch (ret) {
        case ZOK:
            return ZE_OK;
        case ZNONODE:
            return ZE_NOT_EXIST;
        case ZNOAUTH:
            return ZE_AUTH;
        case ZBADARGUMENTS:
            return ZE_ARG;
        case ZINVALIDSTATE:
            return ZE_SESSION;
        case ZMARSHALLINGERROR:
            return ZE_SYSTEM;
        default:
            return ZE_UNKNOWN;
    }
}

} // namespace zk
} // namespace mytera
} // namespace bubblefs