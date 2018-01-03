// Copyright (c) 2015, Baidu.com, Inc. All Rights Reserved
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Author: likang01(com@baidu.com)

// tera/src/zk/zk_lock.cc

#include "utils/tera_zk_lock.h"
#include "platform/base_error.h"
#include "utils/tera_zk_adapter.h"
#include "utils/tera_zk_util.h"

namespace bubblefs {
namespace mytera {
namespace zk {

ZooKeeperLock::ZooKeeperLock(ZooKeeperAdapter * adapter,
                             const std::string& lock_path, LOCK_CALLBACK func,
                             void * param)
    : adapter_(adapter), lock_path_(lock_path), is_acquired_(false),
      callback_func_(func), callback_param_(param) {
    pthread_mutex_init(&mutex_, NULL);
}

ZooKeeperLock::~ZooKeeperLock() {
    pthread_mutex_destroy(&mutex_);
}

bool ZooKeeperLock::BeginLock(int* zk_errno) {
    // use session id as GUID
    // get session id
    int64_t session_id = -1;
    if (!adapter_->GetSessionId(&session_id, zk_errno)) {
        SetZkAdapterCode(ZE_SESSION, zk_errno);
        return false;
    }
    char guid[17];
    sprintf(guid, "%016llx", static_cast<unsigned long long>(session_id));
    PRINTF_INFO("lock GUID = %s\n", guid);

    // get all lock nodes
    std::vector<std::string> child_list;
    std::vector<std::string> value_list;
    if (!adapter_->ListChildren(lock_path_, &child_list, &value_list,
                                 zk_errno)) {
        PRINTF_WARN("list lock path fail: %s\n", ZkErrnoToString(*zk_errno).c_str());
        return false;
    }

    // delete lock nodes with same GUID to avoid conflict
    *zk_errno = ZE_OK;
    std::vector<std::string>::iterator itor;
    for (itor = child_list.begin(); itor != child_list.end(); ++itor) {
        const std::string& name = *itor;
        if (name.size() > 16 && 0 == strncmp(name.c_str(), guid, 16)
            && name[16] == '#') {
            std::string child_path = lock_path_ + "/" + name;
            int zk_ret;
            if (!adapter_->DeleteNode(child_path, &zk_ret)) {
                PRINTF_WARN("delete same GUID lock node fail: %s\n", ZkErrnoToString(*zk_errno).c_str());
                SetZkAdapterCode(zk_ret, zk_errno);
            }
        }
    }
    if (ZE_OK != *zk_errno) {
        return false;
    }

    // create lock node
    std::string lock_node_path = lock_path_ + "/" + guid + "#";
    std::string lock_node_data;
    adapter_->GetId(&lock_node_data);
    std::string ret_path;
    if (!adapter_->CreateSequentialEphemeralNode(lock_node_path, lock_node_data,
                                                  &ret_path, zk_errno)) {
        PRINTF_WARN("create my lock node fail: %s\n", ZkErrnoToString(*zk_errno).c_str());
        return false;
    }

    child_list.clear();
    value_list.clear();
    if (!adapter_->ListChildren(lock_path_, &child_list, &value_list,
                                 zk_errno)) {
        PRINTF_WARN("list lock path fail: %s\n", ZkErrnoToString(*zk_errno).c_str());
        return false;
    }
    if (child_list.size() == 0) {
        PRINTF_WARN("lock path is empty. where is my node?\n");
        SetZkAdapterCode(ZE_SYSTEM, zk_errno);
        return false;
    }

    const std::string& self_name = ret_path;
    int32_t self_seq_no = ZooKeeperUtil::GetSequenceNo(self_name);
    if (self_seq_no < 0) {
        PRINTF_WARN("sequence node name is invalid\n");
        SetZkAdapterCode(ZE_SYSTEM, zk_errno);
        return false;
    }
    self_node_.name = ZooKeeperUtil::GetNodeName(ret_path.c_str());
    self_node_.seq = self_seq_no;

    for (itor = child_list.begin(); itor != child_list.end(); ++itor) {
        const std::string& name = *itor;
        int32_t seq_no = ZooKeeperUtil::GetSequenceNo(name);
        if (seq_no >= 0 && seq_no < self_seq_no) {
            struct SeqNode child = {name, seq_no};
            node_list_.push(child);
        }
    }

    if (node_list_.empty()) {
        PRINTF_INFO("get lock success\n");
        is_acquired_ = true;
        callback_func_(lock_path_, ZE_OK, callback_param_);
        SetZkAdapterCode(ZE_OK, zk_errno);
        return true;
    }

    // std::sort(node_list_.begin(), node_list_.end());

    do {
        watch_path_ = lock_path_ + "/" + node_list_.top().name;
        bool is_exist;
        if (!adapter_->CheckAndWatchExistForLock(watch_path_, &is_exist,
                                                  zk_errno)) {
            return false;
        }
        if (is_exist) {
            SetZkAdapterCode(ZE_OK, zk_errno);
            return true;
        } else {
            node_list_.pop();
        }
    } while (!node_list_.empty());

    PRINTF_INFO("get lock success\n");
    is_acquired_ = true;
    callback_func_(lock_path_, ZE_OK, callback_param_);
    SetZkAdapterCode(ZE_OK, zk_errno);
    return true;
}

bool ZooKeeperLock::CancelLock(int* zk_errno) {
    pthread_mutex_lock(&mutex_);
    if (IsAcquired()) {
        pthread_mutex_unlock(&mutex_);
        PRINTF_WARN("lock is acquired\n");
        SetZkAdapterCode(ZE_LOCK_ACQUIRED, zk_errno);
        return false;
    }

    pthread_mutex_unlock(&mutex_);
    callback_func_(lock_path_, ZE_LOCK_CANCELED, callback_param_);
    PRINTF_INFO("unlock success\n");
    SetZkAdapterCode(ZE_OK, zk_errno);
    return true;
}

bool ZooKeeperLock::Unlock(int* zk_errno) {
    pthread_mutex_lock(&mutex_);
    if (!IsAcquired()) {
        pthread_mutex_unlock(&mutex_);
        PRINTF_WARN("lock is not acquired\n");
        SetZkAdapterCode(ZE_LOCK_NOT_ACQUIRED, zk_errno);
        return false;
    }

    if (!adapter_->DeleteNode(lock_path_ + "/" + self_node_.name, zk_errno)) {
        pthread_mutex_unlock(&mutex_);
        PRINTF_WARN("unlock fail: %s\n", ZkErrnoToString(*zk_errno).c_str());
        return false;
    }

    pthread_mutex_unlock(&mutex_);
    PRINTF_INFO("unlock success\n");
    SetZkAdapterCode(ZE_OK, zk_errno);
    return true;
}

void ZooKeeperLock::OnWatchNodeDeleted(const std::string& path) {
    pthread_mutex_lock(&mutex_);
    if (IsAcquired()) {
        pthread_mutex_unlock(&mutex_);
        return;
    }
    if (watch_path_.compare(path) != 0) {
        pthread_mutex_unlock(&mutex_);
        return;
    }
    PRINTF_INFO("node [%s] is deleted\n", path.c_str());

    int zk_ret = ZE_OK;
    node_list_.pop();
    while (!node_list_.empty()) {
        watch_path_ = lock_path_ + "/" + node_list_.top().name;
        bool is_exist;
        if (!adapter_->CheckAndWatchExistForLock(watch_path_, &is_exist,
                                                  &zk_ret)) {
            pthread_mutex_unlock(&mutex_);
            callback_func_(lock_path_, zk_ret, callback_param_);
            return;
        }
        if (is_exist) {
            pthread_mutex_unlock(&mutex_);
            PRINTF_INFO("watch next node [%s]\n", watch_path_.c_str());
            return;
        } else {
            PRINTF_INFO("next node [%s] dead, skip\n", watch_path_.c_str());
            node_list_.pop();
        }
    }

    is_acquired_ = true;
    PRINTF_INFO("get lock success\n");
    callback_func_(lock_path_, zk_ret, callback_param_);
}

LockCompletion::LockCompletion()
    : lock_(NULL), errno_(ZE_OK) {
    pthread_mutex_init(&mutex_, NULL);
    pthread_cond_init(&cond_, NULL);
}

LockCompletion::~LockCompletion() {
    pthread_mutex_destroy(&mutex_);
    pthread_cond_destroy(&cond_);
}

void LockCompletion::SetLock(ZooKeeperLock * lock) {
    lock_ = lock;
}

bool LockCompletion::Wait(int* zk_errno, const timeval * end_time) {
    pthread_mutex_lock(&mutex_);
    while (1) {
        if (lock_->IsAcquired()) {
            pthread_mutex_unlock(&mutex_);
            SetZkAdapterCode(ZE_OK, zk_errno);
            return true;
        } else if (errno_ != ZE_OK) {
            pthread_mutex_unlock(&mutex_);
            SetZkAdapterCode(errno_, zk_errno);
            return false;
        } else if (end_time != NULL) {
            struct timespec abs_time;
            abs_time.tv_sec = end_time->tv_sec;
            abs_time.tv_nsec = end_time->tv_usec * 1000;
            int err = pthread_cond_timedwait(&cond_, &mutex_, &abs_time);
            if (err == ETIMEDOUT && !lock_->IsAcquired() && errno_ == ZE_OK) {
                pthread_mutex_unlock(&mutex_);
                SetZkAdapterCode(ZE_LOCK_TIMEOUT, zk_errno);
                return false;
            }
        } else {
            pthread_cond_wait(&cond_, &mutex_);
        }
    }
}

void LockCompletion::Signal(int err) {
    pthread_mutex_lock(&mutex_);
    errno_ = err;
    pthread_cond_signal(&cond_);
    pthread_mutex_unlock(&mutex_);
}

} // namespace zk
} // namespace mytera
} // namespace bubblefs