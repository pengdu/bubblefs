// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>
// Created: 05/31/11

// toft/system/threading/thread_types.cpp

#include "platform/toft_system_threading_thread_types.h"
#include <string>
#include "platform/toft_system_check_error.h"

namespace bubblefs {
namespace mytoft {

ThreadAttributes& ThreadAttributes::SetName(const std::string& name) {
    m_name = name;
    return *this;
}

ThreadAttributes::ThreadAttributes() {
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_attr_init(&m_attr));
}

ThreadAttributes::~ThreadAttributes() {
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_attr_destroy(&m_attr));
}

ThreadAttributes& ThreadAttributes::SetStackSize(size_t size) {
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_attr_setstacksize(&m_attr, size));
    return *this;
}

ThreadAttributes& ThreadAttributes::SetDetached(bool detached) {
    int state = detached ? PTHREAD_CREATE_DETACHED : PTHREAD_CREATE_JOINABLE;
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_attr_setdetachstate(&m_attr, state));
    return *this;
}

ThreadAttributes& ThreadAttributes::SetPriority(int priority) {
    return *this;
}

bool ThreadAttributes::IsDetached() const {
    int state = 0;
    MYTOFT_CHECK_PTHREAD_ERROR(pthread_attr_getdetachstate(&m_attr, &state));
    return state == PTHREAD_CREATE_DETACHED;
}

} // namespace mutoft
} // namespace bubblefs