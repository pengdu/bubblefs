// Copyright (c) 2011, The Toft Authors.
// All rights reserved.
//
// Author: CHEN Feng <chen3feng@gmail.com>

// toft/system/event_dispatcher/event_dispatcher.cpp

#include "platform/toft_system_event_dispatcher_event_dispatcher.h"

#include "ev++.h" // libev

namespace bubblefs {
namespace mytoft {

EventDispatcher::EventDispatcher() : m_loop(ev_loop_new()) {
}

EventDispatcher::~EventDispatcher() {
    ev_loop_destroy(m_loop);
}

} // namespace mytoft
} // namespace bubblefs